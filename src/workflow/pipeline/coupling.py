import datajoint as dj
import numpy as np
import sys
from datetime import timedelta
from element_array_ephys.ephys_no_curation import map_channel_to_electrode, get_probe_type
from tensorpac import Pac, PreferredPhase, EventRelatedPac
from tensorpac.utils import ITC, PeakLockedTF

from workflow import DB_PREFIX
from workflow.pipeline import analysis, mua
from .ephys import ephys, probe

schema = dj.schema(DB_PREFIX + "coupling")


"""
STTFA (power spectrum based)
"""
@schema
class STTFA(dj.Computed):
    """
    Spike-Triggered Time-Frequency Analysis (STTFA) for each electrodes. Shows the impact of spikes on LFP spectral power.
    """

    definition = """
    -> analysis.LFPSpectrogram.ChannelSpectrogram
    ---
    spike_count: int # number of spikes
    a_sttfa: longblob  # average frequency power during spike-triggered time window
    r_sttfa: longblob # randomized STTFA (random spike times)
    n_sttfa: longblob # normalized STTFA (log(STTFA) - log(rSTTFA))
    frequency: longblob  # frequency values
    """

    @property
    def key_source(self): # only process sessions with all MUA spikes processed

        min_spikes = 10

        lfp_table = dj.U("organoid_id", "start_time", "end_time").aggr(analysis.LFPSpectrogram)

        electrode_map = probe.ElectrodeConfig.Electrode.proj("channel_idx")

        valid_keys = []
        for lfp_key in lfp_table.fetch(as_dict=True):

            lfp_with_channel = ((analysis.LFPSpectrogram & lfp_key) * electrode_map).proj("channel_idx")

            mua_keys = (mua.MUASpikes
                        & f"organoid_id = '{lfp_key['organoid_id']}'"
                        & f"start_time >= '{lfp_key['start_time']}'"
                        & f"start_time < '{lfp_key['end_time']}'").fetch("KEY")

            summed_spikes_table = lfp_with_channel.aggr(
                (mua.MUASpikes.Channel & mua_keys).proj("spike_count", mua_start="start_time"),
                "channel_idx",
                total_spike_count="sum(spike_count)"
            )

            electrodes, total_spikes = summed_spikes_table.fetch("electrode", "total_spike_count")

            min_spikes_bool = total_spikes >= min_spikes
            for electrode in electrodes[min_spikes_bool]:
                valid_keys.append({
                    **lfp_key,
                    "electrode": electrode
                })

        return (
            analysis.LFPSpectrogram.ChannelSpectrogram
            & valid_keys
        )

    def make(self, key):

        # define parameters
        fs = 20000 # sampling frequency in Hz
        max_freq = 300 # Hz
        num_rand_iterations = 1000 # number of randomizations for rSTTFA

        # find the channel idx for the spectrogram electrode
        probe_type = get_probe_type(key)
        channel_idx = map_channel_to_electrode(probe_type, input_indices=np.array([key['electrode']]), electrode_to_channel=True)[0]

        # fetch MUA parameters within the spectrogram time window
        spike_indices, start_times = (mua.MUASpikes.Channel &
                                                    f"organoid_id='{key['organoid_id']}'" &
                                                    f"start_time BETWEEN '{key['start_time']}' AND '{key['end_time']}'" &
                                                    f"channel_idx = '{channel_idx}'"
                                                    ).fetch('spike_indices', 'start_time')

        # get array of all spike times (relative to frame start)
        start_ms = (start_times - key['start_time']).astype('timedelta64[ms]') / np.timedelta64(1, 'ms') # ms from frame start
        rel_spike_times_ms = spike_indices / fs / (np.timedelta64(1,'ms')/np.timedelta64(1,'s'))
        spike_times_ms = np.hstack(rel_spike_times_ms + start_ms).astype(int) # relative to spectrogram start time

        # remove boundary spikes (account for MUA outside spectrogram time)
        num_ms = (key['end_time'] - key['start_time']) / timedelta(milliseconds=1)
        spike_times_ms = spike_times_ms[(0 <= spike_times_ms) & (spike_times_ms <= num_ms)]

        spike_count = len(spike_times_ms)

        # fetch spectrogram
        freq, time, spectrogram = (analysis.LFPSpectrogram.ChannelSpectrogram & key).fetch1('frequency', 'time', 'spectrogram')

        # convert time to ms
        time_ms = (time * 1000).astype(int)  # in ms
        times_array = np.arange(0, time_ms[-1] + 1)  # array of all ms time points within spectrogram duration

        # calculate STTFA
        spike_spec_bins = np.histogram(spike_times_ms, bins=np.concatenate([[0], time_ms]))[0].astype(bool) # binary array indicating which spectrogram time bins have spikes
        a_sttfa = np.mean(spectrogram[:, spike_spec_bins], axis=1)

        a_sttfa = a_sttfa[freq <= max_freq]
        frequency = freq[freq <= max_freq]

        # calculate randomized STTFA
        r_sttfa_list = []
        for _ in range(num_rand_iterations):
            rand_spike_times = np.random.choice(times_array, size=spike_count, replace=False)
            rand_spike_indices_spec_bins = np.histogram(rand_spike_times, bins=np.concatenate([[0], time_ms]))[0].astype(bool)
            r_sttfa = np.mean(spectrogram[:, rand_spike_indices_spec_bins], axis=1)
            r_sttfa_list.append(r_sttfa[freq <= max_freq])

        r_sttfa = np.mean(np.vstack(r_sttfa_list), axis=0)

        # calculate normalized STTFA
        n_sttfa = np.log10(a_sttfa) - np.log10(r_sttfa)  # shape: (frequency)

        # insert into table
        self.insert1(
            {
                **key,
                'spike_count': spike_count,
                'a_sttfa': a_sttfa,
                'r_sttfa': r_sttfa,
                'n_sttfa': n_sttfa,
                'frequency': frequency,
            }
        )


"""
Phase Amplitude Coupling
"""

@schema
class TensorpacParamset(dj.Lookup):
    """
    Parameters for phase-amplitude coupling analysis with Tensorpac.
    """

    definition = """
    tensorpac_param_idx: int # Unique identifier for the Tensorpac parameter set
    ---
    idpac: blob # tuple containing (PAC method, surrogate method, normalization method)
    dcomplex: varchar(16) # method for calculating complex phase ('wavelet' or 'hilbert')
    cycles: blob # number of cycles for wavelet convolution (phase, amplitude) (if dcomplex='wavelet')
    width: float # width of morlet wavelet (if dcomplex='wavelet')
    dt: float # time step in seconds for calculating PAC over time (length of epoch)
    """
    contents = [
        (0, (1, 3, 4), 'wavelet', np.nan, 7, 10)
    ]

@schema
class PhaseAmplitudeBands(dj.Lookup):
    """
    Manual insert of phase and amplitude frequency bands for coupling analysis.
    """

    definition = """
    band_param_idx: int # Unique identifier for the frequency band parameter set
    ---
    f_pha: longblob # phase frequencies of interest (see CREATE_new_coupling_session for details)
    f_amp: longblob # amplitude frequencies of interest (see CREATE_new_coupling_session for details)
    """
    contents = [
        (0,
         [[2, 4], [4, 6], [6, 8], [8, 10], [10, 12]],
         [[20, 25], [25, 30], [30, 35], [35, 40], [40, 45], [45, 50]]),
        (1,
         [[4, 6]],
         [[20, 30], [30, 40], [40, 50]])

    ]

@schema
class CouplingSession(dj.Manual):
    """
    Manual insert of Tensorpac coupling analysis (across entire session)
    """

    definition = """
    -> ephys.LFP.Trace
    -> TensorpacParamset
    -> PhaseAmplitudeBands
    """

@schema
class PhaseAmplitudeCoupling(dj.Computed):
    """
    Phase-amplitude coupling analysis using Tensorpac (tensorpac.Pac).
    """

    definition = """
    -> CouplingSession
    ---
    pha_vec: longblob # phase frequency bin (center frequencies)
    amp_vec: longblob # amplitude frequency bin (center frequencies)
    pac_array: longblob # PAC values for each phase-amplitude pair (shape: amp_vec, pha_vec, num_epochs)
    pvalues: longblob # p-values for each phase-amplitude pair (shape: amp_vec, pha_vec)
    """

    class PreferredPhase(dj.Part):
        """
        Phase in which amplitude is strongest for each phase-amplitude pair.
        """

        definition = """
        -> PhaseAmplitudeCoupling
        ---
        preferred_phase: longblob # preferred phase for each amplitude frequency (shape: amp_vec, pha_vec, num_epochs)
        vecbin: longblob # phase bins (-pi to pi) (shape: n_bins)
        amplitude_values: longblob # amplitude values for each phase bin (shape: n_bins, amp_vec, pha_vec, num_epochs)
        """

    def make(self, key):

        # fetch Tensorpac parameters
        idpac, dcomplex, cycles, width, dt = (TensorpacParamset & key).fetch1(
            "idpac", "dcomplex", "cycles", "width", "dt"
        )
        dcomplex = sys.intern(str(dcomplex))

        # fetch frequency bands
        f_pha, f_amp = (PhaseAmplitudeBands & key).fetch1("f_pha", "f_amp")

        # fetch data
        trace = (ephys.LFP.Trace & key).fetch1("lfp")
        fs = (ephys.LFP & key).fetch1("lfp_sampling_rate")

        # break data into epochs (data = num_epochs x epoch_len)
        epoch_len = int(fs * dt)  # number of samples in each epoch
        num_epochs = int(len(trace) // epoch_len)
        data = trace[:num_epochs * epoch_len].reshape((num_epochs, epoch_len)) # cut off excess samples if dt not perfectly divisible

        # initialize PAC object
        # define PAC object
        if dcomplex == 'wavelet':
            p = Pac(idpac=idpac, f_pha=f_pha, f_amp=f_amp, dcomplex=dcomplex, width=width, verbose=False)
        elif dcomplex == 'hilbert':
            p = Pac(idpac=idpac, f_pha=f_pha, f_amp=f_amp, dcomplex=dcomplex, cycles=cycles, verbose=False)
        else:
            raise ValueError(f"Invalid dcomplex method: {dcomplex}")

        # filter traces
        pha = p.filter(fs, data, ftype='phase', n_jobs=-1) # (num_phase, num_epochs, epoch_len)
        amp = p.filter(fs, data, ftype='amplitude', n_jobs=-1) # (num_amp, num_epochs, epoch_len)

        # fit pac to phase and amplitude
        pac = p.fit(pha, amp, n_perm=200, p=1, mcp='bonferroni', verbose=False) # (num_amp, num_phase, num_epochs)

        # extract pac metrics
        pha_vec = p.xvec  # phase frequency bin (center frequencies)
        amp_vec = p.yvec  # amplitude frequency bin (center frequencies)
        pvalues = p.pvalues  # average p-values across epochs (shape: num_amp, num_phase) (done with null distribution from permutations)

        # initialize preferred phase object
        if dcomplex == 'wavelet':
            pp = PreferredPhase(f_pha=f_pha, f_amp=f_amp, dcomplex=dcomplex, width=width, verbose=False)
        elif dcomplex == 'hilbert':
            pp = PreferredPhase(f_pha=f_pha, f_amp=f_amp, dcomplex=dcomplex, cycles=cycles, verbose=False)
        else:
            raise ValueError(f"Invalid dcomplex method: {dcomplex}")

        # fit preferred phase
        ampbin, preferred_phase, vecbin = pp.fit(pha, amp, n_bins=72)

        # insert into table
        self.insert1(
            {
                **key,
                'pha_vec': pha_vec,
                'amp_vec': amp_vec,
                'pac_array': pac,
                'pvalues': pvalues,
            }
        )
        self.PreferredPhase.insert1(
            {
                **key,
                'preferred_phase': preferred_phase,
                'vecbin': vecbin,
                'amplitude_values': ampbin,
            }
        )

@schema
class UnitCouplingSession(dj.Manual):
    """
    Manual insert for Tensorpac Event-based Phase-Amplitude Coupling analysis (coupling during specific single unit spikes).
    """

    definition = """
    -> ephys.CuratedClustering.Unit
    -> TensorpacParamset
    -> PhaseAmplitudeBands
    """

@schema
class EventBasedCoupling(dj.Computed):
    """
    Event-based phase-amplitude coupling analysis using Tensorpac (tensorpac.Pac). Shows coupling during specific single unit spikes.
    """

    definition = """
    -> UnitCouplingSession
    ---
    pha_vec: longblob # phase frequency bin (center frequencies)
    amp_vec: longblob # amplitude frequency bin (center frequencies)
    t: longblob # time vector relative to spike time
    erpac_array: longblob # Event-Related PAC values for each phase-amplitude pair (shape: amp_vec, pha_vec, time)
    """

    class InterTrialCoherence(dj.Part):
        """
        Inter-trial coherence (ITC) of phase for each phase frequency during specific single unit spikes.
        """

        definition = """
        -> EventBasedCoupling
        ---
        itc_array: longblob # ITC values for each phase frequency and time point (shape: pha_vec, time)
        """

    class PeakLockedTF(dj.Part):
        """
        Peak-locked time-frequency analysis for each phase-amplitude pair during specific single unit spikes.
        """

        definition = """
        -> EventBasedCoupling
        ---
        pltf_array: longblob # Peak-locked time-frequency amplitude for each phase-amplitude pair (shape: amp_vec, pha_vec, time)
        """

    def make(self, key):

        spike_buffer = .5 # seconds extract around spike times (+/-)
        edge = 10 # samples to trim from edges of extracted LFP epochs (to avoid edge artifacts in filtering) (only relavent for ITC)

        # make sure an ephys.LFP session exists for the same time window (will use LFP trace for coupling analysis)
        if not (ephys.LFP & key):
            raise ValueError(f"No corresponding LFP session found for {key} - cannot perform event-based coupling analysis")

        # fetch Tensorpac parameters
        idpac, dcomplex, cycles, width, dt = (TensorpacParamset & key).fetch1(
            "idpac", "dcomplex", "cycles", "width", "dt"
        )
        dcomplex = sys.intern(str(dcomplex))

        # fetch frequency bands
        f_pha, f_amp = (PhaseAmplitudeBands & key).fetch1("f_pha", "f_amp")

        # fetch single unit spike info
        electrode, spike_times, spike_sites = (ephys.CuratedClustering.Unit & key).fetch1("electrode", "spike_times", "spike_sites")
        spike_times = spike_times[spike_sites == electrode] # only use spikes from the electrode corresponding to the LFP trace (will be used for coupling analysis)

        # fetch LFP trace and sampling rate
        trace = (ephys.LFP.Trace & key & f"electrode = '{electrode}'").fetch1("lfp")
        lfp_fs = (ephys.LFP & key).fetch1("lfp_sampling_rate")

        # get time vector (for lfp trace)
        time_vector = np.arange(len(trace)) / lfp_fs

        # remove boundary spikes
        spike_times = spike_times[(spike_times > spike_buffer) & (spike_times < time_vector[-1] - spike_buffer)] # remove boundary spikes

        # insert empty arrays if no spikes remain after removing boundary spikes
        if len(spike_times) == 0:
            self.insert1(
                {
                    **key,
                    'pha_vec': np.array([]),
                    'amp_vec': np.array([]),
                    't': np.array([]),
                    'erpac_array': np.array([]),
                }
            )

            self.InterTrialCoherence.insert1(
                {
                    **key,
                    'itc_array': np.array([]),
                }
            )

            self.PeakLockedTF.insert1(
                {
                    **key,
                    'pltf_array': np.array([]),
                }
            )
            return

        # loop through spikes and extract lfp segments
        lfp_epochs_itc = []
        n_samples = int(spike_buffer * lfp_fs) + edge # number of samples to extract on either side of spike time (accounting for edge trimming)
        for spike_time in spike_times:

            # find nearest index in lfp trace to spike time
            spike_idx = np.argmin(np.abs(time_vector - spike_time))

            # get lfp segment around spike time (accounting for spike_buffer)
            start_idx = spike_idx - n_samples
            end_idx = spike_idx + n_samples
            lfp_segment = trace[start_idx:end_idx]

            lfp_epochs_itc.append(lfp_segment)
        lfp_epochs_itc = np.array(lfp_epochs_itc)

        # remove edge samples from epochs (for non itc analysis)
        lfp_epochs = lfp_epochs_itc[:, edge:-edge]

        # initialize preferred phase object
        if dcomplex == 'wavelet':
            p = EventRelatedPac(f_pha=f_pha, f_amp=f_amp, dcomplex=dcomplex, width=width, verbose=False)
        elif dcomplex == 'hilbert':
            p = EventRelatedPac(f_pha=f_pha, f_amp=f_amp, dcomplex=dcomplex, cycles=cycles, verbose=False)
        else:
            raise ValueError(f"Invalid dcomplex method: {dcomplex}")

        # extract phases and amplitudes
        pha = p.filter(lfp_fs, lfp_epochs, ftype='phase')
        amp = p.filter(lfp_fs, lfp_epochs, ftype='amplitude')

        # compute the erpac
        erpac_array = p.fit(pha, amp, method='gc', smooth=50)

        # get pvalues
        pha_vec = p.xvec  # phase frequency bin (center frequencies)
        amp_vec = p.yvec  # amplitude frequency bin (center frequencies)
        t = np.arange(-spike_buffer, spike_buffer, 1/lfp_fs) # time vector relative to spike time

        # ITC analysis
        if dcomplex == 'wavelet':
            itc = ITC(lfp_epochs_itc, sf=lfp_fs, f_pha=f_pha, dcomplex=dcomplex, width=width, edges=edge, verbose=False)
        elif dcomplex == 'hilbert':
            itc = ITC(lfp_epochs_itc, sf=lfp_fs, f_pha=f_pha, dcomplex=dcomplex, cycle=cycles, edges=edge, verbose=False)
        else:
            raise ValueError(f"Invalid dcomplex method: {dcomplex}")

        itc_array = itc.itc

        # PeakLockedTF analysis
        cue = 0
        pltf_array = []
        for phase_band in f_pha:
            pltf = PeakLockedTF(lfp_epochs, lfp_fs, cue, times=t, f_pha=phase_band, f_amp=f_amp, verbose=False)
            pltf_array.append(pltf.amp_a)
        pltf_array = np.stack(pltf_array, axis=1) # shape = amp_vec, pha_vec, num_spikes, time

        # average across spikes
        pltf_array = np.mean(pltf_array, axis=2) # shape = amp_vec, pha_vec, time

        # insert into table
        self.insert1(
            {
                **key,
                'pha_vec': pha_vec,
                'amp_vec': amp_vec,
                't': t,
                'erpac_array': erpac_array,
            }
        )

        self.InterTrialCoherence.insert1(
            {
                **key,
                'itc_array': itc_array,
            }
        )

        self.PeakLockedTF.insert1(
            {
                **key,
                'pltf_array': pltf_array,
            }
        )
