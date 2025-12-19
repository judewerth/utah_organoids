# Import Modules
import datajoint as dj
from workflow import DB_PREFIX, ORG_NAME, WORKFLOW_NAME
from workflow.pipeline import culture, ephys, mua, probe, analysis
from element_interface.utils import find_full_path
from element_array_ephys.ephys_no_curation import get_ephys_root_data_dir

import os
import intanrhdreader
import numpy as np
from random import randint
import bottleneck as bn
from scipy.signal import find_peaks, coherence, butter, sosfiltfilt, hilbert
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
from specparam import SpectralModel
from datetime import datetime, timezone
import matplotlib.pyplot as plt
import plotly.tools as tls
import plotly.io as pio


"""
frame.py 
"""

# Set up schema (connects to database and manages table creation)
schema = dj.schema(DB_PREFIX + "frame")
logger = dj.logger

dj.config["stores"]["datajoint-blob"] = dict(
    protocol="s3",
    endpoint="s3.amazonaws.com:9000",
    bucket="dj-sciops",
    location=f"{ORG_NAME}_{WORKFLOW_NAME}/datajoint/blob",
    access_key=os.getenv("AWS_ACCESS_KEY", None),
    secret_key=os.getenv("AWS_ACCESS_SECRET", None),
)


# Define Lookup Tables
@schema
class NumElectrodesInside(dj.Lookup):
    """
    Number of electrodes inside the organoid for each session (defined by organoid images).
    """

    definition = """
    organoid_id: varchar(4) # e.g. O17
    ---
    num_electrodes: int  # Number of electrodes inside the organoid
    """
    contents = [
        ("O09", 32), ("O10", 16), ("O11", 20), ("O12", 14), # Control Batch 1
        ("O13", 25), ("O14", 13), ("O15", 11), ("O16", 11), # Control Batch 2
        ("O17", 22), ("O18", 19), ("O19", 20), ("O20", 17), # GBM Batch 1
        ("O21", 18), ("O22", 21), ("O23", 22), ("O24", 23), # Control Batch 3
        ("O25", 20), ("O26", 32), ("O27", 26), ("O28", 24), # GBM Batch 2
        ("O29", 22), ("O30", 20), ("O31", 20), ("O32", 20), # Control Batch 4
    ]

@schema
class TimeFrameParamset(dj.Lookup):
    """
    Time frame extraction parameters for LFP and spike analyses. 
    """

    definition = """
    frame_param_idx: int  # Unique identifier for the frame parameter set
    ---
    num_frames: int  # Number of frames to extract
    min_per_frame: int  # Length of each frame in minutes
    """
    contents = [
        (1, 12, 5),  # 12 frames at 5 minutes each (1 hour total)
        (2, 4, 15), # 4 frames at 15 minutes each (1 hour total)
        (3, 8, 15), # 8 frames at 15 minutes each (2 hours total)
    ]

@schema
class FrameSession(dj.Manual):
    """
    Manually define time boundaries for analysis of each organoid session.
    """

    definition = """ 
    -> culture.Experiment
    start_boundary     : datetime # Start datetime for analysis
    end_boundary       : datetime # End datetime for analysis
    frame_param_idx   : int     # Reference to TimeFrameParamset
    """

@schema
class FrameAnalysis(dj.Computed):
    """
    Compute active time frames within defined analysis boundaries for the defined organoid session.
    """
    definition = """
    -> FrameSession
    """

    class ActiveTimeFrames(dj.Part):
        """
        Identify the "num_frames" most active time frames (length "sec_per_frame") within the analysis boundaries for each organoid session.
        """

        definition = """
        -> master
        frame_start: datetime # Start of active time frame
        frame_end: datetime   # End of active time frame
        ---
        frame_firing_rate: float # Firing rates for each frame
        """

    def make(self, key):
        
        # fetch electrode parameters
        num_elec_inside = (NumElectrodesInside & key).fetch1('num_electrodes')

        # fetch frame parameters
        num_frames, min_per_frame = (TimeFrameParamset & key).fetch1('num_frames', 'min_per_frame')

        # fetch MUA values (needs to be previously )
        spike_rates, start_times, channel_ids = (mua.MUASpikes.Channel & 
                                                 f"organoid_id='{key['organoid_id']}'" &
                                                 f"start_time BETWEEN '{key['start_boundary']}' AND '{key['end_boundary']}'"
                                                 ).fetch('spike_rate', 'start_time', 'channel_idx')
        
        # convert channel ids to electrode indices
        electrode_ids = map_channel_to_electrode(channel_ids)

        time_vector, population_firing_vector = create_population_firing_vector(spike_rates, start_times, electrode_ids, num_elec_inside)

        # filter population firing vector - boxcar with the length of min_per_frame
        population_firing_vector = bn.move_mean(population_firing_vector, window=min_per_frame, min_count=1)

        # find active frames
        active_frames = find_active_frames(start_times, time_vector, population_firing_vector, num_frames, min_per_frame)

        # find probe info
        port_id = set((ephys.EphysSessionProbe & key).fetch("port_id"))
            # Figure out `Port ID` from the existing EphysSession
        if not (ephys.EphysSessionProbe & key):
            raise ValueError(
                f"No EphysSessionProbe found for the {key} - cannot determine the port ID"
            )
        # Check if there are multiple port IDs for the same experiment, if so, it needs to be fixed in the EphysSessionProbe table
        if len(port_id) > 1:
            raise ValueError(
                f"Multiple Port IDs found for the {key} - cannot determine the port ID"
            )
        port_id = port_id.pop

        probe_name = set((ephys.EphysSessionProbe & key).fetch("probe"))
        # Check if there are multiple port IDs for the same experiment, if so, it needs to be fixed in the EphysSessionProbe table
        if len(probe_name) > 1:
            raise ValueError(
                f"Multiple Probes found for the {key} - cannot determine the probe name"
            )
        probe_name = probe_name.pop

        # insert the parent FrameAnalysis record
        self.insert1(key)

        # insert active frames (and acompannying ephys sessions)
        for active_frame in active_frames:
            # insert into lfp ephys session (needed for downstream analyses)
            # ephys.EphysSession.insert1({
            #     "organoid_id": key['organoid_id'],
            #     "experiment_start_time": key['experiment_start_time'],
            #     "insertion_number": 1,
            #     "start_time": active_frame['frame_start'],
            #     "end_time": active_frame['frame_end'],
            #     "session_type": "both"
            # }, skip_duplicates=True)
            # ephys.EphysSessionProbe.insert1({
            #     "organoid_id": key['organoid_id'],
            #     "experiment_start_time": key['experiment_start_time'],
            #     "insertion_number": 1,
            #     "start_time": active_frame['frame_start'],
            #     "end_time": active_frame['frame_end'],
            #     "probe": probe,
            #     "port_id": port_id,
            #     "used_electrodes": []
            # }, skip_duplicates=True)


            # insert into frame table
            self.ActiveTimeFrames.insert1({
                **key,
                'frame_start': active_frame['frame_start'],
                'frame_end': active_frame['frame_end'],
                'frame_firing_rate': active_frame['frame_firing_rate'],
            })            


# Helpful Functions
def map_channel_to_electrode(channel_ids):

    electrode_mapping, channel_mapping = probe.ElectrodeConfig.Electrode.fetch("electrode", "channel_idx")

    # create lookup to convert
    lookup = np.empty(32, dtype=int)
    lookup[channel_mapping] = electrode_mapping

    # correctly map electrode indices
    electrode_ids = lookup[channel_ids]
    
    return electrode_ids

def create_population_firing_vector(spike_rates, start_times, electrode_ids, num_elec_inside):

    # create full time vector from recording start to end (1 minute increments)
    unique_start_times = np.unique(start_times)
    time_vector = np.arange(min(unique_start_times.astype("datetime64[m]")), max(unique_start_times.astype("datetime64[m]"))+np.timedelta64(1, 'm'), np.timedelta64(1, 'm')) # full array of recording timeline (needed to account for missing data)
    population_firing_vector = np.zeros(time_vector.shape)    

    # loop through start times and insert data into population firing vector
    for start_time in unique_start_times:

        time_bool = (start_times == start_time)

        # only consider electrodes inside organoid
        elec_bool = (electrode_ids < num_elec_inside)

        # sum valid electrodes for each time window (minute)
        time_index = np.where(time_vector == np.datetime64(start_time, 'm'))[0][0]
        population_firing_vector[time_index] = np.sum(spike_rates[time_bool & elec_bool])
    
    return time_vector, population_firing_vector

def find_active_frames(start_times, time_vector, population_firing_vector, num_frames, min_per_frame):

    # find active frames
    frame_indices, properties = find_peaks(population_firing_vector, height=0, distance=min_per_frame)
    frame_amplitudes = properties['peak_heights']

    # remove boundary peaks (these will raise an error when trying to extract burst windows)
    boundary_bool = (min_per_frame <= frame_indices)

    frame_indices = frame_indices[boundary_bool]
    frame_amplitudes = frame_amplitudes[boundary_bool]
    
    # find most active regions -> extract windows
    active_frame_indices = frame_indices[np.argsort(frame_amplitudes)[-num_frames:]]  # indexes of the most active peaks

    # account if not enough peaks -> use random windows
    while len(active_frame_indices) < num_frames:

        rand_idx = randint(0, len(population_firing_vector))

        if not np.any(np.isin(np.arange(rand_idx-min_per_frame, rand_idx+1), active_frame_indices)):
            active_frame_indices = np.concatenate([active_frame_indices, [rand_idx]])
    
    active_frames = []
    for active_frame_idx in active_frame_indices:
        
        # determine frame boundaries
        frame_bounds = np.array([-min_per_frame, 0]) + active_frame_idx

        # find frame metrics
        start_time, end_time = np.unique(start_times[np.isin(start_times.astype("datetime64[m]"), time_vector[frame_bounds])])
        frame_firing_rate = np.mean(population_firing_vector[frame_bounds[0]:frame_bounds[1]])

        # extract frame info
        active_frames.append(
            {
                'frame_start': start_time,
                'frame_end': end_time,
                'frame_firing_rate': frame_firing_rate
            }
        )
    
    return active_frames

"""
ephys.Impedance
"""
@schema
class ImpedanceFile(dj.Manual):
    definition = """ # Insert files and organoid_id for impedance measurements
    -> ephys.EphysRawFile
    organoid_id        : varchar(4) # e.g. O17
    """

@schema
class ImpedanceMeasurements(dj.Imported):
    definition = """ # Store impedance measurements per channel
    -> ImpedanceFile
    ---
    port_id: char(2)  # Port ID of the Intan acquisition system
    """

    class Channel(dj.Part):
        definition = """
        -> master
        channel_idx: int  # channel index
        ---
        channel_id: varchar(64)  # channel id
        impedance_magnitude: float  # in Ohms
        impedance_phase: float  # in Degrees
        """

    def make(self, key):
        # fetch file path from ephysrawfile entry
        file_path = (ephys.EphysRawFile & key).fetch1("file_path")

        # import file
        file = find_full_path(get_ephys_root_data_dir(), file_path)
        try:
            data = intanrhdreader.load_file(file)
        except OSError:
            raise OSError(f"OS error occurred when loading file {file.name}")

        # extract amplifier channels
        amplifier_channels = data['header'].pop("amplifier_channels")

        # Figure out `Port ID` from the existing EphysSessionProbe
        port_id = set((ephys.EphysSessionProbe & key).fetch("port_id"))

        # Figure out `Port ID` from the existing EphysSession
        if not (ephys.EphysSessionProbe & key):
            raise ValueError(
                f"No EphysSessionProbe found for the {key} - cannot determine the port ID"
            )

        # Check if there are multiple port IDs for the same experiment, if so, it needs to be fixed in the EphysSessionProbe table
        if len(port_id) > 1:
            raise ValueError(
                f"Multiple Port IDs found for the {key} - cannot determine the port ID"
            )
        port_id = port_id.pop()

        # get channels for the correct port
        port_channels = [channel for channel in amplifier_channels if channel['port_prefix'] == port_id]

        # insert into master
        self.insert1(
            {
                **key,
                "port_id": port_id,
            }
        )

        # loop through channels and insert impedance data
        for channel in port_channels:

            channel_idx = channel['custom_order']
            channel_id = channel['custom_channel_name']
            impedance_magnitude = channel['electrode_impedance_magnitude']
            impedance_phase = channel['electrode_impedance_phase']

            self.Channel.insert1(
                {
                    **key,
                    "channel_idx": channel_idx,
                    "channel_id": channel_id,
                    "impedance_magnitude": impedance_magnitude,
                    "impedance_phase": impedance_phase,
                }
            )

"""
analysis.Coherence
"""

@schema
class Coherence(dj.Computed):
    """
    Compute pairwise coherence between electrodes within an active time frame.
    """

    definition = """
    -> ephys.EphysSessionProbe
    ---
    execution_duration: float  # Time taken to compute coherence (in minutes)
    """

    class Connectivity(dj.Part):
        """
        Pairwise coherence between electrodes (LFP signals).
        """
        definition = """
        -> master
        electrode_a: int  # Electrode in coherence calculation
        electrode_b: int  # Electrode in coherence calculation
        ---
        f: longblob  # Frequency values
        coherence: longblob  # Coherence values between electrode A and B
        """
    
    class Synchrony(dj.Part):
        """
        Coherence between each electrode LFP signal to each frequency band signal
        """
        definition = """
        -> master
        -> analysis.SpectralBand
        electrode: int  # Electrode index
        ---
        f: longblob  # Frequency values
        synchrony: longblob  # Coherence between electrode LFP and frequency band signal
        """

    @property
    def key_source(self):
        
        # find the number of electrodes for each probe type
        num_electrodes = dj.U('probe_type').aggr(
            probe.ElectrodeConfig.Electrode,
            num_electrodes='count(distinct electrode)'
        )

        # find the number of processed electrodes (lfp.trace) for each ephys session
        processed_electrodes = ephys.EphysSessionProbe.aggr(
            ephys.LFP.Trace,
            processed_electrodes='count(distinct electrode)'
        )

        query = ephys.EphysSessionProbe  * num_electrodes * processed_electrodes

        # only process sessions where processed_electrodes matches num_electrodes or the length of used_electrodes
        # loop through incomplete sessions and check if used_electrodes length matches processed_electrodes 
        incomplete_sessions = query & ('processed_electrodes < num_electrodes')
        valid_keys = [key for wanted_electrodes, processed_electrodes, key in zip(*incomplete_sessions.fetch("used_electrodes", "processed_electrodes", "KEY")) if len(wanted_electrodes) == processed_electrodes]
        # add keys where all electrodes are processed
        valid_keys.extend((query & ('processed_electrodes = num_electrodes')).fetch("KEY"))
        
        return (
            ephys.EphysSessionProbe 
            & valid_keys
#             & "insertion_number = '1'" # option so it only processes automatic insertions
        )
        
    def make(self, key):

        execution_time = datetime.now(timezone.utc)

        # define parameters
        fs = 2500
        max_freq = 200 # Hz
        tw = 1
        nperseg = int(tw*fs) # samples per window

        # fetch traces
        traces = (ephys.LFP.Trace & key).fetch("lfp", order_by="electrode")

        # define synchronny parameters
        order = 4
        nyquist = fs/2

        # apply low pass filter to each electrode trace
        lfp_traces = []
        for trace in traces:
            sos = butter(order, np.array([1, max_freq])/nyquist, btype='bandpass', output='sos')
            filtered = sosfiltfilt(sos, trace)

            lfp_traces.append(filtered)
        lfp_traces = np.array(lfp_traces)

        num_elec = lfp_traces.shape[0]

        # insert into parent table
        self.insert1(
            {
                **key,
                "execution_duration": 0, # placeholder
            }
        )

        """ 
        Connectivity Analysis
        """

        # loop through electrodes and find coherence between adjacent electrode pairs
        
        for electrode_A in range(num_elec):
            for electrode_B in range(num_elec):

                # skip duplicate electrode pairings
                if electrode_A >= electrode_B:
                    continue
                
                # get traces
                el_A_trace = lfp_traces[electrode_A, :]
                el_B_trace = lfp_traces[electrode_B, :]

                # compute coherence
                f, Cxy = coherence(el_A_trace, el_B_trace, fs=fs, nperseg=nperseg)

                # remove frequencies greater than max_freq
                frequencies = f[f <= max_freq]
                connectivity = Cxy[f <= max_freq]

                # insert into part table
                self.Connectivity.insert1({
                    **key,
                    'electrode_a': electrode_A,
                    'electrode_b': electrode_B,
                    'f': frequencies,
                    'coherence': connectivity,
                })
        
        """
        Synchrony Analysis
        """

        # loop through electrodes and find coherence between lfp signal and freq bands
        for elec in range(num_elec):
     
            # get traces
            elec_trace = lfp_traces[elec, :]

            # loop through frequency bands and calculate coherence
            for band in (analysis.SpectralBand()).fetch(as_dict=True):

                # get signal of specific frequency band
                freq_cutoff = np.array([band['lower_freq']-1, band['upper_freq']+1]) # includes 1 Hz buffer
                if freq_cutoff[0] < 1:
                    freq_cutoff[0] = 1
                sos = butter(order, freq_cutoff/nyquist, btype='bandpass', output='sos')
                filtered = sosfiltfilt(sos, elec_trace)

                # get magnitude of hilbert transform (doing instead of morlet wavelets)
                hilbert_signal = hilbert(filtered)
                freq_power_signal = np.abs(hilbert_signal) ** 2

                # find coherence between original signal and the power signal (for each frequency)
                f, Cxy = coherence(elec_trace, freq_power_signal, fs=fs, nperseg=nperseg)
                # remove frequencies greater than max_freq
                frequencies = f[f <= max_freq]
                synchrony = Cxy[f <= max_freq]

                # insert into part table
                self.Synchrony.insert1({
                    **key,
                    'band_name': band['band_name'],
                    'electrode': elec,
                    'f': frequencies,
                    'synchrony': synchrony,
                })
        
        # update execution duration
        self.update1(
            {
                **key,
                "execution_duration": (
                    datetime.now(timezone.utc) - execution_time
                ).total_seconds()
                / 60,
            }
        )

"""
analysis.FOOOF
"""

@schema
class FOOOFParamset(dj.Lookup):
    """
    FOOOF parameter sets for spectral fitting.
    """

    definition = """ 
    fooof_param_idx: int  # Unique identifier for the FOOOF parameter set
    ---
    peak_width_limits: blob  # Lower and upper bounds on peak widths in Hz. e.g. [1, 12]
    max_n_peaks: int         # Maximum number of peaks the model can fit. e.g. 6
    min_peak_height: float   # Minimum absolute height of a peak above the aperiodic component. e.g. 0.1
    peak_threshold: float    # Relative threshold that candidate peaks must exceed to be included. e.g. 2.0
    aperiodic_mode: varchar(16) # Form of the aperiodic fit. e.g. 'fixed', 'knee'
    """

    contents = [
        (0, [1, 12], 6, 0.1, 2.0, 'fixed'),
        (1, [5, 12], 3, .05, 3.5, 'fixed'),
    ]

@schema
class FOOOFSession(dj.Manual):
    """
    Manual insert of FOOOF sessions for spectral fitting.
    """

    definition = """ 
    -> ephys.EphysSession
    -> FOOOFParamset
    start_freq: float  # Start frequency for FOOOF fitting
    end_freq: float   # End frequency for FOOOF fitting
    ---
    analysis_electrodes: blob # List of electrodes to perform FOOOF analysis on (will be averaged)
    """

@schema
class FOOOFAnalysis(dj.Computed):
    """
    Compute FOOOF spectral fitting for defined electrodes within an ephys session.
    """

    definition = """ 
    -> FOOOFSession
    spec_param_idx: int  # Reference to SpectrogramParamset
    ---
    plot: longblob  # Plot of FOOOF fit (as json)
    aperiodic_offset: float  # Aperiodic offset
    aperiodic_knee: float   # Aperiodic knee
    aperiodic_exponent: float  # Aperiodic exponent
    peak_center_frequencies: longblob  # Center frequencies of detected peaks
    peak_powers: longblob      # Power of detected peaks (above aperiodic fit)
    peak_bandwidths: longblob  # Bandwidths of detected peaks
    error: float           # Error of FOOOF fit (RMSE)
    r_squared: float      # R^2 of FOOOF fit
    """

    def make(self, key):

        # fetch electrodes to analyze
        analysis_electrodes = (FOOOFSession & key).fetch1("analysis_electrodes")

        # fetch spectrogram parameters
        spec_param_idx = (analysis.LFPSpectrogram & key).fetch("param_idx")[0]

        # fetch lfp spectrograms
        spectrograms = (analysis.LFPSpectrogram.ChannelSpectrogram & key & f"electrode IN {tuple(analysis_electrodes)}").fetch("spectrogram")
        spectrograms = np.stack(spectrograms, axis=-1)  # shape: (frequency, time, electrodes)
        mean_spectrum = np.mean(spectrograms, axis=(1, 2))  # shape: (frequency,)

        # fetch frequency vector
        frequency = (analysis.LFPSpectrogram.ChannelSpectrogram & key & f"electrode IN {tuple(analysis_electrodes)}").fetch("frequency")[0]

        # fetch fooof parameters
        peak_width_limits, max_n_peaks, min_peak_height, peak_threshold, aperiodic_mode = (FOOOFParamset & key).fetch1(
            "peak_width_limits", "max_n_peaks", "min_peak_height", "peak_threshold", "aperiodic_mode"
        )

        # fetch fooof session parameters
        start_freq, end_freq = (FOOOFSession & key).fetch1(
            "start_freq", "end_freq"
        )

        # initialize model
        fm = SpectralModel(
            peak_width_limits=peak_width_limits,
            max_n_peaks=max_n_peaks,
            min_peak_height=min_peak_height,
            peak_threshold=peak_threshold,
            aperiodic_mode=aperiodic_mode
        )

        # interpolate mean_spectrum to account for 60 Hz line noise removal
        notch_freqs = np.arange(0, frequency.max(), 60)
        # create mask for frequencies to remove
        mask = np.ones_like(frequency, dtype=bool)
        for notch_freq in notch_freqs:
            freq_mask = (notch_freq - 5 <= frequency) & (frequency <= notch_freq + 5)
            mask = mask & (~freq_mask)
        # interpolate
        interp_func = interp1d(frequency[mask], mean_spectrum[mask], kind='linear', fill_value="extrapolate")
        interp_mean_spectrum = interp_func(frequency)

        # fit model
        fm.fit(frequency, interp_mean_spectrum, freq_range=(start_freq, end_freq))

        # extract parameters
        # Aperiodic
        aperiodic = fm.get_params('aperiodic_params')
        aperiodic_offset = aperiodic[0]
        aperiodic_exponent = aperiodic[-1]  # works for both fixed & knee
        aperiodic_knee = aperiodic[1] if aperiodic_mode == 'knee' else 0.0  # optional

        # Peaks (safe for 0 peaks)
        peak_params = fm.get_params('peak_params')  # shape: (n_peaks, 3)
        peak_center_frequencies = peak_params[:, 0].tolist() if peak_params.size else []
        peak_powers = peak_params[:, 1].tolist() if peak_params.size else []
        peak_bandwidths = peak_params[:, 2].tolist() if peak_params.size else []

        # Fit metrics
        error = fm.get_params("metrics")["error_mae"]
        r_squared = fm.get_params("metrics")["gof_rsquared"]

        # generate plot
        fm.plot()
        mpl_fig = plt.gcf()
        plotly_fig = tls.mpl_to_plotly(mpl_fig)
        json_fig = pio.to_json(plotly_fig)

        # insert into table
        self.insert1(
            {
                **key,
                "spec_param_idx": spec_param_idx,
                "plot": json_fig,
                "aperiodic_offset": aperiodic_offset,
                "aperiodic_knee": aperiodic_knee,
                "aperiodic_exponent": aperiodic_exponent,
                "peak_center_frequencies": peak_center_frequencies,
                "peak_powers": peak_powers,
                "peak_bandwidths": peak_bandwidths,
                "error": error,
                "r_squared": r_squared,
            }
        )

"""
mua.PopulationBursts
"""
@schema
class BurstDetectionParamset(dj.Lookup):
    """
    Parameters for burst detection with multi-unit population activity.
    """

    definition = """
    burst_param_idx: int # Unique identifier for the burst detection parameter set
    ---
    gaus_len_ms: int # Gaussian kernel length in milliseconds
    boxcar_len_ms: int # Boxcar kernel length in milliseconds
    detection_threshold: float # Threshold for burst detection in standard deviations
    min_distance_ms: float # Minimum distance between bursts in milliseconds
    """
    contents = [
        (1, 100, 20, 2.0, 1000.0), # Parameters used in Sharf et al. 2021
    ]

@schema 
class BurstSession(dj.Manual):
    """
    Manual insert of burst detection sessions for population burst analysis.
    """

    definition = """ 
    -> ephys.EphysSession
    -> BurstDetectionParamset
    """

@schema
class PopulationBursts(dj.Computed):
    """
    Detect population bursts within an active time frame using specified burst detection parameters.
    """

    definition = """
    -> BurstSession
    ---
    burst_indices: longblob # Indices of detected bursts within the time frame
    burst_peak_heights: longblob # Peak heights of detected bursts
    burst_bounds: longblob # Start and end indices of detected bursts (firing rate >= 10%% of peak height)
    burst_spike_array: longblob # Single electrode spike array for each burst (num_bursts x num_electrodes x time_window)
    """

    def make(self, key):

        # define parameters
        fs = 20000 # sampling frequency in Hz
        burst_extract_dur = np.timedelta64(1, 's') # time for extracting burst spike array (+ and - from peak)
        burst_bound_thresh = 0.1 # threshold for defining burst bounds (percentage of peak height)

        # Fetch MUA parameters within the frame
        spike_indices, start_times, channel_ids = (mua.MUASpikes.Channel & 
                                                 f"organoid_id='{key['organoid_id']}'" &
                                                 f"start_time BETWEEN '{key['start_time']}' AND '{key['end_time']}'"
                                                 ).fetch('spike_indices', 'start_time', 'channel_idx')
        
        # check if we have spike indices for all times
        if len(np.unique(start_times)) < np.timedelta64(key['end_time'] - key['start_time'], 'm') / np.timedelta64(1,'m'):
            raise ValueError(f"Not all time windows have MUA spike data for {key} - cannot perform burst detection")

        # convert channel ids to electrode indices
        electrode_ids = map_channel_to_electrode(channel_ids)

        # get array of all spike times (relative to frame start)
        start_ms = (start_times - key['start_time']).astype('timedelta64[ms]') / np.timedelta64(1, 'ms') # ms from frame start
        rel_spike_times_ms = spike_indices / fs / (np.timedelta64(1,'ms')/np.timedelta64(1,'s')) 
        spike_times_ms = rel_spike_times_ms + start_ms

        # remove electrodes outside organoid
        num_elec_inside = (NumElectrodesInside & f"organoid_id='{key['organoid_id']}'").fetch1('num_electrodes')
        elec_bool = (electrode_ids < num_elec_inside)

        # create population spike time series (1 ms bins)
        time_bins = np.arange(0, np.timedelta64(key['end_time'] - key['start_time'], 'ms') / np.timedelta64(1, 'ms') + 1) # 1 ms bins
        population_spike_series, _ = np.histogram(np.hstack(spike_times_ms[elec_bool]), bins=time_bins)

        # convert spike series to firing rate
        population_firing_rate = population_spike_series * 1000 # convert to spikes per second

        # smooth firing rate with Gaussian and Boxcar kernels
        # fetch burst detection parameters
        gaus_len_ms, boxcar_len_ms, detection_threshold, min_distance_ms = (BurstDetectionParamset & key).fetch1(
            'gaus_len_ms', 'boxcar_len_ms', 'detection_threshold', 'min_distance_ms'
        )
        # boxcar kernel
        population_firing_rate = bn.move_mean(population_firing_rate, window=boxcar_len_ms, min_count=1)

        # Gaussian kernel 
        truncate = 4
        population_firing_rate = gaussian_filter1d(population_firing_rate, sigma=gaus_len_ms, truncate=truncate, mode="reflect")

        # detect spike bursts
        min_height = detection_threshold * np.std(population_firing_rate)
        
        # find peaks
        burst_indices, properties = find_peaks(population_firing_rate, height=min_height, distance=min_distance_ms)
        burst_peak_heights = properties['peak_heights']

        # find burst bounds (start and end indices where firing rate >= 10% of peak height)

        # define burst extraction parameters
        num_burst_samples = int(burst_extract_dur / np.timedelta64(1,'ms')) # number of samples to extract from burst peak (+ and -)
        
        # remove boundary bursts (will raise an error when extracting burst windows)
        boundary_bool = (num_burst_samples <= burst_indices) & (burst_indices <= (len(population_firing_rate)-num_burst_samples))
        burst_indices = burst_indices[boundary_bool]
        burst_peak_heights = burst_peak_heights[boundary_bool]

        # find burst windows and create spike array
        burst_windows = []
        burst_spike_array = np.zeros((len(burst_indices), num_elec_inside, 2*num_burst_samples), dtype=bool)
        for burst_idx, (index, height) in enumerate(zip(burst_indices, burst_peak_heights)):

            # extract burst waveform
            waveform = population_firing_rate[index-num_burst_samples : index+num_burst_samples]

            # find burst specific window threshold
            window_thresh = burst_bound_thresh * height
            window = np.array([0, 0])

            # find number of indices adjacent to the burst peak are over the burst threshold
            i = 1
            while (waveform[num_burst_samples-i] >= window_thresh) & (num_burst_samples-i > 0): # make sure it doesn't exceed the number of extracted samples
                window[0] -= 1 # indices before burst peak
                i += 1
            i = 1
            while (waveform[num_burst_samples+i] >= window_thresh) & (num_burst_samples+i < len(waveform)-1):
                window[1] += 1 # indices after burst peak
                i += 1        
            
            burst_windows.append(window)

            # fill in spike array for each electrode
            for elec_idx in range(num_elec_inside):

                # get spike times for electrode
                elec_spike_times = np.hstack(spike_times_ms[electrode_ids == elec_idx])

                # find spikes within burst window
                burst_spike_times = elec_spike_times[((index-num_burst_samples) <= elec_spike_times) & (elec_spike_times <= (index+num_burst_samples))]

                # convert to indices within burst spike array
                burst_spike_indices = (burst_spike_times - (index-num_burst_samples)).astype(int)
                burst_spike_array[burst_idx, elec_idx, burst_spike_indices] = True
        burst_bounds = np.array(burst_windows)

        # insert into table
        self.insert1({
            **key,
            'burst_indices': burst_indices,
            'burst_peak_heights': burst_peak_heights,
            'burst_bounds': burst_bounds,
            'burst_spike_array': burst_spike_array,
        })