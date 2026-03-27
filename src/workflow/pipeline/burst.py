import datajoint as dj
import numpy as np
import bottleneck as bn
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
from element_array_ephys.ephys_no_curation import map_channel_to_electrode, get_probe_type

from workflow import DB_PREFIX
from workflow.pipeline import culture, mua
from .ephys import ephys

schema = dj.schema(DB_PREFIX + "burst")


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
    Detect population bursts within a time frame using specified burst detection parameters.
    """

    definition = """
    -> BurstSession
    ---
    burst_indices: longblob # ms since start of detected bursts within the time frame
    burst_peak_heights: longblob # Peak heights of detected bursts
    burst_bounds: longblob # [-ms, +ms] relative to burst peak (firing rate >= 10%% of peak height)
    burst_spike_array: longblob # Single electrode spike array for each burst (num_bursts x num_electrodes x time_window)
    weighted_sttc: longblob # Spike time tiling coefficient across all electrodes weighted by number of spikes
    """

    def make(self, key):
        import neo
        import quantities as pq
        from elephant.spike_train_correlation import spike_time_tiling_coefficient

        # define parameters
        fs = 20000 # sampling frequency in Hz — Intan acquisition rate; hardcoded since MUASpikes.spike_indices are stored as raw sample indices at this rate and changing acquisition systems would require repopulating MUASpikes
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
        probe_type = get_probe_type(key)
        electrode_ids = map_channel_to_electrode(probe_type, input_indices=channel_ids)

        # get array of all spike times (relative to frame start)
        start_ms = (start_times - key['start_time']).astype('timedelta64[ms]') / np.timedelta64(1, 'ms') # ms from frame start
        rel_spike_times_ms = spike_indices / fs / (np.timedelta64(1,'ms')/np.timedelta64(1,'s'))
        spike_times_ms = rel_spike_times_ms + start_ms

        # fetch electrode count from implantation image (source of truth)
        img_query = culture.OrganoidImplantationImage & {"organoid_id": key["organoid_id"]}
        if not img_query:
            raise ValueError(f"No OrganoidImplantationImage entry found for organoid_id='{key['organoid_id']}' - insert a row before running this computation")
        if len(img_query) > 1:
            raise ValueError(f"Multiple OrganoidImplantationImage entries found for organoid_id='{key['organoid_id']}' - expected exactly one")
        num_elec_inside = img_query.fetch1("num_electrodes_inside")
        if num_elec_inside is None:
            raise ValueError(f"num_electrodes_inside is not set in OrganoidImplantationImage for organoid_id='{key['organoid_id']}'")
        elec_bool = (electrode_ids >= 0) & (electrode_ids < num_elec_inside)

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
                burst_spike_times = elec_spike_times[((index-num_burst_samples) <= elec_spike_times) & (elec_spike_times < (index+num_burst_samples))]

                # convert to indices within burst spike array
                burst_spike_indices = (burst_spike_times - (index-num_burst_samples)).astype(int)
                burst_spike_array[burst_idx, elec_idx, burst_spike_indices] = True
        burst_bounds = np.array(burst_windows)

        # determine spike time tiling coefficient (functional connectivity) for each burst (weighted by spikes per pair)
        sttc_array = np.zeros((len(burst_indices), num_elec_inside, num_elec_inside))
        weight_array = np.zeros((len(burst_indices), num_elec_inside, num_elec_inside))

        dt = 5 # time window for STTC in ms
        t_stop = 2*num_burst_samples # total time window for spike trains in ms
        for b_idx in range(len(burst_indices)):
            for i in range(num_elec_inside):
                for j in range(i+1, num_elec_inside):

                    # define spike times for each electrode within burst (in ms)
                    spike_times_i = np.where(burst_spike_array[b_idx, i, :])[0]
                    spike_times_j = np.where(burst_spike_array[b_idx, j, :])[0]

                    # skip if either electrode has no spikes in burst
                    if len(spike_times_i) == 0 or len(spike_times_j) == 0:
                        continue

                    # convert to spike trains (neo)
                    spiketrain_A = neo.SpikeTrain(spike_times_i, units='ms', t_stop=t_stop)
                    spiketrain_B = neo.SpikeTrain(spike_times_j, units='ms', t_stop=t_stop)

                    # calculate STTC
                    sttc = spike_time_tiling_coefficient(spiketrain_A, spiketrain_B, dt=dt*pq.ms)
                    sttc_array[b_idx, i, j] = sttc

                    # calculate weight (number of spike pairs)
                    weight_array[b_idx, i, j] = len(spike_times_i) * len(spike_times_j)

        # determine weighted average STTC for each burst; NaN for bursts with no spike pairs
        total_weight = weight_array.sum(axis=(1, 2))
        weighted_sttc = np.where(
            total_weight > 0,
            (sttc_array * weight_array).sum(axis=(1, 2)) / total_weight,
            np.nan,
        )

        # insert into table
        self.insert1({
            **key,
            'burst_indices': burst_indices,
            'burst_peak_heights': burst_peak_heights,
            'burst_bounds': burst_bounds,
            'burst_spike_array': burst_spike_array,
            'weighted_sttc': weighted_sttc,
        })
