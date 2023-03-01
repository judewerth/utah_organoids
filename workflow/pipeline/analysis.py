import datajoint as dj
from .ephys import ephys, event
from workflow import db_prefix
from scipy import signal
import numpy as np

schema = dj.schema(db_prefix + "analysis")


@schema
class Impedence(dj.Imported):
    definition = """
    -> ephys.EphysRecording
    ---
    value: float
    """


@schema
class SpectralBand(dj.Lookup):
    definition = """
    band_name: varchar(16)
    ---
    lower_freq: float # (Hz)
    upper_freq: float # (Hz)
    """
    contents = [
        ("delta", 0.5, 4.0),
        ("theta", 4.0, 7.0),
        ("alpha", 8.0, 12.0),
        ("beta", 18.0, 22.0),
        ("gamma", 30.0, 70.0),
        ("highgamma", 80.0, 500.0),
    ]


@schema
class SpectrogramParameters(dj.Lookup):
    definition = """
    param_idx: int
    ---
    window_size:     float    # Time in seconds
    overlap_size=0:  float    # Time in seconds
    description="":  varchar(32)
    """
    contents = [(0, 2.0, 0.0, "Default 2s time segments without overlap.")]


@schema
class LFPSpectrogram(dj.Computed):
    definition = """
    -> ephys.LFP
    -> SpectrogramParameters
    -> event.AlignmentEvent
    """

    class ChannelSpectrogram(dj.Part):
        definition = """
        -> master
        -> ephys.LFP.Electrode
        ---
        spectrogram: longblob # Power with dimensions (frequecy, time).
        time: longblob        # Timestamps
        frequency: longblob   # Fourier frequencies
        """

    class Power(dj.Part):
        definition = """
        -> master.ChannelSpectrogram
        -> SpectralBand
        ---
        power: longblob
        mean_power: float
        std_power: float
        """

    def make(self, key):
        """
        Assuming LFP at each channel is
            1. bandpass 0.2Hz - 500Hz
            2. notch-filter at 60Hz
            3. resample to 1000Hz (Nyquist)
        """
        lfp_time_stamps, lfp_sampling_rate, window_size, overlap_size = (
            ephys.LFP * ephys.EphysRecording * SpectrogramParameters * SpectralBand
            & key
        ).fetch(
            "lfp_time_stamps",
            "lfp_sampling_rate",
            "window_size",
            "overlap_size",
        )
        even_specs = (event.AlignmentEvent & key).fetch1()

        start_time = (
            event.Event & key & {"event_type": even_specs["start_event_type"]}
        ).fetch("event_start_time")[0] + even_specs["start_time_shift"]
        end_time = (
            event.Event & key & {"event_type": even_specs["end_event_type"]}
        ).fetch("event_start_time")[0] + even_specs["end_time_shift"]

        time_mask = np.logical_and(
            lfp_time_stamps >= start_time, lfp_time_stamps < end_time
        )

        self.insert1(key)
        for lfp_key, lfp in (ephys.LFP.Electrode & key).fetch("KEY", "lfp"):
            f, t, Sxx = signal.spectrogram(
                lfp[time_mask],
                fs=int(lfp_sampling_rate),
                nperseg=int(window_size * lfp_sampling_rate),
                noverlap=int(overlap_size * lfp_sampling_rate),
                window="boxcar",
            )

            self.ChannelSpectrogram.insert1(
                {**key, **lfp_key, "sxx": Sxx, "f": f, "t": t}
            )
            for power_key, fl, fh in SpectralBand.fetch(
                "KEY", "lower_freq", "upper_freq"
            ):
                freq_mask = np.logical_and(f >= fl, f < fh)
                power = Sxx[freq_mask, :].mean(axis=0)  # mean across freq domain
                self.Power.insert1(
                    {
                        **key,
                        **lfp_key,
                        "power": power,
                        "mean_power": power.mean(),
                        "std_power": power.std(),
                    }
                )
