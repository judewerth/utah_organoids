import datajoint as dj
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

from workflow import DB_PREFIX

from .ephys import ephys

schema = dj.schema(DB_PREFIX + "analysis")

logger = dj.logger


@schema
class SpectralBand(dj.Lookup):
    definition = """
    band_name: varchar(16)
    ---
    lower_freq: float # (Hz)
    upper_freq: float # (Hz)
    """
    contents = [
        ("delta", 1.0, 4.0),
        ("theta", 4.0, 7.0),
        ("alpha", 8.0, 12.0),
        ("beta", 13.0, 30.0),
        ("gamma", 30.0, 50.0),
        ("highgamma1", 70.0, 110.0),
        ("highgamma2", 130.0, 200.0),
    ]


@schema
class LFPTraceQC(dj.Computed):
    """
    Time-domain QC metrics for each LFP trace (per electrode).
    Includes variance, noise level, and waveform shape (skewness/kurtosis).
    """

    definition = """
    -> ephys.LFP.Trace
    ---
    lfp_std: float # Overall signal amplitude (spread, uV)
    noise_level: float # Median absolute deviation (noise level estimate, uV)
    lfp_skewness: float # Skewness of the voltage distribution (Asymmetry)
    lfp_kurtosis: float # Kurtosis of the voltage distribution (Tail heaviness)
    """

    def make(self, key):
        import scipy.stats as stats

        lfp = (ephys.LFP.Trace & key).fetch1("lfp")

        # Standard deviation (variance)
        lfp_std = float(np.std(lfp))

        # Median absolute deviation
        noise_level = float(stats.median_abs_deviation(lfp, scale="normal"))

        # Waveform shape
        lfp_skewness = float(stats.skew(lfp))
        lfp_kurtosis = float(stats.kurtosis(lfp))

        self.insert1({
            **key,
            "lfp_std": lfp_std,
            "noise_level": noise_level,
            "lfp_skewness": lfp_skewness,
            "lfp_kurtosis": lfp_kurtosis
        })


@schema
class SpectrogramParameters(dj.Lookup):
    definition = """
    param_idx: int
    ---
    window_size:     float    # Time in seconds
    overlap_size=0:  float    # Time in seconds
    description="":  varchar(64)
    """
    contents = [
        (0, 0.5, 0.0, "Default 0.5s time segments without overlap."),
        (1, 2.0, 1.0, "2s window, 50% overlap (delta)"),
        (2, 0.5, 0.25, "0.5s window, 50% overlap (theta/beta)"),  
        (3, 0.2, 0.1, "0.2s window, 50% overlap (gamma/high-gamma)")
    ]

@schema
class LFPSpectrogram(dj.Computed):
    """Calculate spectrogram at each channel, extracts power in frequency bands,
    and handles electrode mapping.

    Assumes the LFP is:
        1. Low-pass filtered at 1000 Hz.
        2. Notch filtered at 50/60 Hz.
        3. Resampled to 2500 Hz.
    """

    definition = """
    -> ephys.LFP.Trace
    -> SpectrogramParameters
    """

    class ChannelSpectrogram(dj.Part):
        definition = """
        -> master
        ---
        spectrogram: longblob # Power with dimensions (frequency, time)
        time: longblob        # Timestamps 
        frequency: longblob   # Fourier frequencies
        """

    class ChannelPower(dj.Part):
        definition = """
        -> master
        -> SpectralBand
        ---
        power: longblob   # Mean power in spectral band as a function of time
        mean_power: float # Mean power in a spectral band for a time window.
        std_power: float  # Standard deviation of the power in a spectral band for a time window.
        """

    def make(self, key):
        self.insert1(key)

        lfp = (ephys.LFP.Trace & key).fetch1("lfp")
        fs = (ephys.LFP & key).fetch1("lfp_sampling_rate")
        win_params = (SpectrogramParameters & key).fetch1("window_size", "overlap_size")

        nperseg = int(win_params[0] * fs)
        noverlap = int(win_params[1] * fs)

        freq, t, Sxx = signal.spectrogram(
            lfp,
            fs=fs,
            window="tukey",
            nperseg=nperseg,
            noverlap=noverlap,
            scaling="density",
            mode="psd",
        )

        # Store spectrogram results
        self.ChannelSpectrogram.insert1(
            {**key, "spectrogram": Sxx, "frequency": freq, "time": t}
        )

        # Calculate power in each frequency band
        band_keys, f_lo, f_hi = SpectralBand.fetch("KEY", "lower_freq", "upper_freq")

        for band_key, fl, fh in zip(band_keys, f_lo, f_hi):
            band_mask = np.logical_and(freq >= fl, freq < fh)
            power = Sxx[band_mask].mean(axis=0) if band_mask.any() else np.zeros_like(t)
            self.ChannelPower.insert1({
                **key,
                **band_key,
                "power": power,
                "mean_power": float(power.mean()),
                "std_power": float(power.std()),
            })

