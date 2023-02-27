from pathlib import Path
from typing import Any

import datajoint as dj
import numpy as np
import yaml
from element_interface.intanloader import load_file
from element_interface.utils import find_full_path

from workflow.pipeline import probe
from workflow.utils.paths import get_ephys_root_data_dir, get_session_directory

logger = dj.logger


def get_probe_info() -> list[dict]:
    """Find probe.yaml in the root directory."""
    try:
        probe_meta_file = next(get_ephys_root_data_dir().glob("probe.yaml"))
    except StopIteration:
        raise FileNotFoundError("probe.yaml not found in the root data directory")
    else:
        with open(probe_meta_file, "r") as f:
            return yaml.safe_load(f)


def array_generator(arr: np.array, chunk_size: int = 10):
    """Generates an array at a specified chunk

    Args:
        arr (np.array): 1d numpy array
        chunk_size (int, optional): Size of the output array. Defaults to 10.

    Yields:
        Iterator[np.array]: generator object
    """
    start_ind = end_ind = 0

    while end_ind < arr.shape[0]:
        end_ind += chunk_size

        yield arr[start_ind:end_ind]

        start_ind += chunk_size


def make_continuous_dat(save_dir: Path, arr: np.array):
    """Save concatenated LFP in "continuous.dat" for cluster cutting.

    Args:
        save_dir (Path): Path to the save directory.
        arr (np.array): concatenated LFP array.
    """
    #
    continuous_file = save_dir / "continuous.dat"
    memmap_arr = np.memmap(
        continuous_file, dtype=np.float64, mode="w+", shape=arr.shape
    )
    memmap_arr[:] = arr[:]


def read_continuous_dat(file: Path, nb_channels: int) -> np.memmap:
    """Reads continuous.dat file

    Returns:
        np.memmap: numpy memmap object
    """
    data = np.memmap(file, dtype=np.float64)
    data_length = data.size // nb_channels
    return np.reshape(data, (nb_channels, data_length))
