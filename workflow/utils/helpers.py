from typing import Any, Iterator

import numpy as np
from element_interface.utils import find_full_path

from workflow.utils.paths import get_ephys_root_data_dir, get_session_dir


def get_probe_info(session_key: dict[str, Any]) -> dict[str, Any]:
    """Find probe.yaml in a session folder

    Args:
        session_key (dict[str, Any]): session key

    Returns:
        dict[str, Any]: probe meta information
    """
    import yaml

    experiment_dir = find_full_path(
        get_ephys_root_data_dir(), get_session_dir(session_key)
    )

    probe_meta_file = next(experiment_dir.glob("probe*"))

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
        new_arr = arr[start_ind:end_ind]

        yield new_arr

        start_ind += chunk_size


def downsample_1d_arr(
    arr: np.array, ds_factor: int = 10, selected_ind: int = 0
) -> np.array:
    """Function for downsampling an 1d array

    Args:
        arr (np.array): 1d numpy array
        ds_factor (int, optional): Downsampling factor. Defaults to 10. If 10, 20kHz will be downsampled to 2kHz.
        selected_ind (int, optional): Select which index to keep. Defaults to 0 (first index).

    Returns:
        np.array: Final downsampled array.
    """
    if arr.shape[0] < ds_factor:
        raise NotImplementedError("Array is smaller than the downsampling factor")

    if arr.ndim != 1:
        raise NotImplementedError("Only 1d array is accepted.")

    assert selected_ind < ds_factor, "selected_ind should be < ds_factor"

    downsampled_arr = []

    for new_arr in array_generator(arr, chunk_size=ds_factor):
        if new_arr.shape[0] <= selected_ind:
            selected_ind = -1
        downsampled_arr.append(new_arr[selected_ind])

    return np.array(downsampled_arr)
