from typing import Any
import datajoint as dj

from element_array_ephys import ephys_no_curation as ephys
from element_array_ephys import ephys_report, probe
from element_interface.utils import find_full_path

from workflow import db_prefix
from workflow.pipeline import reference

from .induction import OrganoidExperiment

__all__ = ["ephys", "ephys_report", "probe"]


@ephys.schema
class RawEphys(dj.Imported):
    definition = """
    -> Subject
    probe_id:       int
    start_time:     datetime # date and time of file creation
    ---
    file_name :     varchar(32)  # name of the file
    file :          filepath@store  
    end_time :      datetime
    sampling_rate:  float # (Hz)
    header:         longblob  # meta information about the file.
    """

    class Channel(dj.Part):
        definition = """
        -> master
        channel_id : varchar(16)
        ---
        lfp:         blob@store  # LFP recording at this electrode in microvolts.
        """

    def make(self, key):
        ...


@ephys.schema
class EphysSession(dj.Manual):
    # activate the ephys element
    definition = """
    -> OrganoidExperiment
    start_time : datetime
    end_time : datetime
    """


Session = EphysSession
SkullReference = reference.SkullReference


if not ephys.schema.is_activated():
    ephys.activate(db_prefix + "ephys", db_prefix + "probe", linking_module=__name__)


# ------------- Activate "ephys" schema -------------
# Add a default kilosort2 paramset

# default_params = {
#     "fs": 2000,
#     "fshigh": 150,
#     "minfr_goodchannels": 0.1,
#     "Th": [10, 4],
#     "lam": 10,
#     "AUCsplit": 0.9,
#     "minFR": 0.02,
#     "momentum": [20, 400],
#     "sigmaMask": 30,
#     "ThPre": 8,
#     "spkTh": -6,
#     "reorder": 1,
#     "nskip": 25,
#     "GPU": 1,
#     "Nfilt": 1024,
#     "nfilt_factor": 4,
#     "ntbuff": 64,
#     "whiteningRange": 32,
#     "nSkipCov": 25,
#     "scaleproc": 200,
#     "nPCs": 3,
#     "useRAM": 0,
# }

# ephys.ClusteringParamSet.insert_new_params(
#     clustering_method="kilosort2.5",
#     paramset_desc="Default parameter set for Kilosort2.5",
#     params=default_params,
#     paramset_idx=0,
# )

# Populate ephys.AcquisitionSoftware
ephys.AcquisitionSoftware.insert1(
    {"acq_software": "Intan"},
    skip_duplicates=True,
)
