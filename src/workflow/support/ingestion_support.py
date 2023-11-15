import re
from datetime import datetime, timedelta
from pathlib import Path

import datajoint as dj
import numpy as np

from workflow import REL_PATH_INBOX, SUPPORT_DB_PREFIX
from workflow.pipeline import ephys
from workflow.support import FileManifest

logger = dj.logger  # type: ignore

schema = dj.schema(f"{SUPPORT_DB_PREFIX}ingestion_support")


@schema
class FileProcessing(dj.Imported):
    definition = """
    -> FileManifest
    ---
    execution_time: datetime  # UTC time
    log_message='': varchar(1000)
    """

    def make(self, key):
        """
        For each new file in FileManifest, process the file to attempt to register new entries for ephys.EphysRawFile (from .rhs files)
        """
        log_message = ""
        remote_fullpath = Path(key["remote_fullpath"])
        if Path(REL_PATH_INBOX) in remote_fullpath.parents:
            parent_dir = remote_fullpath.parent
            if remote_fullpath.suffix in [".rhd", ".rhs"]:
                filename_prefix, start_time = re.search(
                    r"(.*)_(\d{6}_\d{6})", remote_fullpath
                ).groups()
                start_time = np.datetime64(
                    datetime.strptime(start_time, "%y%m%d_%H%M%S")
                )  # start time based on the file name
                ephys.EphysRawFile.insert1(
                    {
                        "file_path": remote_fullpath.as_posix(),
                        "acq_software": {".rhd": "Intan", ".rhs": "Intan"}[
                            remote_fullpath.suffix
                        ],
                        "file_time": start_time,
                        "parent_folder": parent_dir.name,
                        "filename_prefix": filename_prefix,
                        "file": (FileManifest & key).fetch1("file"),
                    }
                )
                log_message += f"Added new raw ephys: {remote_fullpath.name}" + "\n"
        self.insert1(
            {**key, "execution_time": datetime.utcnow(), "log_message": log_message}
        )
