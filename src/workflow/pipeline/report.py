import datajoint as dj
import tempfile
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt

from workflow import DB_PREFIX
from workflow.pipeline import ephys, ephys_sorter


logger = dj.logger
schema = dj.schema(DB_PREFIX + "report")


@schema
class SpikeInterfaceReport(dj.Computed):
    definition = """
    -> ephys_sorter.SIExport
    """

    class Plot(dj.Part):
        definition = """
        -> master
        name: varchar(64)
        ---
        plot: attach
        """

    def make(self, key):
        png_query = ephys_sorter.SIExport.File & key & "file_name LIKE '%png'"

        self.insert1(key)

        for f in png_query.fetch("file"):
            f = Path(f)
            self.Plot.insert1({**key, "name": f.stem, "plot": f.as_posix()})
