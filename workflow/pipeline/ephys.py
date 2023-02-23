from element_array_ephys import ephys_organoids as ephys
from element_array_ephys import probe

from workflow import db_prefix
from .lineage import Induction as Subject

__all__ = ["ephys", "probe"]



if not ephys.schema.is_activated():
    ephys.activate(db_prefix + "ephys", db_prefix + "probe", linking_module=__name__)

