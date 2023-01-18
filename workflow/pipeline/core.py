from element_animal import subject
from element_lab import lab
from element_session import session_with_id as session

from workflow import db_prefix

__all__ = ["lab", "subject", "session"]

# ------------- Activate "lab" -------------

lab.activate(db_prefix + "lab")

# ------------- Activate "subject" schema -------------

Source = lab.Source
Lab = lab.Lab
Protocol = lab.Protocol
User = lab.User
Location = lab.Location
Project = lab.Project

subject.activate(db_prefix + "subject", linking_module=__name__)

# ------------- Activate "session" schema -------------

Subject = subject.Subject
Experimenter = lab.User

session.activate(db_prefix + "session", linking_module=__name__)
