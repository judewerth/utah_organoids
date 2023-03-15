import datajoint as dj
from scripts.initiate_session import upload_session_data

db_prefix = dj.config["custom"].get("database.prefix", "")
