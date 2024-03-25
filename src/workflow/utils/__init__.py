import pathlib

from workflow import REL_PATH_OUTBOX
from workflow.utils.paths import get_processed_root_data_dir


def download_results(relative_dir: str):
    """Download data from S3 to local outbox.

    Args:
        relative_dir (str): path relative to the output directory.
    """

    import djsciops.authentication as dj_auth
    import djsciops.axon as dj_axon
    import djsciops.settings as dj_settings

    dj_sciops_config = dj_settings.get_config()
    s3_session = dj_auth.Session(
        aws_account_id=dj_sciops_config["aws"]["account_id"],
        s3_role=dj_sciops_config["s3"]["role"],
        auth_client_id=dj_sciops_config["djauth"]["client_id"],
    )
    s3_bucket = dj_sciops_config["s3"]["bucket"]

    remote_dir = pathlib.Path(REL_PATH_OUTBOX) / relative_dir
    local_dir = pathlib.Path(get_processed_root_data_dir() / relative_dir)
    local_dir.mkdir(parents=True, exist_ok=True)

    if not local_dir.as_posix().endswith("\\"):
        local_dir = local_dir.as_posix() + "\\"
    else:
        local_dir = local_dir.as_posix()

    dj_axon.download_files(
        session=s3_session,
        s3_bucket=s3_bucket,
        source=remote_dir.as_posix(),
        destination=local_dir,
    )
    return local_dir
