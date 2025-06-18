from nomad.processing.data import Entry


def get_reference_from_mainfile(
    upload_id: str, mainfile: str, archive_path: str = 'data'
) -> str:
    """
    Uses the upload_id and mainfile to find the entry_id of an entry and returns
    a MProxy reference of a section in the entry.

    Args:
        upload_id (str): Upload ID of the upload in which the entry is located.
        mainfile (str): Mainfile of the entry to be referenced.
        archive_path (str, Optional): Path in the entry where the section is located.
            Defaults to 'data'.

    Returns:
        str: _description_
    """
    entry_id = None
    for entry in Entry.objects(upload_id=upload_id):
        if entry.mainfile == mainfile:
            entry_id = entry.entry_id
    if entry_id is None:
        return None
    return f'../uploads/{upload_id}/archive/{entry_id}#/{archive_path}'
