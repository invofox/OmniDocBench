import os


def discover_documents(origin_data_url: str, subdirectory: str) -> list[str]:
    """
    Discover individual documents in the dataset.

    First checks if there's a 'X' subdirectory. If found, uses that directory.
    If not found, assumes we're already in the gt directory and uses the current directory.
    """
    if not os.path.isdir(origin_data_url):
        return []

    # Check if there's a 'X' subdirectory
    subdir = os.path.join(origin_data_url, subdirectory)
    search_dir = subdir if os.path.isdir(subdir) else origin_data_url

    document_paths = []
    for item in sorted(os.listdir(search_dir)):
        item_path = os.path.join(search_dir, item)
        if os.path.isdir(item_path):
            document_paths.append(item_path)

    return document_paths
