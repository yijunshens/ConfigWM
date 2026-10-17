import os
from pathlib import Path


def get_cache_dir() -> Path:
    """Get the cache directory for stable_worldmodel data."""
    repo_cache_dir = Path("/data/syj/jepa/stable-worldmodel/two_room_data")
    default_cache_dir = (
        repo_cache_dir
        if repo_cache_dir.exists()
        else Path(os.path.expanduser("~/.stable_worldmodel"))
    )
    cache_dir = Path(os.getenv("STABLEWM_HOME", str(default_cache_dir)))
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir
