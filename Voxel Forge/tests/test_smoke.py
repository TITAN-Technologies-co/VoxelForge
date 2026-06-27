import importlib.util
from pathlib import Path

from voxelforge.storage import VoxelForgeStore


def test_voxelforge_import_is_safe():
    path = Path(__file__).resolve().parents[1] / "VoxelForge.py"
    spec = importlib.util.spec_from_file_location("voxelforge_app", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert hasattr(module, "main")
    assert hasattr(module, "VoxelForgeStore")


def test_store_round_trips_state_blob_and_user(tmp_path):
    store = VoxelForgeStore(tmp_path / "voxelforge.db")
    try:
        store.save_state("settings", {"theme": "Dark", "resolution": 96})
        assert store.load_state("settings") == {"theme": "Dark", "resolution": 96}

        store.save_blob("asset:test", b"voxel-data", "model")
        assert store.load_blob("asset:test") == b"voxel-data"

        created = store.create_email_user("Maker@example.com", "password123", "Maker")
        authed = store.authenticate_email_user("maker@example.com", "password123")
        assert authed["id"] == created["id"]
        assert authed["email"] == "maker@example.com"
    finally:
        store.close()
