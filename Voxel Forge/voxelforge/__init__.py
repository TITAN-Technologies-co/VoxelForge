from .engine2d import Camera2D, DrawCommand, Engine2D, Layer2D, Rect, Scene2D, Sprite2D, Transform2D, Vec2
from .storage import VoxelForgeStore

__all__ = [
    "Camera2D",
    "DrawCommand",
    "Engine2D",
    "HeightMapViewer",
    "Layer2D",
    "Rect",
    "Scene2D",
    "Sprite2D",
    "Transform2D",
    "Vec2",
    "VoxelForgeStore",
    "main",
]


def __getattr__(name):
    if name in {"HeightMapViewer", "main"}:
        from .app import HeightMapViewer, main

        return {"HeightMapViewer": HeightMapViewer, "main": main}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
