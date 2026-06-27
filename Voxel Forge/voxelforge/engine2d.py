from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Callable, Iterable


@dataclass(frozen=True)
class Vec2:
    x: float = 0.0
    y: float = 0.0

    def __add__(self, other: Vec2) -> Vec2:
        return Vec2(self.x + other.x, self.y + other.y)

    def __sub__(self, other: Vec2) -> Vec2:
        return Vec2(self.x - other.x, self.y - other.y)

    def __mul__(self, scalar: float) -> Vec2:
        return Vec2(self.x * float(scalar), self.y * float(scalar))

    __rmul__ = __mul__

    def length(self) -> float:
        return math.hypot(self.x, self.y)

    def normalized(self) -> Vec2:
        size = self.length()
        if size <= 1e-9:
            return Vec2()
        return Vec2(self.x / size, self.y / size)

    def tuple(self) -> tuple[float, float]:
        return self.x, self.y


@dataclass(frozen=True)
class Rect:
    x: float
    y: float
    width: float
    height: float

    @property
    def left(self) -> float:
        return self.x

    @property
    def right(self) -> float:
        return self.x + self.width

    @property
    def top(self) -> float:
        return self.y

    @property
    def bottom(self) -> float:
        return self.y + self.height

    @property
    def center(self) -> Vec2:
        return Vec2(self.x + self.width * 0.5, self.y + self.height * 0.5)

    def contains(self, point: Vec2) -> bool:
        return self.left <= point.x <= self.right and self.top <= point.y <= self.bottom

    def intersects(self, other: Rect) -> bool:
        return (
            self.left < other.right
            and self.right > other.left
            and self.top < other.bottom
            and self.bottom > other.top
        )

    def moved(self, delta: Vec2) -> Rect:
        return Rect(self.x + delta.x, self.y + delta.y, self.width, self.height)


@dataclass
class Transform2D:
    position: Vec2 = field(default_factory=Vec2)
    scale: Vec2 = field(default_factory=lambda: Vec2(1.0, 1.0))
    rotation_degrees: float = 0.0
    origin: Vec2 = field(default_factory=Vec2)

    def matrix(self) -> tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]:
        radians = math.radians(self.rotation_degrees)
        cos_v = math.cos(radians)
        sin_v = math.sin(radians)
        sx, sy = self.scale.x, self.scale.y
        ox, oy = self.origin.x, self.origin.y
        tx = self.position.x - ox * sx * cos_v + oy * sy * sin_v
        ty = self.position.y - ox * sx * sin_v - oy * sy * cos_v
        return (
            (cos_v * sx, -sin_v * sy, tx),
            (sin_v * sx, cos_v * sy, ty),
            (0.0, 0.0, 1.0),
        )

    def transform_point(self, point: Vec2) -> Vec2:
        m = self.matrix()
        return Vec2(
            point.x * m[0][0] + point.y * m[0][1] + m[0][2],
            point.x * m[1][0] + point.y * m[1][1] + m[1][2],
        )


@dataclass
class Sprite2D:
    name: str
    size: Vec2
    transform: Transform2D = field(default_factory=Transform2D)
    velocity: Vec2 = field(default_factory=Vec2)
    texture: str = ""
    visible: bool = True
    solid: bool = False
    z_index: int = 0
    tags: set[str] = field(default_factory=set)
    metadata: dict[str, Any] = field(default_factory=dict)

    def step(self, dt: float) -> None:
        self.transform.position = self.transform.position + self.velocity * dt

    def bounds(self) -> Rect:
        scaled = Vec2(abs(self.size.x * self.transform.scale.x), abs(self.size.y * self.transform.scale.y))
        return Rect(
            self.transform.position.x - self.transform.origin.x * self.transform.scale.x,
            self.transform.position.y - self.transform.origin.y * self.transform.scale.y,
            scaled.x,
            scaled.y,
        )


@dataclass
class Layer2D:
    name: str
    order: int = 0
    visible: bool = True
    parallax: Vec2 = field(default_factory=lambda: Vec2(1.0, 1.0))
    sprites: list[Sprite2D] = field(default_factory=list)

    def add(self, sprite: Sprite2D) -> Sprite2D:
        if self.find(sprite.name) is not None:
            raise ValueError(f"Sprite '{sprite.name}' already exists in layer '{self.name}'.")
        self.sprites.append(sprite)
        return sprite

    def remove(self, name: str) -> Sprite2D:
        for index, sprite in enumerate(self.sprites):
            if sprite.name == name:
                return self.sprites.pop(index)
        raise KeyError(name)

    def find(self, name: str) -> Sprite2D | None:
        for sprite in self.sprites:
            if sprite.name == name:
                return sprite
        return None


@dataclass
class Camera2D:
    position: Vec2 = field(default_factory=Vec2)
    viewport: Vec2 = field(default_factory=lambda: Vec2(1280.0, 720.0))
    zoom: float = 1.0

    def world_to_screen(self, point: Vec2, parallax: Vec2 | None = None) -> Vec2:
        p = parallax or Vec2(1.0, 1.0)
        zoom = max(1e-6, float(self.zoom))
        return Vec2(
            (point.x - self.position.x * p.x) * zoom + self.viewport.x * 0.5,
            (point.y - self.position.y * p.y) * zoom + self.viewport.y * 0.5,
        )

    def screen_to_world(self, point: Vec2, parallax: Vec2 | None = None) -> Vec2:
        p = parallax or Vec2(1.0, 1.0)
        zoom = max(1e-6, float(self.zoom))
        return Vec2(
            (point.x - self.viewport.x * 0.5) / zoom + self.position.x * p.x,
            (point.y - self.viewport.y * 0.5) / zoom + self.position.y * p.y,
        )


@dataclass(frozen=True)
class DrawCommand:
    layer: str
    sprite: str
    texture: str
    bounds: Rect
    screen_position: Vec2
    rotation_degrees: float
    z_index: int
    metadata: dict[str, Any]


class Scene2D:
    def __init__(self, name: str = "Scene") -> None:
        self.name = str(name)
        self.camera = Camera2D()
        self.layers: list[Layer2D] = []
        self.time = 0.0
        self._systems: list[Callable[[Scene2D, float], None]] = []

    def add_layer(self, name: str, order: int = 0, visible: bool = True, parallax: Vec2 | None = None) -> Layer2D:
        if self.layer(name) is not None:
            raise ValueError(f"Layer '{name}' already exists.")
        layer = Layer2D(name=name, order=int(order), visible=bool(visible), parallax=parallax or Vec2(1.0, 1.0))
        self.layers.append(layer)
        self.layers.sort(key=lambda item: item.order)
        return layer

    def layer(self, name: str) -> Layer2D | None:
        for layer in self.layers:
            if layer.name == name:
                return layer
        return None

    def add_sprite(self, layer_name: str, sprite: Sprite2D) -> Sprite2D:
        layer = self.layer(layer_name)
        if layer is None:
            layer = self.add_layer(layer_name)
        return layer.add(sprite)

    def sprite(self, name: str) -> Sprite2D | None:
        for layer in self.layers:
            sprite = layer.find(name)
            if sprite is not None:
                return sprite
        return None

    def add_system(self, system: Callable[[Scene2D, float], None]) -> None:
        self._systems.append(system)

    def step(self, dt: float) -> None:
        dt = max(0.0, float(dt))
        self.time += dt
        for layer in self.layers:
            for sprite in layer.sprites:
                sprite.step(dt)
        for system in self._systems:
            system(self, dt)

    def render_plan(self) -> list[DrawCommand]:
        commands: list[DrawCommand] = []
        for layer in sorted(self.layers, key=lambda item: item.order):
            if not layer.visible:
                continue
            for sprite in sorted(layer.sprites, key=lambda item: item.z_index):
                if not sprite.visible:
                    continue
                commands.append(
                    DrawCommand(
                        layer=layer.name,
                        sprite=sprite.name,
                        texture=sprite.texture,
                        bounds=sprite.bounds(),
                        screen_position=self.camera.world_to_screen(sprite.transform.position, layer.parallax),
                        rotation_degrees=sprite.transform.rotation_degrees,
                        z_index=sprite.z_index,
                        metadata=dict(sprite.metadata),
                    )
                )
        return commands

    def query_point(self, point: Vec2, tag: str | None = None) -> list[Sprite2D]:
        found = []
        for sprite in self._sprites():
            if not sprite.visible:
                continue
            if tag is not None and tag not in sprite.tags:
                continue
            if sprite.bounds().contains(point):
                found.append(sprite)
        return found

    def collision_pairs(self, tag: str | None = None) -> list[tuple[Sprite2D, Sprite2D]]:
        sprites = [
            sprite
            for sprite in self._sprites()
            if sprite.visible and sprite.solid and (tag is None or tag in sprite.tags)
        ]
        pairs = []
        for index, left in enumerate(sprites):
            left_bounds = left.bounds()
            for right in sprites[index + 1 :]:
                if left_bounds.intersects(right.bounds()):
                    pairs.append((left, right))
        return pairs

    def _sprites(self) -> Iterable[Sprite2D]:
        for layer in self.layers:
            for sprite in layer.sprites:
                yield sprite


class Engine2D:
    def __init__(self, fixed_dt: float = 1.0 / 60.0) -> None:
        if fixed_dt <= 0:
            raise ValueError("fixed_dt must be greater than zero.")
        self.fixed_dt = float(fixed_dt)
        self.scene = Scene2D()
        self._accumulator = 0.0

    def load_scene(self, scene: Scene2D) -> None:
        self.scene = scene
        self._accumulator = 0.0

    def update(self, elapsed: float, max_steps: int = 8) -> int:
        self._accumulator += max(0.0, float(elapsed))
        steps = 0
        while self._accumulator >= self.fixed_dt and steps < max_steps:
            self.scene.step(self.fixed_dt)
            self._accumulator -= self.fixed_dt
            steps += 1
        if steps == max_steps:
            self._accumulator = min(self._accumulator, self.fixed_dt)
        return steps

    def render_plan(self) -> list[DrawCommand]:
        return self.scene.render_plan()
