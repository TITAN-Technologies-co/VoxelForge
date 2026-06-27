from __future__ import annotations

from dataclasses import dataclass
import shlex

from .engine2d import Scene2D, Sprite2D, Transform2D, Vec2


SCRIPT_HELP = (
    "2D commands:\n"
    "SCENE name\n"
    "VIEWPORT width height\n"
    "CAMERA x y zoom=1\n"
    "LAYER name order=0 parallax=1,1 visible=true\n"
    "SPRITE name layer=actors x=0 y=0 w=32 h=32 color=#43B5FF texture=name z=0\n"
    "MOVE name vx=0 vy=0\n"
    "SOLID name true|false\n"
    "TAG name tag\n"
    "HIDE name / SHOW name\n"
    "DELETE name\n"
)


DEFAULT_2D_SCRIPT = """# VoxelForge 2D Studio
SCENE Neon Runner
VIEWPORT 720 400
CAMERA 0 0 zoom=1

LAYER background order=0 parallax=0.35,0.35
LAYER actors order=10
LAYER ui order=100 parallax=0,0

SPRITE sky layer=background x=-40 y=-30 w=840 h=480 color=#101820 z=0
SPRITE skyline layer=background x=40 y=230 w=620 h=90 color=#23395B z=1
SPRITE player layer=actors x=60 y=235 w=42 h=54 color=#43B5FF z=5
SPRITE drone layer=actors x=420 y=220 w=64 h=44 color=#F26F4C z=4
SPRITE goal layer=actors x=610 y=235 w=52 h=54 color=#8BC34A z=3
SPRITE label layer=ui x=18 y=16 w=180 h=28 color=#FFD166 z=20

MOVE player vx=110 vy=0
MOVE drone vx=-45 vy=0
SOLID player true
SOLID drone true
SOLID goal true
TAG player actor
TAG drone actor
TAG goal actor
"""


@dataclass(frozen=True)
class ScriptDiagnostic:
    line: int
    message: str


@dataclass(frozen=True)
class ScriptResult:
    scene: Scene2D
    actions: list[str]
    diagnostics: list[ScriptDiagnostic]

    @property
    def ok(self) -> bool:
        return not self.diagnostics


def split_args(tokens: list[str]) -> tuple[list[str], dict[str, str]]:
    pos = []
    kwargs = {}
    for token in tokens:
        if "=" in token:
            key, value = token.split("=", 1)
            key = key.strip().lower()
            if key:
                kwargs[key] = value.strip()
                continue
        pos.append(token)
    return pos, kwargs


def parse_bool(value: object, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on", "show", "visible"}


def parse_vec2(value: object, default: Vec2 = Vec2(1.0, 1.0)) -> Vec2:
    if value is None:
        return default
    text = str(value).strip()
    if "," not in text:
        number = float(text)
        return Vec2(number, number)
    left, right = text.split(",", 1)
    return Vec2(float(left), float(right))


def sprite_or_error(scene: Scene2D, name: str) -> Sprite2D:
    sprite = scene.sprite(name)
    if sprite is None:
        raise ValueError(f"Unknown sprite '{name}'.")
    return sprite


def parse_2d_script(script: str) -> ScriptResult:
    scene = Scene2D("Untitled")
    scene.add_layer("actors", order=10)
    actions: list[str] = []
    diagnostics: list[ScriptDiagnostic] = []

    for line_no, raw in enumerate(str(script).splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        try:
            tokens = shlex.split(line)
            if not tokens:
                continue
            command = tokens[0].upper()
            pos, kwargs = split_args(tokens[1:])

            if command == "HELP":
                actions.append(SCRIPT_HELP.strip())
                continue
            if command == "SCENE":
                if not pos:
                    raise ValueError("SCENE requires a name.")
                scene.name = " ".join(pos)
                actions.append(f"L{line_no}: scene set to '{scene.name}'")
                continue
            if command == "VIEWPORT":
                if len(pos) < 2:
                    raise ValueError("VIEWPORT requires width and height.")
                scene.camera.viewport = Vec2(float(pos[0]), float(pos[1]))
                actions.append(f"L{line_no}: viewport updated")
                continue
            if command == "CAMERA":
                if len(pos) < 2:
                    raise ValueError("CAMERA requires x and y.")
                scene.camera.position = Vec2(float(pos[0]), float(pos[1]))
                scene.camera.zoom = float(kwargs.get("zoom", scene.camera.zoom))
                actions.append(f"L{line_no}: camera updated")
                continue
            if command == "LAYER":
                if not pos:
                    raise ValueError("LAYER requires a name.")
                name = pos[0]
                existing = scene.layer(name)
                if existing is None:
                    layer = scene.add_layer(
                        name,
                        order=int(float(kwargs.get("order", 0))),
                        visible=parse_bool(kwargs.get("visible"), True),
                        parallax=parse_vec2(kwargs.get("parallax")),
                    )
                else:
                    layer = existing
                    layer.order = int(float(kwargs.get("order", layer.order)))
                    layer.visible = parse_bool(kwargs.get("visible"), layer.visible)
                    layer.parallax = parse_vec2(kwargs.get("parallax"), layer.parallax)
                actions.append(f"L{line_no}: layer '{layer.name}' ready")
                continue
            if command == "SPRITE":
                if not pos:
                    raise ValueError("SPRITE requires a name.")
                name = pos[0]
                layer_name = kwargs.get("layer", "actors")
                color = kwargs.get("color", "#43B5FF")
                sprite = Sprite2D(
                    name=name,
                    size=Vec2(float(kwargs.get("w", 32)), float(kwargs.get("h", 32))),
                    transform=Transform2D(
                        position=Vec2(float(kwargs.get("x", 0)), float(kwargs.get("y", 0))),
                        scale=parse_vec2(kwargs.get("scale"), Vec2(1.0, 1.0)),
                        rotation_degrees=float(kwargs.get("rot", 0)),
                    ),
                    texture=kwargs.get("texture", name),
                    visible=parse_bool(kwargs.get("visible"), True),
                    solid=parse_bool(kwargs.get("solid"), False),
                    z_index=int(float(kwargs.get("z", 0))),
                    metadata={"color": color},
                )
                scene.add_sprite(layer_name, sprite)
                actions.append(f"L{line_no}: sprite '{name}' added")
                continue
            if command == "MOVE":
                if not pos:
                    raise ValueError("MOVE requires a sprite name.")
                sprite = sprite_or_error(scene, pos[0])
                sprite.velocity = Vec2(float(kwargs.get("vx", 0)), float(kwargs.get("vy", 0)))
                actions.append(f"L{line_no}: sprite '{sprite.name}' velocity updated")
                continue
            if command == "SOLID":
                if not pos:
                    raise ValueError("SOLID requires a sprite name.")
                sprite = sprite_or_error(scene, pos[0])
                sprite.solid = parse_bool(pos[1] if len(pos) > 1 else kwargs.get("value"), True)
                actions.append(f"L{line_no}: sprite '{sprite.name}' solid={sprite.solid}")
                continue
            if command == "TAG":
                if len(pos) < 2:
                    raise ValueError("TAG requires a sprite name and tag.")
                sprite = sprite_or_error(scene, pos[0])
                sprite.tags.add(pos[1])
                actions.append(f"L{line_no}: tag '{pos[1]}' added to '{sprite.name}'")
                continue
            if command in {"HIDE", "SHOW"}:
                if not pos:
                    raise ValueError(f"{command} requires a sprite name.")
                sprite = sprite_or_error(scene, pos[0])
                sprite.visible = command == "SHOW"
                actions.append(f"L{line_no}: sprite '{sprite.name}' visible={sprite.visible}")
                continue
            if command == "DELETE":
                if not pos:
                    raise ValueError("DELETE requires a sprite name.")
                deleted = False
                for layer in scene.layers:
                    if layer.find(pos[0]) is not None:
                        layer.remove(pos[0])
                        deleted = True
                        break
                if not deleted:
                    raise ValueError(f"Unknown sprite '{pos[0]}'.")
                actions.append(f"L{line_no}: sprite '{pos[0]}' deleted")
                continue
            raise ValueError(f"Unknown 2D command '{command}'.")
        except Exception as exc:
            diagnostics.append(ScriptDiagnostic(line_no, str(exc)))

    if not actions and not diagnostics:
        actions.append("No commands executed.")
    return ScriptResult(scene=scene, actions=actions, diagnostics=diagnostics)
