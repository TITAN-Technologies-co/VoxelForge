# VoxelForge

A next-generation visual computing engine for voxel generation, 2D simulation, and procedural world creation.

Built by **TITAN Technologies**, VoxelForge unifies scripting, simulation, and geometry into a single extensible platform.

---

## What is VoxelForge?

VoxelForge is a hybrid development engine designed for:

- Voxel-based 3D generation
- Real-time 2D simulation systems
- Visual scripting through the VF language
- Procedural content creation pipelines

It is designed as a **renderer-agnostic engine**, allowing future support for:
PyQt5, OpenGL, Web, and custom render backends.

---

## Core Features

### Voxel Engine
- Image → voxel mesh conversion
- Procedural shape generation
- Material and color targeting system

### 2D Simulation Core
- Scene graph architecture
- Sprite-based entities
- Layer system with parallax support
- Deterministic update loop
- Collision detection and tagging system

### VF Scripting Language
A lightweight domain-specific language for controlling scenes and objects.

```vf
SCENE Demo
VIEWPORT 720 400
CAMERA 0 0 zoom=1

LAYER actors order=10

SPRITE player layer=actors x=60 y=200 w=40 h=50 color=#43B5FF
MOVE player vx=120 vy=0
