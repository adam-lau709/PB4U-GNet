#!/usr/bin/env python
import sys
import os
import glob
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.append(str(SCRIPT_DIR))

from renderer import GarmentRenderer

try:
    index = sys.argv.index("--path")
    path = sys.argv[index + 1]
except ValueError:
    print(
        "Usage: blender --background rendering/scene.blend --python rendering/render.py -- --path <path_to_meshes>"
    )
    raise

mesh_dir = Path(path)
if not mesh_dir.is_absolute():
    mesh_dir = (SCRIPT_DIR / mesh_dir).resolve()

print(f"Using mesh dir: {mesh_dir}")


renderer = GarmentRenderer(
    cloth_paths=sorted(glob.glob(str(mesh_dir / "garment*.obj"))),
    body_paths=sorted(glob.glob(str(mesh_dir / "body*.obj"))),
    cloth_material="ClothMaterialLightGreen",
    body_material="MannequinMaterialDark",
    export_path=str(mesh_dir / "render"),
)

renderer.render(resolution_percentage=100, fov=50, start_frame=0, end_frame=None)
renderer.generate_video()
