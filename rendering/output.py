import pickle
from pathlib import Path


def write_obj(path: Path, verts, faces, obj_name: str) -> None:
    """Write one mesh OBJ file; numpy face indices are assumed 0-based."""
    lines: list[str] = []

    lines.append(f"o {obj_name}")
    for v in verts:
        lines.append(f"v {float(v[0])} {float(v[1])} {float(v[2])}")

    for tri in faces:
        i, j, k = int(tri[0]) + 1, int(tri[1]) + 1, int(tri[2]) + 1
        lines.append(f"f {i} {j} {k}")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    with open("output.pkl", "rb") as f:
        data = pickle.load(f)

    pred = data["pred"]
    obstacle = data["obstacle"]
    cloth_faces = data["cloth_faces"]
    obstacle_faces = data["obstacle_faces"]

    n_frames = pred.shape[0]
    if obstacle.shape[0] != n_frames:
        raise ValueError(
            f"pred and obstacle must share frame count; got {n_frames} vs {obstacle.shape[0]}"
        )

    out_dir = Path("obj_frames")
    out_dir.mkdir(parents=True, exist_ok=True)

    for t in range(n_frames):
        write_obj(out_dir / f"garment_{t:04d}.obj", pred[t], cloth_faces, "cloth")
        write_obj(out_dir / f"body_{t:04d}.obj", obstacle[t], obstacle_faces, "body")

    print(f"Wrote {n_frames} garment and {n_frames} body OBJ files to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
