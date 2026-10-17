"""Orthographic engineering views (top plan + front elevation) for the vision inspector."""
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Circle, Polygon as MplPolygon, Rectangle  # noqa: E402

from .checks import footprint  # noqa: E402
from .scene import Scene  # noqa: E402

COLORS = {"pillar": "#55595f", "conveyor": "#2f6db5", "bin": "#e07b1a", "robot": "#d4a017",
          "fence": "#888888", "rack": "#7a6a55", "pallet": "#a0784a", "sensor": "#8e44ad",
          "walkway": "#f4d03f", "rack": "#1b1d21", "cdu": "#aab7c4", "panel": "#7f8c8d", "busway": "#e67e22"}


def render_views(scene_dict, out_path, title=""):
    s = Scene(scene_dict)
    W, D = s.floor_w, s.floor_d
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(11, 10), gridspec_kw={"height_ratios": [D, 4.2]})

    ax.set_title(f"TOP VIEW (plan, meters)  {title}", fontsize=11, loc="left", family="monospace")
    ax.add_patch(Rectangle((0, 0), W, D, fill=False, lw=2, ec="black"))
    for o in sorted(s.data["objects"], key=lambda o: o["kind"] != "walkway"):
        poly = footprint(o)
        xy = list(poly.exterior.coords)
        k = o["kind"]
        if k == "walkway":
            ax.add_patch(MplPolygon(xy, fc="#fdf2c4", ec="#c9a400", hatch="//", lw=1))
            ax.text(o["x"], o["y"], f"{o['id']} (pedestrians)", ha="center", va="center", fontsize=8)
            continue
        if k == "keepout":
            ax.add_patch(MplPolygon(xy, fc="#fde2e0", ec="#c0392b", hatch="xx", lw=1, alpha=0.8))
            ax.text(o["x"], o["y"], f"{o['id']}\n{o.get('reason', 'keep out')}", ha="center", va="center",
                    fontsize=6.5, color="#922b21", wrap=True)
            continue
        if k == "busway":
            ax.add_patch(MplPolygon(xy, fill=False, ec="#e67e22", ls="--", lw=1.5))
            ax.text(o["x"] - o["w"] / 2 - 0.1, o["y"], f"{o['id']} (overhead)", ha="right", va="center",
                    fontsize=6.5, color="#a04000")
            continue
        if k == "rack":
            ax.add_patch(MplPolygon(xy, fc="#1b1d21", ec="#5dade2", lw=0.6))
            ax.text(o["x"], o["y"], o["id"], ha="center", va="center", fontsize=4.5, color="white", rotation=90)
            continue
        if k == "sensor":
            ax.plot(o["x"], o["y"], marker="*", ms=14, color=COLORS[k])
            ax.text(o["x"] + 0.2, o["y"] + 0.25, f"{o['id']} (overhead)", fontsize=7, color=COLORS[k])
            continue
        ax.add_patch(MplPolygon(xy, fc=COLORS.get(k, "#999"), ec="black", lw=1, alpha=0.9))
        if k == "robot":
            ax.add_patch(Circle((o["x"], o["y"]), o["reach"], fill=False, ls="--", ec="#b8860b", lw=1.2))
            ax.text(o["x"], o["y"] + o["reach"] - 0.35, f"{o['id']} reach {o['reach']} m",
                    ha="center", fontsize=7, color="#7a5a00")
        ax.text(o["x"], o["y"] - o["d"] / 2 - 0.28, o["id"], ha="center", fontsize=7.5, family="monospace")
        for name in o.get("points", {}):
            px, py = s.point(f"{o['id']}.{name}")
            ax.plot(px, py, marker="x", ms=7, color="black")
            ax.text(px + 0.1, py + 0.12, name, fontsize=6.5)
    ax.set_xlim(-0.3, W + 0.3)
    ax.set_ylim(-0.3, D + 0.3)
    ax.set_aspect("equal")
    ax.set_xticks(range(0, int(W) + 1))
    ax.set_yticks(range(0, int(D) + 1))
    ax.grid(True, lw=0.3, alpha=0.6)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")

    ax2.set_title("FRONT ELEVATION (looking +y, meters)", fontsize=11, loc="left", family="monospace")
    for o in sorted(s.data["objects"], key=lambda o: -o["y"]):
        if o["kind"] in ("walkway", "keepout"):
            continue
        x0, _, x1, _ = footprint(o).bounds
        z = o.get("z", 0.0)
        ax2.add_patch(Rectangle((x0, z), x1 - x0, max(o["h"], 0.08), fc=COLORS.get(o["kind"], "#999"),
                                ec="black", lw=0.8, alpha=0.75))
        if o["kind"] == "robot":
            # shoulder at column top, arm drawn at max horizontal reach
            top = z + o["h"]
            ax2.plot([o["x"], o["x"] + o["reach"] * 0.9], [top, top - 0.6], lw=4, color=COLORS["robot"])
        ax2.text((x0 + x1) / 2, z + o["h"] + 0.12, o["id"], ha="center", fontsize=6.5, rotation=90,
                 va="bottom", family="monospace")
    ax2.set_xlim(-0.3, W + 0.3)
    ax2.set_ylim(0, 7.5)
    ax2.set_aspect("equal")
    ax2.grid(True, lw=0.3, alpha=0.6)
    ax2.set_xlabel("x (m)")
    ax2.set_ylabel("z (m)")

    fig.tight_layout()
    fig.savefig(out_path, dpi=90)
    plt.close(fig)
    return out_path
