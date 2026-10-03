"""
Overlay tracking logs saved by demo_openarm_xr_ros_teleop_hw.py --log, e.g. one per speed limit:

    for v in 1.5 2.0 2.5; do
        python src/openarm_SEW_teleop/demo_openarm_xr_ros_teleop_hw.py --target sim --no_viewer \
            --csv References/SEW-Geometric-Teleop/References/recordings/ipman_roll.csv \
            --max_joint_vel $v --log logs/v$v.npz
    done
    python src/openarm_SEW_teleop/plot_tracking_logs.py logs/v*.npz --arm right

For each joint and log it plots SEW's goal (dashed), the command sent after the limits (thin)
and the measured position (thick), in one colour per log, against the time since tracking
started. Replays of the same recording line up in time, but SEW's goals only match across runs
with --no_safety_filter: the self-collision filter works from the measured pose, so a different
speed limit also changes what SEW asks for.
"""

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Overlay SEW teleop tracking logs")
    parser.add_argument("logs", nargs="+", type=Path, help=".npz files from --log")
    parser.add_argument("--arm", default="right", choices=["right", "left"], help="Arm to plot")
    parser.add_argument("--out", default=None, type=Path, help="Save the figure here instead of opening a window")
    parser.add_argument("--no_command", action="store_true", help="Plot only goal and measured positions")
    args = parser.parse_args()

    if args.out is not None:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(7, 1, sharex=True, figsize=(12, 15))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for n, path in enumerate(args.logs):
        log = np.load(path)
        if f"{args.arm}_t" not in log:
            print(f"{path}: no {args.arm} arm data, skipped")
            continue
        metadata = json.loads(str(log["metadata"]))
        label = f"{metadata['max_joint_vel']:g} rad/s" if "max_joint_vel" in metadata else path.stem
        t, goal, cmd, measured = (log[f"{args.arm}_{key}"] for key in ("t", "goal", "cmd", "measured"))
        color = colors[n % len(colors)]

        for j, ax in enumerate(axes):
            ax.plot(t, np.degrees(goal[:, j]), "--", color=color, lw=1.0, label=f"{label} SEW goal")
            if not args.no_command:
                ax.plot(t, np.degrees(cmd[:, j]), color=color, lw=0.8, alpha=0.5)
            max_error = np.degrees(np.max(np.abs(measured[:, j] - cmd[:, j])))
            ax.plot(t, np.degrees(measured[:, j]), color=color, lw=1.8,
                    label=f"{label} measured (max err {max_error:.0f} deg)")

    for j, ax in enumerate(axes):
        ax.set_ylabel(f"J{j + 1} (deg)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, loc="upper right")
    axes[-1].set_xlabel("time since tracking started (s)")
    fig.suptitle(f"{args.arm} arm: SEW goal (dashed), command (thin), measured (thick)")
    fig.tight_layout()

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.out, dpi=110)
        print(f"Saved {args.out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
