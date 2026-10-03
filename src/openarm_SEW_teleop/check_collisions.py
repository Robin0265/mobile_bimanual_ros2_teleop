"""
Check a tracking log from demo_openarm_xr_ros_teleop_hw.py --log for collisions between the
OpenArm's own links (and the floor), before sending that motion to the hardware.

It replays the logged joint positions through the MuJoCo model of the sim follower, both arms
at once, and reports every contact between separate parts: which bodies, when, and how deep.
The commanded trajectory is what the real arm would push toward, so a collision there means the
hardware would drive into itself with a force proportional to the penetration. The measured
trajectory (from the sim) shows what physically happened.

    python src/openarm_SEW_teleop/check_collisions.py logs/v2.5_x0.5.npz
    python src/openarm_SEW_teleop/check_collisions.py logs/*.npz --clearance 0.02

The model has no table or other surroundings; only self-collisions and the floor are checked.
"""

import argparse
import collections
import json
from pathlib import Path

import mujoco
import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
_MODEL = (_ROOT / "install" / "mobile_bimanual_description" / "share" / "mobile_bimanual_description"
          / "mujoco" / "openarm_mujoco" / "v1" / "scene.xml")

SIDES = ("right", "left")


def arm_trajectories(log, key):
    """Time and {side: (N, 7) positions} for one logged series; missing arms are None."""
    sides = [side for side in SIDES if f"{side}_t" in log]
    t = log[f"{sides[0]}_t"]
    positions = {}
    for side in SIDES:
        if side not in sides:
            positions[side] = None  # not commanded: held at home by the bringup
        elif len(log[f"{side}_t"]) == len(t):
            positions[side] = log[f"{side}_{key}"]
        else:
            q = log[f"{side}_{key}"]
            positions[side] = np.stack(
                [np.interp(t, log[f"{side}_t"], q[:, j]) for j in range(7)], axis=1
            )
    return t, positions


def find_contacts(model, data, t, positions, clearance):
    """[(time, body1, body2, distance)] for contacts closer than clearance, finger pairs excluded."""
    qpos_addrs = {
        side: [model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT,
                                                   f"openarm_{side}_joint{i}")] for i in range(1, 8)]
        for side in SIDES
    }
    body = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[g])
            for g in range(model.ngeom)]

    contacts = []
    for k, time in enumerate(t):
        mujoco.mj_resetData(model, data)  # fingers closed, other joints at home
        for side in SIDES:
            if positions[side] is not None:
                data.qpos[qpos_addrs[side]] = positions[side][k]
        mujoco.mj_forward(model, data)
        for i in range(data.ncon):
            contact = data.contact[i]
            b1, b2 = sorted((body[contact.geom1], body[contact.geom2]))
            # A hand's two fingers touch whenever the gripper is closed
            if b1.endswith("_left_finger") and b2.endswith("_right_finger") and b1[:12] == b2[:12]:
                continue
            if contact.dist < clearance:
                contacts.append((time, b1, b2, contact.dist))
    return contacts


def report(name, contacts, duration, dt):
    collisions = [c for c in contacts if c[3] < 0]
    near = [c for c in contacts if c[3] >= 0]
    if not contacts:
        print(f"  {name}: no collisions or near misses")
        return
    print(f"  {name}: {len({c[0] for c in collisions}) * dt:.2f} s in collision, "
          f"{len({c[0] for c in near}) * dt:.2f} s near-miss (of {duration:.1f} s)")
    by_pair = collections.defaultdict(list)
    for time, b1, b2, dist in contacts:
        by_pair[(b1, b2)].append((time, dist))
    for (b1, b2), hits in sorted(by_pair.items(), key=lambda kv: min(d for _, d in kv[1])):
        times = [time for time, _ in hits]
        deepest = min(dist for _, dist in hits)
        kind = f"penetration {-deepest * 1000:.1f} mm" if deepest < 0 else f"clearance {deepest * 1000:.1f} mm"
        print(f"    {b1.replace('openarm_', ''):>20s} <-> {b2.replace('openarm_', ''):<20s} "
              f"{kind:>24s}   t = {min(times):.2f}-{max(times):.2f} s")


def main():
    parser = argparse.ArgumentParser(description="Check teleop tracking logs for self-collisions")
    parser.add_argument("logs", nargs="+", type=Path, help=".npz files from --log")
    parser.add_argument("--clearance", default=0.01, type=float,
                        help="Also report parts closer than this many meters (default 0.01)")
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(_MODEL.as_posix())
    data = mujoco.MjData(model)
    # Contacts are only generated within a geom's margin: widen it to see near misses
    colliding = (model.geom_contype != 0) | (model.geom_conaffinity != 0)
    model.geom_margin[colliding] = args.clearance

    for path in args.logs:
        log = np.load(path)
        metadata = json.loads(str(log["metadata"]))
        csv = metadata.get("csv")
        source = (f"replay of {Path(csv).name} at {metadata.get('playback_speed', '?')}x"
                  if csv not in (None, "None") else "live session")
        print(f"{path}: {metadata.get('max_joint_vel', '?')} rad/s, {source}")
        for key, name in (("cmd", "commanded"), ("measured", "measured")):
            t, positions = arm_trajectories(log, key)
            dt = float(np.median(np.diff(t))) if len(t) > 1 else 0.0
            contacts = find_contacts(model, data, t, positions, args.clearance)
            report(name, contacts, t[-1] - t[0], dt)


if __name__ == "__main__":
    main()
