"""
OpenArm teleoperation through ROS 2 (ros2_control) using WebRTC body pose estimation.
Same pipeline as SEW-Geometric-Teleop's projects/openarm_teleop/demo_openarm_xr_robot_teleop_v1.py,
but joint targets go to the OpenArm's ros2_control controllers instead of an in-process MuJoCo sim:

    XR device (or CSV replay) -> OpenArmSEWSolver (+ SEW self-collision filter)
        -> joint limits + joint speed limit -> <namespace>/<side>_..._controller/commands

The robot is selected with --target (see TARGETS). "sim" drives the MuJoCo follower from
    ros2 launch mobile_bimanual_sim openarm_bimanual.launch.py
The physical OpenArm uses the same ros2_control interface under /leader; add it to TARGETS
once the pipeline has been verified in sim and on fake hardware.

Run from a shell that has sourced scripts/env.sh:
    python src/openarm_SEW_teleop/demo_openarm_xr_ros_teleop.py --target sim
    python src/openarm_SEW_teleop/demo_openarm_xr_ros_teleop.py --target sim \
        --csv References/SEW-Geometric-Teleop/References/recordings/shoulder_jumping.csv
"""

import argparse
import bisect
import contextlib
import csv
import importlib.util
import os
import sys
import types

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import time
import threading
import numpy as np
from dataclasses import dataclass
from pathlib import Path

import mujoco
import mujoco.viewer
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

_HERE = Path(__file__).resolve().parent
_SEW_ROOT = _HERE.parents[1] / "References" / "SEW-Geometric-Teleop"
sys.path.insert(0, _SEW_ROOT.as_posix())

# SEW's XR client subclasses robosuite's Device only for its interface and never uses it
# when env=None. Stand in for it so robosuite, whose mink==0.0.5 pin conflicts with this
# workspace's mink, doesn't have to be installed.
if importlib.util.find_spec("robosuite") is None:
    _robosuite = types.ModuleType("robosuite")
    _robosuite.devices = types.ModuleType("robosuite.devices")
    _robosuite.devices.Device = object
    sys.modules["robosuite"] = _robosuite
    sys.modules["robosuite.devices"] = _robosuite.devices

from projects.shared_devices.xr_robot_teleop_client import XRRTCBodyPoseDevice
from projects.openarm_teleop.openarm_sew_solver import OpenArmSEWSolver
from projects.openarm_teleop.openarm_mujoco_controller_v1 import OpenArmMuJoCoController
from projects.openarm_teleop.viz_utils import (
    visualize_sew_geometry,
    visualize_capsules,
    get_sew_transform,
)
from xr_robot_teleop_server.schemas.body_pose import Bone
import traceback


# The model the SEW solver uses. Here it only mirrors the measured robot state for the
# viewer and provides the joint limits.
_XML = _SEW_ROOT / "projects" / "openarm_teleop" / "v1" / "scene.xml"

SIDES = ("right", "left")


@dataclass(frozen=True)
class RobotTarget:
    """The ros2_control interface of one OpenArm bringup."""

    namespace: str
    arm_controller: str  # forward_command_controller name, formatted with side="right"/"left"
    max_joint_vel: float  # default joint speed limit, rad/s
    has_grippers: bool = False

    def command_topic(self, side):
        return f"{self.namespace}/{self.arm_controller.format(side=side)}/commands"

    @property
    def joint_states_topic(self):
        return f"{self.namespace}/joint_states"


TARGETS = {
    # MuJoCo follower: ros2 launch mobile_bimanual_sim openarm_bimanual.launch.py
    "sim": RobotTarget(
        namespace="/sim",
        arm_controller="{side}_arm_position_controller",
        max_joint_vel=2.0,
    ),
}


class OpenArmROSInterface(Node):
    """Publishes arm joint commands to ros2_control and tracks the measured joint positions."""

    def __init__(self, target):
        super().__init__("openarm_sew_teleop")
        self.joint_names = {
            side: [f"openarm_{side}_joint{i}" for i in range(1, 8)] for side in SIDES
        }
        self._lock = threading.Lock()
        self._q = {side: None for side in SIDES}
        self._stamp = {side: None for side in SIDES}  # receive time, not sim time

        self._pubs = {
            side: self.create_publisher(Float64MultiArray, target.command_topic(side), 10)
            for side in SIDES
        }
        self.create_subscription(
            JointState, target.joint_states_topic, self._on_joint_states, 10
        )

    def _on_joint_states(self, msg):
        # joint_states order is not fixed, so look joints up by name
        positions = dict(zip(msg.name, msg.position))
        now = time.monotonic()
        with self._lock:
            for side, names in self.joint_names.items():
                if all(name in positions for name in names):
                    self._q[side] = np.array([positions[name] for name in names])
                    self._stamp[side] = now

    def get_joint_positions(self, side, max_age):
        """Latest measured arm joint positions, or None if missing or older than max_age seconds."""
        with self._lock:
            if self._stamp[side] is None or time.monotonic() - self._stamp[side] > max_age:
                return None
            return self._q[side].copy()

    def publish_arm_command(self, side, q):
        self._pubs[side].publish(Float64MultiArray(data=[float(v) for v in q]))


class JointCommandLimiter:
    """Clips joint goals to the joint limits and steps the command toward them at a bounded speed."""

    def __init__(self, lower, upper, max_vel):
        self.lower = lower
        self.upper = upper
        self.max_vel = max_vel

    def clip(self, q_goal):
        return np.clip(q_goal, self.lower, self.upper)

    def step(self, q_cmd, q_goal, dt):
        max_step = self.max_vel * dt
        return q_cmd + np.clip(q_goal - q_cmd, -max_step, max_step)


class OpenArmROSController:
    """
    Drop-in for SEW's OpenArmMuJoCoController that drives the OpenArm through ros2_control.

    It keeps the same interface (set_joint_goals, update_position_control, q_current_right/left,
    model, data) so the teleop loop and SEW's visualization work unchanged. Goals are wrapped
    and clipped to the joint limits, and the command sent to the robot moves toward them at no
    more than max_joint_vel, starting from the measured pose. model/data mirror the measured
    robot state and are only used for visualization.
    """

    def __init__(self, ros_interface, mujoco_model, mujoco_data, max_joint_vel, control_dt,
                 state_timeout=0.2):
        self.ros = ros_interface
        self.model = mujoco_model
        self.data = mujoco_data
        self.control_dt = control_dt
        self.state_timeout = state_timeout

        # Reuse SEW's joint bookkeeping for the mirror model
        self._mirror = OpenArmMuJoCoController(mujoco_model, mujoco_data)
        joint_names = {
            "right": self._mirror.right_arm_joint_names,
            "left": self._mirror.left_arm_joint_names,
        }
        self._qpos_addrs = {
            "right": self._mirror.right_arm_qpos_addrs,
            "left": self._mirror.left_arm_qpos_addrs,
        }

        self._limiters = {}
        for side in SIDES:
            joint_ids = [
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
                for name in joint_names[side]
            ]
            limits = self.model.jnt_range[joint_ids]
            self._limiters[side] = JointCommandLimiter(limits[:, 0], limits[:, 1], max_joint_vel)

        self._lock = threading.Lock()
        self._q_current = {side: None for side in SIDES}
        self._q_goal = {side: None for side in SIDES}
        self._q_cmd = {side: None for side in SIDES}
        self._last_stale_warning = 0.0

    @property
    def q_current_right(self):
        return self._q_current["right"]

    @property
    def q_current_left(self):
        return self._q_current["left"]

    def wait_for_state(self, timeout, is_running):
        """Wait until both arms report joint states; the commands then start from the measured pose."""
        deadline = time.monotonic() + timeout
        while is_running() and time.monotonic() < deadline:
            q = {side: self.ros.get_joint_positions(side, self.state_timeout) for side in SIDES}
            if all(v is not None for v in q.values()):
                with self._lock:
                    for side in SIDES:
                        self._q_current[side] = q[side]
                        self._q_cmd[side] = q[side].copy()
                self._update_mirror()
                return True
            time.sleep(0.05)
        return False

    def set_joint_goals(self, goals):
        """
        Set target joint angles from a dictionary.

        Args:
            goals: Dictionary containing 'q_goal_right', 'q_goal_left'.
        """
        with self._lock:
            for side in SIDES:
                q_goal = goals.get(f"q_goal_{side}")
                if q_goal is not None:
                    self._q_goal[side] = np.asarray(q_goal, dtype=float)

    def goals_reached(self, tol=1e-3):
        """True once the command sent to both arms has reached their (clipped) goals."""
        with self._lock:
            for side in SIDES:
                if self._q_goal[side] is None or self._q_cmd[side] is None:
                    return False
                q_goal = self._limit_goal(side)
                if np.max(np.abs(q_goal - self._q_cmd[side])) > tol:
                    return False
        return True

    def _limit_goal(self, side):
        # Wrap toward the current command (SEW's real_angle), then clip to the joint limits
        q_goal = OpenArmMuJoCoController.real_angle(self._q_goal[side], self._q_cmd[side])
        return self._limiters[side].clip(q_goal)

    def update_position_control(self):
        """Step each arm's command toward its goal and publish it. Holds if joint states are stale."""
        stale = []
        with self._lock:
            for side in SIDES:
                q_measured = self.ros.get_joint_positions(side, self.state_timeout)
                if q_measured is None:
                    # Watchdog: without fresh feedback, stop sending; the controller holds its last command
                    stale.append(side)
                    continue
                self._q_current[side] = q_measured
                if self._q_goal[side] is None or self._q_cmd[side] is None:
                    continue

                q_goal = self._limit_goal(side)
                self._q_cmd[side] = self._limiters[side].step(
                    self._q_cmd[side], q_goal, self.control_dt
                )
                self.ros.publish_arm_command(side, self._q_cmd[side])

        if stale and time.monotonic() - self._last_stale_warning > 1.0:
            print(f"Warning: joint states stale for {', '.join(stale)} arm; holding last command.")
            self._last_stale_warning = time.monotonic()

        self._update_mirror()

    def _update_mirror(self):
        for side in SIDES:
            if self._q_current[side] is not None:
                self.data.qpos[self._qpos_addrs[side]] = self._q_current[side]
        mujoco.mj_forward(self.model, self.data)


class CSVBodyPoseReplay:
    """
    Replays a body-pose CSV recorded by SEW's XRRTCBodyPoseDevice (record_data=True) through the
    device's own bone -> action processing. Stands in for the live device: is_connected and
    get_controller_state(). Playback starts on the first get_controller_state() call.
    """

    def __init__(self, csv_path, playback_speed=1.0, loop=False):
        self.frame_times, self.frames = self._load(csv_path)
        self.playback_speed = playback_speed
        self.loop = loop
        self.process_bones_to_action_fn = XRRTCBodyPoseDevice._default_process_bones_to_action
        self.finished = False

        self._start_time = None
        self._frame_idx = None
        self._action = None
        print(
            f"Loaded {len(self.frames)} frames ({self.frame_times[-1]:.1f} s) from {csv_path}"
        )

    @staticmethod
    def _load(csv_path):
        # Newer recordings have data_type/id columns (bone and action rows); older ones only bone_id.
        frames = {}
        with open(csv_path, newline="") as f:
            for row in csv.DictReader(f):
                if row.get("data_type", "bone") != "bone":
                    continue
                bone_id = int(row["id"] if "id" in row else row["bone_id"])
                position = (float(row["pos_x"]), float(row["pos_y"]), float(row["pos_z"]))
                rotation = (
                    float(row["rot_x"]),
                    float(row["rot_y"]),
                    float(row["rot_z"]),
                    float(row["rot_w"]),
                )
                frames.setdefault(float(row["time_elapsed"]), []).append(
                    Bone(bone_id, position, rotation)
                )
        if not frames:
            raise ValueError(f"No bone data in {csv_path}")
        times = sorted(frames)
        return times, [frames[t] for t in times]

    @property
    def is_connected(self):
        return not self.finished

    def get_controller_state(self):
        now = time.time()
        if self._start_time is None:
            self._start_time = now

        t = (now - self._start_time) * self.playback_speed
        if t > self.frame_times[-1]:
            if not self.loop:
                self.finished = True
                return None
            self._start_time = now
            t = 0.0

        idx = max(bisect.bisect_right(self.frame_times, t) - 1, 0)
        if idx != self._frame_idx:
            self._frame_idx = idx
            self._action = self.process_bones_to_action_fn(self.frames[idx])
        return self._action


def main():
    """Main function for OpenArm SEW teleoperation through ROS 2."""
    parser = argparse.ArgumentParser(description="SEW teleoperation of OpenArm through ROS 2 (ros2_control)")
    parser.add_argument("--target", default="sim", choices=sorted(TARGETS), help="OpenArm bringup to command")
    parser.add_argument("--max_fr", default=60, type=int, help="Maximum frame rate for IK solver")
    parser.add_argument("--control_hz", default=200.0, type=float, help="Rate of joint commands sent to the controllers")
    parser.add_argument("--max_joint_vel", default=None, type=float, help="Joint speed limit in rad/s (default: per target)")
    parser.add_argument("--csv", default=None, type=Path, help="Replay a recorded body-pose CSV instead of the live XR device")
    parser.add_argument("--playback_speed", default=1.0, type=float, help="CSV playback speed multiplier")
    parser.add_argument("--loop", action="store_true", help="Loop the CSV replay")
    parser.add_argument("--no_safety_filter", action="store_true", help="Disable SEW's self-collision filter")
    parser.add_argument("--no_viewer", action="store_true", help="Run without the MuJoCo viewer")
    args = parser.parse_args()

    target = TARGETS[args.target]
    max_joint_vel = args.max_joint_vel if args.max_joint_vel is not None else target.max_joint_vel
    control_dt = 1.0 / args.control_hz

    # Check if XML file exists
    if not _XML.exists():
        print(f"Error: XML file not found at {_XML}")
        return

    try:
        print(f"Loading model from: {_XML}")
        model = mujoco.MjModel.from_xml_path(_XML.as_posix())
        data = mujoco.MjData(model)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        traceback.print_exc()
        return

    # Start ROS 2 and spin the interface node in the background
    rclpy.init()
    ros_interface = OpenArmROSInterface(target)
    executor = SingleThreadedExecutor()
    executor.add_node(ros_interface)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    try:
        run_teleop(args, target, model, data, ros_interface, max_joint_vel, control_dt)
    finally:
        executor.shutdown()
        ros_interface.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        print("Shutting down...")


def run_teleop(args, target, model, data, ros_interface, max_joint_vel, control_dt):
    """Ready pose, then SEW teleoperation until the viewer closes, Ctrl-C, or the replay ends."""
    # Initialize teleoperation components
    print("Initializing teleoperation system...")
    try:
        if args.csv is not None:
            device = CSVBodyPoseReplay(args.csv, args.playback_speed, args.loop)
        else:
            device = XRRTCBodyPoseDevice(env=None)
        controller = OpenArmROSController(ros_interface, model, data, max_joint_vel, control_dt)
        ik_solver = OpenArmSEWSolver(safety_filter=not args.no_safety_filter, debug=False)
        print("Teleoperation system initialized successfully!")
    except Exception as e:
        print(f"Error initializing teleoperation system: {e}")
        traceback.print_exc()
        return

    print(f"Target '{args.target}': commands on {target.command_topic('right')} and "
          f"{target.command_topic('left')}, speed limit {max_joint_vel:.2f} rad/s")
    if not target.has_grippers:
        print(f"Grippers are not commanded on target '{args.target}'.")

    # Define ready poses (same as the SEW demo)
    RIGHT_READY_RAD = np.array([0.0, 0.0, 0.0, np.pi / 2.0, 0.0, 0.0, 0.0])
    LEFT_READY_RAD = np.array([0.0, 0.0, 0.0, np.pi / 2.0, 0.0, 0.0, 0.0])

    stop_event = threading.Event()
    viewer_ctx = (
        contextlib.nullcontext()
        if args.no_viewer
        else mujoco.viewer.launch_passive(
            model=model,
            data=data,
            show_left_ui=False,
            show_right_ui=False,
        )
    )

    with viewer_ctx as viewer:
        if viewer is not None:
            viewer.cam.distance = 0.8
            viewer.cam.azimuth = 135
            viewer.cam.elevation = -15
            viewer.cam.lookat[:] = [0, 0, 0.5]

        def is_running():
            if stop_event.is_set() or not rclpy.ok():
                return False
            return viewer is None or viewer.is_running()

        # Visualization throttling
        viz_fps = 30
        viz_interval = 1.0 / viz_fps
        last_viz_time = 0.0

        def control_step():
            """Send one command to the robot and refresh the viewer."""
            nonlocal last_viz_time
            start_time = time.time()

            controller.update_position_control()

            # Visualize SEW Geometry
            if viewer is not None and time.time() - last_viz_time >= viz_interval:
                viewer.user_scn.ngeom = 0
                visualize_sew_geometry(viewer, ik_solver, controller)

                # Visualize Filter Capsules
                if ik_solver.last_filtered_sew is not None:
                    sew_dict = ik_solver.sew_filter.parse_sew(ik_solver.last_filtered_sew)
                    capsules = ik_solver.sew_filter.sew_to_capsules(sew_dict)
                    to_world = get_sew_transform(controller)
                    visualize_capsules(viewer, capsules, transform_func=to_world)

                viewer.sync()
                last_viz_time = time.time()

            elapsed = time.time() - start_time
            if elapsed < control_dt:
                time.sleep(control_dt - elapsed)

        ik_thread = None
        try:
            print(f"\nWaiting for joint states on {target.joint_states_topic}...")
            if not controller.wait_for_state(timeout=10.0, is_running=is_running):
                print("Error: no joint states received. Is the OpenArm bringup running?")
                return

            # Ramp from the measured pose to the ready pose while waiting for the client
            controller.set_joint_goals(
                {
                    "q_goal_right": RIGHT_READY_RAD,
                    "q_goal_left": LEFT_READY_RAD,
                }
            )

            print("Moving to ready pose. Waiting for a WebRTC client to connect...")
            while is_running() and not (controller.goals_reached() and device.is_connected):
                control_step()

            if not is_running():
                return
            print("Ready pose reached and client connected! Starting teleoperation.")

            # Lock for synchronizing reset operations
            ik_lock = threading.Lock()

            def ik_loop():
                while not stop_event.is_set():
                    loop_start = time.time()
                    if device is not None:
                        # Get action from device (WebRTC or CSV replay)
                        action = device.get_controller_state()
                        if action:
                            with ik_lock:

                                # Solve IK
                                joint_targets = ik_solver.solve(
                                    action,
                                    q_current_right=controller.q_current_right,
                                    q_current_left=controller.q_current_left,
                                )

                                # Update controller goals
                                if (
                                    joint_targets.get("q_goal_right") is not None
                                    or joint_targets.get("q_goal_left") is not None
                                ):
                                    controller.set_joint_goals(joint_targets)

                    # Rate limit IK loop
                    elapsed = time.time() - loop_start
                    if elapsed < 1 / args.max_fr:
                        time.sleep(1 / args.max_fr - elapsed)

            # Start IK thread
            ik_thread = threading.Thread(target=ik_loop)
            ik_thread.start()

            # Without new actions (tracking lost) the goals stay put and the robot holds
            while is_running() and not getattr(device, "finished", False):
                control_step()

            if getattr(device, "finished", False):
                print("Replay finished; holding the last command.")

        except KeyboardInterrupt:
            print("\nInterrupted.")
        finally:
            stop_event.set()
            if ik_thread is not None:
                ik_thread.join()


if __name__ == "__main__":
    main()
    print("Demo completed.")
