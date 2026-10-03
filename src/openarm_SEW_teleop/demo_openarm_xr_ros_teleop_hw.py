"""
OpenArm hardware teleoperation through ROS 2 (ros2_control) using WebRTC body pose estimation.
Same SEW pipeline as demo_openarm_xr_ros_teleop_simu.py (whose classes it reuses), with the
additions the physical arm needs:

    - pre-flight checks: controller manager, hardware and controllers active, REAL/FAKE/SIM
      detection, joint states inside the joint limits and near home
    - joint position and velocity limits inherited from the bringup's robot_description,
      positions pulled in by a margin (they are the mechanical stops); both arms and grippers
      by default (--arms, --no_grippers)
    - no prompts: ramp to the ready pose, then track as soon as the client is connected
    - hold on tracking loss and resume when data returns; fault (hold at the measured pose
      until Ctrl-C) on stale joint states, excessive tracking error or an inactive
      controller/hardware component
    - on exit (Ctrl-C, viewer closed, replay end) ramp back to home before the motors are
      switched off; a second Ctrl-C skips it

Bring up the arm WITHOUT hardware_leader_sim_follower.launch.py, which switches the motors off:
    ros2 launch openarm_bringup openarm.bimanual.launch.py arm_type:=v1.0 \
        use_fake_hardware:=false right_can_interface:=can0 left_can_interface:=can1 \
        robot_controller:=forward_position_controller arm_prefix:=leader \
        runtime_config_package:=mobile_bimanual_bringup \
        controllers_file:=openarm_leader_controllers.yaml launch_rviz:=false
or, for the same interface without motors: ros2 launch mobile_bimanual_bringup openarm_fake.launch.py

Run from a shell that has sourced scripts/env.sh:
    python src/openarm_SEW_teleop/demo_openarm_xr_ros_teleop_hw.py \
        --csv References/SEW-Geometric-Teleop/References/recordings/ipman_roll.csv --playback_speed 0.5
    python src/openarm_SEW_teleop/demo_openarm_xr_ros_teleop_hw.py
--target sim rehearses the same checks and sequence on the MuJoCo follower.
"""

import argparse
import contextlib
import json
import signal
import threading
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

import mujoco
import mujoco.viewer
import rclpy
from controller_manager_msgs.srv import ListControllers, ListHardwareComponents
from rclpy.executors import SingleThreadedExecutor
from rclpy.qos import DurabilityPolicy, QoSProfile
from rclpy.signals import SignalHandlerOptions
from std_msgs.msg import String

import demo_openarm_xr_ros_teleop_simu as simu
from demo_openarm_xr_ros_teleop_simu import (
    SIDES,
    CSVBodyPoseReplay,
    OpenArmMuJoCoController,
    OpenArmROSController,
    OpenArmROSInterface,
    OpenArmSEWSolver,
    RobotTarget,
    XRRTCBodyPoseDevice,
    get_sew_transform,
    visualize_capsules,
    visualize_sew_geometry,
)
import traceback


TARGETS = {
    # Physical OpenArm (or openarm_fake.launch.py) through the upstream bimanual bringup
    "hardware": RobotTarget(
        namespace="/leader",
        arm_controller="{side}_forward_position_controller",
        max_joint_vel=1.5,
        gripper_controller="{side}_gripper_controller",
        max_gripper_vel=0.1,
    ),
    # MuJoCo follower, to rehearse the hardware sequence with the same defaults
    "sim": RobotTarget(
        namespace="/sim",
        arm_controller="{side}_arm_position_controller",
        max_joint_vel=1.5,
        gripper_controller="{side}_gripper_controller",
        max_gripper_vel=0.1,
    ),
}

# ros2_control hardware plugins known not to move a physical arm; anything else counts as REAL
SIMULATED_HARDWARE = {
    "mujoco_ros2_control/MujocoSystemInterface": "SIM",
    "mock_components/GenericSystem": "FAKE",
}

HOME_RAD = np.zeros(7)  # arm hanging straight down: the driver's homing pose, ~0 gravity torque
READY_RAD = np.array([0.0, 0.0, 0.0, np.pi / 2.0, 0.0, 0.0, 0.0])  # same as the SEW demo

LIMIT_TOLERANCE = 0.1  # rad a measured joint may sit outside its limit (sag, calibration)
NEAR_HOME = 0.2  # rad; farther than this from home at start asks for confirmation
TRACKING_LOST_TIMEOUT = 0.5  # s without new device data before holding
STALE_STATE_TIMEOUT = 0.5  # s without joint states before faulting
TRACKING_ERROR_DURATION = 0.3  # s the tracking error may stay above the threshold
# SEW's thumb-index distance when the headset sends no fingertips (maps to fully open)
NO_FINGER_TRACKING_DIST = 0.1
# Gripper squeeze: a closed hand is commanded this many motor degrees past the gripper's
# closed position (zero it at closed first). Change it here, or per run with
# --gripper_close_overtravel_deg; 0 turns it off.
GRIPPER_CLOSE_OVERTRAVEL_DEG = 5.0
# openarm_hardware's parallel-gripper scale: 0.044 m of opening per 1.0472 rad of motor travel
GRIPPER_M_PER_MOTOR_RAD = 0.044 / 1.0472


class StopSession(Exception):
    """Raised when the viewer window is closed."""


def _raise_keyboard_interrupt(signum, frame):
    raise KeyboardInterrupt


class OpenArmHWController(OpenArmROSController):
    """
    OpenArmROSController with the hardware additions: joint limits from the robot description
    (pulled in by a margin, or removed), only the selected arms commanded, and helpers for the
    safety supervisor.

    joint_limits: {side: (lower, upper, velocity)} arrays for joints 1-7;
    gripper_limits: {side: (lower, upper)} for finger_joint1 (needed when commanding grippers).
    """

    def __init__(self, *args, joint_limits, gripper_limits=None, arms=SIDES,
                 joint_limit_margin=0.0, use_joint_limits=True, gripper_max_opening=None,
                 gripper_close_overtravel=0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.arms = tuple(arms)

        # Description limits, also used for the pre-flight check
        self.joint_limits = {side: (lower, upper) for side, (lower, upper, _) in joint_limits.items()}
        for side, limiter in self._limiters.items():
            lower, upper, velocity = joint_limits[side]
            if use_joint_limits:
                limiter.lower = lower + joint_limit_margin
                limiter.upper = upper - joint_limit_margin
            else:
                limiter.lower = np.full_like(lower, -np.inf)
                limiter.upper = np.full_like(upper, np.inf)
            # Never faster than the description's joint velocity limits
            limiter.max_vel = np.minimum(limiter.max_vel, velocity)
        # Openings in m; closing may go gripper_close_overtravel past closed to squeeze
        self.gripper_close_overtravel = gripper_close_overtravel
        for side, limiter in self._gripper_limiters.items():
            lower, upper = gripper_limits[side]
            limiter.lower = lower - gripper_close_overtravel
            limiter.upper = upper if gripper_max_opening is None else min(upper, gripper_max_opening)

        self._stats = None  # tracking statistics, collected between start_stats and stop_stats

    def set_joint_goals(self, goals):
        # Arms that are not selected are never commanded; they hold where the bringup left them
        skipped = [side for side in SIDES if side not in self.arms]
        goals = {
            key: value
            for key, value in goals.items()
            if key not in [f"q_goal_{side}" for side in skipped]
            and key not in [f"{side}_gripper_val" for side in skipped]
        }
        super().set_joint_goals(goals)

        # The base class sets each gripper goal from SEW's mapping on every call; add the squeeze
        if self.command_grippers and self.gripper_close_overtravel > 0:
            with self._lock:
                for side in self.arms:
                    opening = getattr(self._mirror, f"q_goal_{side}_hand")
                    if opening is not None:
                        self._g_goal[side] = self._with_close_overtravel(side, float(opening[0]))

    def _with_close_overtravel(self, side, opening):
        """Closed -> overtravel past closed, blending back to unchanged at the largest opening."""
        upper = self._gripper_limiters[side].upper
        if upper <= 0:
            return opening
        fraction_open = min(max(opening / upper, 0.0), 1.0)
        return opening - self.gripper_close_overtravel * (1.0 - fraction_open)

    def measured_positions(self, side):
        return self._q_current[side]

    def goals_reached(self, tol=1e-3):
        """True once the commands to the selected arms (and grippers) have reached their goals."""
        with self._lock:
            for side in self.arms:
                if self._q_goal[side] is None or self._q_cmd[side] is None:
                    return False
                if np.max(np.abs(self._limit_goal(side) - self._q_cmd[side])) > tol:
                    return False
                if self.command_grippers and self._g_goal[side] is not None:
                    g_goal = self._gripper_limiters[side].clip(self._g_goal[side])
                    if self._g_cmd[side] is None or abs(g_goal - self._g_cmd[side]) > 1e-4:
                        return False
        return True

    def seconds_to_goals(self):
        """Time the speed-limited arm commands need to reach their goals."""
        with self._lock:
            seconds = [
                np.max(np.abs(self._limit_goal(side) - self._q_cmd[side])
                       / self._limiters[side].max_vel)
                for side in self.arms
                if self._q_goal[side] is not None and self._q_cmd[side] is not None
            ]
        return max(seconds, default=0.0)

    def tracking_error(self):
        """Largest |measured - commanded| arm joint error over the selected arms, rad."""
        with self._lock:
            errors = [
                np.max(np.abs(self._q_current[side] - self._q_cmd[side]))
                for side in self.arms
                if self._q_current[side] is not None and self._q_cmd[side] is not None
            ]
        return max(errors, default=0.0)

    def hold_at_measured(self):
        """Stop pushing: make the measured pose both the goal and the command."""
        with self._lock:
            for side in self.arms:
                if self._q_current[side] is not None:
                    self._q_goal[side] = self._q_current[side].copy()
                    self._q_cmd[side] = self._q_current[side].copy()
                if self._g_current[side] is not None:
                    self._g_goal[side] = self._g_current[side]
                    self._g_cmd[side] = self._g_current[side]

    def update_position_control(self):
        with self._lock:
            before = {side: None if self._q_cmd[side] is None else self._q_cmd[side].copy()
                      for side in self.arms}
        super().update_position_control()
        if self._stats is None:
            return

        with self._lock:
            for side in self.arms:
                if before[side] is None or self._q_cmd[side] is None or self._q_current[side] is None:
                    continue
                # A joint is speed-limited when its command moved by the full allowed step
                max_step = self._limiters[side].max_vel * self.control_dt
                limited = np.abs(self._q_cmd[side] - before[side]) >= 0.999 * max_step
                self._stats["limited"][side] += limited
                self._stats["samples"][side] += 1
                self._stats["errors"].append(
                    np.max(np.abs(self._q_current[side] - self._q_cmd[side]))
                )
                if self._stats["record"] is not None and self._q_goal[side] is not None:
                    # SEW's goal as asked (wrapped like the command, but not clipped)
                    goal = OpenArmMuJoCoController.real_angle(self._q_goal[side], self._q_cmd[side])
                    self._stats["record"][side].append(np.concatenate((
                        [time.monotonic() - self._stats["start"]],
                        goal, self._q_cmd[side], self._q_current[side],
                    )))

    def start_stats(self, record=False):
        """
        Collect speed-limit and tracking-error statistics until stop_stats(). With record=True
        also keep every step's goal, command and measured positions for a log file.
        """
        self._stats = {
            "start": time.monotonic(),
            "samples": {side: 0 for side in self.arms},
            "limited": {side: np.zeros(7) for side in self.arms},
            "errors": [],
            "record": {side: [] for side in self.arms} if record else None,
        }

    def stop_stats(self, fault_threshold=None, log_path=None, metadata=None):
        """
        Stop collecting and return a printable summary (None if nothing was collected).
        With log_path, also save the recorded steps there (see plot_tracking_logs.py).
        """
        stats, self._stats = self._stats, None
        if not stats or not stats["errors"]:
            return None

        lines = [f"Tracking summary ({time.monotonic() - stats['start']:.0f} s):"]
        for side in self.arms:
            samples = stats["samples"][side]
            if samples:
                share = "  ".join(
                    f"J{i + 1} {100 * count / samples:3.0f}%"
                    for i, count in enumerate(stats["limited"][side])
                )
                lines.append(f"  {side:5s} arm at its speed limit: {share}")
        errors = np.array(stats["errors"])
        threshold = f" (fault at {fault_threshold:.2f})" if fault_threshold else ""
        lines.append(f"  tracking error mean {errors.mean():.3f}, 95% {np.percentile(errors, 95):.3f}, "
                     f"max {errors.max():.3f} rad{threshold}")

        if log_path is not None and stats["record"] is not None:
            metadata = dict(metadata or {})
            metadata["max_joint_vel"] = float(np.max(self._limiters[self.arms[0]].max_vel))
            data = {"metadata": np.array(json.dumps(metadata))}
            for side, rows in stats["record"].items():
                if rows:
                    rows = np.array(rows)
                    data[f"{side}_t"] = rows[:, 0]
                    data[f"{side}_goal"] = rows[:, 1:8]
                    data[f"{side}_cmd"] = rows[:, 8:15]
                    data[f"{side}_measured"] = rows[:, 15:22]
            Path(log_path).parent.mkdir(parents=True, exist_ok=True)
            np.savez(log_path, **data)
            lines.append(f"  log saved to {log_path}")
        return "\n".join(lines)


class ControllerManagerClient:
    """Queries <namespace>/controller_manager for its hardware components and controllers."""

    def __init__(self, node, namespace):
        self.name = f"{namespace}/controller_manager"
        self._hardware = node.create_client(
            ListHardwareComponents, f"{self.name}/list_hardware_components"
        )
        self._controllers = node.create_client(ListControllers, f"{self.name}/list_controllers")

    @staticmethod
    def _call(client, timeout):
        # The node is spun by the background executor; just wait for the future
        if not client.wait_for_service(timeout_sec=timeout):
            return None
        future = client.call_async(client.srv_type.Request())
        deadline = time.monotonic() + timeout
        while not future.done() and time.monotonic() < deadline:
            time.sleep(0.01)
        return future.result() if future.done() else None

    def check(self, required_controllers, timeout=2.0):
        """Return (hardware plugin classes, problems); problems is empty when all is active."""
        hardware = self._call(self._hardware, timeout)
        controllers = self._call(self._controllers, timeout)
        if hardware is None or controllers is None:
            return [], [f"{self.name} is not responding"]

        problems = [
            f"hardware component '{c.name}' is {c.state.label}"
            for c in hardware.component
            if c.state.label != "active"
        ]
        states = {c.name: c.state for c in controllers.controller}
        problems += [
            f"controller '{name}' is {states.get(name, 'not loaded')}"
            for name in required_controllers
            if states.get(name) != "active"
        ]
        return [c.class_type for c in hardware.component], problems


def fetch_urdf_limits(node, topic, timeout=5.0):
    """
    Joint limits from the robot_description the bringup publishes (transient local):
    {joint name: (lower, upper, velocity)}, or None if nothing arrives within timeout.
    """
    received = []
    qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
    subscription = node.create_subscription(String, topic, lambda msg: received.append(msg.data), qos)
    deadline = time.monotonic() + timeout
    while not received and time.monotonic() < deadline:
        time.sleep(0.05)
    node.destroy_subscription(subscription)
    if not received:
        return None

    limits = {}
    for joint in ET.fromstring(received[0]).findall("joint"):
        limit = joint.find("limit")
        if limit is not None and joint.get("type") in ("revolute", "prismatic"):
            limits[joint.get("name")] = (
                float(limit.get("lower", "-inf")),
                float(limit.get("upper", "inf")),
                float(limit.get("velocity", "inf")),
            )
    return limits


def hardware_kind(classes):
    kinds = {SIMULATED_HARDWARE.get(c, "REAL") for c in classes}
    if not kinds or "REAL" in kinds:
        return "REAL"
    return "/".join(sorted(kinds))


class HealthMonitor:
    """Checks the controller manager once a second; fault holds the first problem found."""

    def __init__(self, cm_client, required_controllers, period=1.0):
        self.fault = None
        self._cm = cm_client
        self._required = required_controllers
        self._period = period
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()

    def stop(self):
        # Join so no service call is in flight when the node is destroyed
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=3.0 * self._period)

    def _run(self):
        while not self._stop.wait(self._period):
            _, problems = self._cm.check(self._required, timeout=self._period)
            if problems and self.fault is None:
                self.fault = "; ".join(problems)


def main():
    """Main function for OpenArm SEW teleoperation on hardware through ROS 2."""
    parser = argparse.ArgumentParser(description="SEW teleoperation of the physical OpenArm through ROS 2 (ros2_control)")
    parser.add_argument("--target", default="hardware", choices=sorted(TARGETS), help="OpenArm bringup to command")
    parser.add_argument("--arms", default="both", choices=["right", "left", "both"], help="Arms to command")
    parser.add_argument("--max_fr", default=60, type=int, help="Maximum frame rate for IK solver")
    parser.add_argument("--control_hz", default=200.0, type=float, help="Rate of joint commands sent to the controllers")
    parser.add_argument("--max_joint_vel", default=None, type=float, help="Joint speed limit in rad/s (default: per target)")
    parser.add_argument("--joint_limit_margin_deg", default=5.0, type=float, help="Keep commands this far inside the joint limits")
    parser.add_argument("--no_joint_limits", action="store_true", help="Do not clip to joint limits (refused on real hardware)")
    parser.add_argument("--max_tracking_error", default=0.35, type=float, help="Fault above this tracking error in rad (0 disables)")
    parser.add_argument("--no_grippers", action="store_true", help="Leave the grippers alone")
    parser.add_argument("--gripper_max_opening", default=0.044, type=float, help="Largest gripper opening in m")
    parser.add_argument("--gripper_close_overtravel_deg", default=GRIPPER_CLOSE_OVERTRAVEL_DEG, type=float,
                        help="Command a closed hand this many motor degrees past closed, to squeeze (0: off)")
    parser.add_argument("--max_gripper_vel", default=None, type=float, help="Gripper speed limit in m/s (default: per target)")
    parser.add_argument("--csv", default=None, type=Path, help="Replay a recorded body-pose CSV instead of the live XR device")
    parser.add_argument("--playback_speed", default=1.0, type=float, help="CSV playback speed multiplier")
    parser.add_argument("--loop", action="store_true", help="Loop the CSV replay")
    parser.add_argument("--record", default=None, type=Path, help="Save the live XR session as a CSV in this directory")
    parser.add_argument("--log", default=None, type=Path, help="Save goal/command/measured traces while tracking to this .npz (see plot_tracking_logs.py)")
    parser.add_argument("--no_safety_filter", action="store_true", help="Disable SEW's self-collision filter")
    parser.add_argument("--no_viewer", action="store_true", help="Run without the MuJoCo viewer")
    args = parser.parse_args()

    if args.record is not None and args.csv is not None:
        print("Error: --record saves live XR sessions; it cannot be combined with --csv.")
        return

    target = TARGETS[args.target]

    try:
        print(f"Loading model from: {simu._XML}")
        model = mujoco.MjModel.from_xml_path(simu._XML.as_posix())
        data = mujoco.MjData(model)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        traceback.print_exc()
        return

    # Keep ROS running on Ctrl-C/SIGTERM so the arm can still be sent home; both raise
    # KeyboardInterrupt in the main thread instead. Set SIGINT explicitly: a shell starts
    # background jobs with it ignored.
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    signal.signal(signal.SIGINT, signal.default_int_handler)
    signal.signal(signal.SIGTERM, _raise_keyboard_interrupt)
    ros_interface = OpenArmROSInterface(target)
    cm_client = ControllerManagerClient(ros_interface, target.namespace)
    executor = SingleThreadedExecutor()
    executor.add_node(ros_interface)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    try:
        run_session(args, target, model, data, ros_interface, cm_client)
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        # Let the spin thread leave the executor before the node goes away
        executor.shutdown(timeout_sec=1.0)
        spin_thread.join(timeout=2.0)
        ros_interface.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        print("Shutting down...")


def run_session(args, target, model, data, ros_interface, cm_client):
    """Pre-flight, ready pose, SEW teleoperation under supervision, then home."""
    arms = SIDES if args.arms == "both" else (args.arms,)
    command_grippers = not args.no_grippers and target.gripper_controller is not None
    max_joint_vel = args.max_joint_vel if args.max_joint_vel is not None else target.max_joint_vel
    max_gripper_vel = args.max_gripper_vel if args.max_gripper_vel is not None else target.max_gripper_vel
    control_dt = 1.0 / args.control_hz

    required_controllers = ["joint_state_broadcaster"]
    required_controllers += [target.arm_controller.format(side=side) for side in arms]
    if command_grippers:
        required_controllers += [target.gripper_controller.format(side=side) for side in arms]

    # ---------------- Pre-flight ----------------
    print(f"\nPre-flight checks on {cm_client.name}...")
    classes, problems = cm_client.check(required_controllers)
    if problems:
        print("Error: the OpenArm bringup is not ready:")
        for problem in problems:
            print(f"  - {problem}")
        return
    kind = hardware_kind(classes)
    print(f"Hardware: {kind} ({', '.join(sorted(set(classes)))})")
    if kind == "REAL" and args.no_joint_limits:
        print("Error: refusing --no_joint_limits on real hardware: nothing else stops the joints at "
              "their mechanical stops. Use --joint_limit_margin_deg 0 to clip at the stops.")
        return

    # Joint limits come from the bringup's robot description, not from our own model
    description_topic = f"{target.namespace}/robot_description"
    urdf_limits = fetch_urdf_limits(ros_interface, description_topic)
    if urdf_limits is None:
        print(f"Error: no robot description on {description_topic}.")
        return
    try:
        joint_limits = {
            side: tuple(
                np.array([urdf_limits[f"openarm_{side}_joint{i}"][k] for i in range(1, 8)])
                for k in range(3)
            )
            for side in SIDES
        }
        gripper_limits = None
        if command_grippers:
            gripper_limits = {side: urdf_limits[f"openarm_{side}_finger_joint1"][:2] for side in SIDES}
    except KeyError as e:
        print(f"Error: joint {e} has no limits in {description_topic}.")
        return
    print(f"Joint limits from {description_topic}.")

    controller = OpenArmHWController(
        ros_interface, model, data, max_joint_vel, control_dt,
        max_gripper_vel=max_gripper_vel if command_grippers else None,
        joint_limits=joint_limits,
        gripper_limits=gripper_limits,
        arms=arms,
        joint_limit_margin=np.deg2rad(args.joint_limit_margin_deg),
        use_joint_limits=not args.no_joint_limits,
        gripper_max_opening=args.gripper_max_opening,
        gripper_close_overtravel=np.deg2rad(args.gripper_close_overtravel_deg) * GRIPPER_M_PER_MOTOR_RAD,
    )
    print(f"Waiting for joint states on {target.joint_states_topic}...")
    if not controller.wait_for_state(timeout=5.0, is_running=lambda: True):
        print("Error: no joint states received.")
        return

    far_from_home = []
    for side in arms:
        q = controller.measured_positions(side)
        lower, upper = controller.joint_limits[side]
        if np.any(q < lower - LIMIT_TOLERANCE) or np.any(q > upper + LIMIT_TOLERANCE):
            print(f"Error: the {side} arm reads {np.round(np.rad2deg(q), 1)} deg, outside its joint "
                  "limits. Check the stored zero positions before continuing.")
            return
        if np.max(np.abs(q - HOME_RAD)) > NEAR_HOME:
            far_from_home.append(side)

    limits = "OFF" if args.no_joint_limits else f"{args.joint_limit_margin_deg:.1f} deg inside the limits"
    print(f"Commanding: {', '.join(arms)} arm(s) at <= {max_joint_vel:.2f} rad/s, joint limits {limits}, "
          f"self-collision filter {'OFF' if args.no_safety_filter else 'on'}, "
          f"grippers {f'on (closing {args.gripper_close_overtravel_deg:g} deg past closed)' if command_grippers else 'off'}, "
          f"tracking-error fault {'off' if args.max_tracking_error <= 0 else f'> {args.max_tracking_error:.2f} rad'}")

    if kind == "REAL":
        print("\nMoving the REAL OpenArm. E-stop in hand, workspace clear, second person watching.")
    if far_from_home:
        print(f"Warning: the {', '.join(far_from_home)} arm is more than {NEAR_HOME} rad from home. "
              "The ready-pose ramp starts from where it is.")

    # ---------------- Teleoperation components ----------------
    print("Initializing teleoperation system...")
    try:
        if args.csv is not None:
            device = CSVBodyPoseReplay(args.csv, args.playback_speed, args.loop)
        elif args.record is not None:
            device = XRRTCBodyPoseDevice(env=None, record_data=True, output_dir=args.record.as_posix())
        else:
            device = XRRTCBodyPoseDevice(env=None)
        ik_solver = OpenArmSEWSolver(safety_filter=not args.no_safety_filter, debug=False)
        print("Teleoperation system initialized successfully!")
    except Exception as e:
        print(f"Error initializing teleoperation system: {e}")
        traceback.print_exc()
        return

    monitor = HealthMonitor(cm_client, required_controllers)
    monitor.start()
    try:
        teleoperate(args, device, ik_solver, controller, ros_interface, monitor, arms,
                    command_grippers, model, data, control_dt)
    finally:
        monitor.stop()
        if args.record is not None:
            device.cleanup_recording()


def teleoperate(args, device, ik_solver, controller, ros_interface, monitor, arms,
                command_grippers, model, data, control_dt):
    """Ready pose, tracking under supervision once the client connects, then return home."""
    stop_event = threading.Event()
    engaged = threading.Event()
    last_action_time = [time.monotonic()]

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
            if (viewer is not None and viewer.is_running()
                    and time.time() - last_viz_time >= viz_interval):
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

        def wait_for(condition):
            """Keep commanding until condition() holds; closing the viewer ends the session."""
            while not condition():
                if viewer is not None and not viewer.is_running():
                    raise StopSession
                control_step()

        # Lock for synchronizing reset operations
        ik_lock = threading.Lock()

        def ik_loop():
            last_sew = None
            warned_no_fingers = set()
            while not stop_event.is_set():
                loop_start = time.time()
                # Get action from device (WebRTC or CSV replay)
                action = device.get_controller_state()
                # SEW's XR device returns a fresh shallow copy of its last pose on every call, even
                # when no new data has arrived; the pose arrays inside only change with new data
                sew = (action.get("right_sew"), action.get("left_sew")) if action else None
                if sew is not None and (
                    last_sew is None or any(new is not old for new, old in zip(sew, last_sew))
                ):
                    last_sew = sew
                    last_action_time[0] = time.monotonic()

                    for side in arms:
                        if (command_grippers and side not in warned_no_fingers
                                and action.get(f"{side}_gripper_val") == NO_FINGER_TRACKING_DIST):
                            warned_no_fingers.add(side)
                            print(f"Warning: no {side} thumb/index fingertips from the headset; that "
                                  "gripper stays open. Use hand tracking (controllers down).")

                    if engaged.is_set():
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

        def supervise():
            """Run tracking until a fault (returned as text) or the end of the replay (None)."""
            over_limit_since = None
            while True:
                if viewer is not None and not viewer.is_running():
                    raise StopSession
                control_step()

                if getattr(device, "finished", False):
                    print("Replay finished.")
                    return None
                if monitor.fault is not None:
                    return monitor.fault
                for side in arms:
                    if ros_interface.get_joint_positions(side, STALE_STATE_TIMEOUT) is None:
                        return f"no joint states from the {side} arm for {STALE_STATE_TIMEOUT} s"

                error = controller.tracking_error()
                if args.max_tracking_error > 0 and error > args.max_tracking_error:
                    if over_limit_since is None:
                        over_limit_since = time.monotonic()
                    if time.monotonic() - over_limit_since > TRACKING_ERROR_DURATION:
                        return f"tracking error {error:.2f} rad above {args.max_tracking_error:.2f} rad"
                else:
                    over_limit_since = None

                # Hold on tracking loss; the speed limit ramps back in when data returns
                lost = (not device.is_connected
                        or time.monotonic() - last_action_time[0] > TRACKING_LOST_TIMEOUT)
                if engaged.is_set() and lost:
                    engaged.clear()
                    print("Tracking lost: holding.")
                elif not engaged.is_set() and not lost:
                    engaged.set()
                    print("Tracking resumed.")

        ik_thread = threading.Thread(target=ik_loop)
        def report_stats():
            summary = controller.stop_stats(
                args.max_tracking_error,
                log_path=args.log,
                metadata={"csv": str(args.csv), "playback_speed": args.playback_speed},
            )
            if summary:
                print(summary)

        try:
            controller.set_joint_goals({f"q_goal_{side}": READY_RAD for side in SIDES})
            print("\nMoving to the ready pose (Ctrl-C to stop)...")
            wait_for(controller.goals_reached)

            print("Ready pose reached. Waiting for a WebRTC client to connect...")
            wait_for(lambda: device.is_connected)

            last_action_time[0] = time.monotonic()
            engaged.set()
            ik_thread.start()
            controller.start_stats(record=args.log is not None)
            print("Client connected. Tracking.")
            fault = supervise()
            report_stats()

            if fault is not None:
                engaged.clear()
                controller.hold_at_measured()
                print(f"FAULT: {fault}. Holding at the measured pose.")
                print("Ctrl-C to return home (twice to stop here without moving).")
                wait_for(lambda: False)
        except (KeyboardInterrupt, StopSession):
            print("\nStopping.")
        finally:
            report_stats()  # before homing, which is not part of tracking
            stop_event.set()
            engaged.clear()
            if ik_thread.is_alive():
                ik_thread.join()

        return_home(controller, control_step, command_grippers)


def return_home(controller, control_step, command_grippers):
    """Ramp the selected arms back to home so the motors can be switched off safely."""
    goals = {f"q_goal_{side}": HOME_RAD for side in SIDES}
    if command_grippers:
        goals.update({f"{side}_gripper_val": 0.0 for side in SIDES})  # closed
    controller.set_joint_goals(goals)
    deadline = time.monotonic() + controller.seconds_to_goals() + 3.0
    print("Returning home (Ctrl-C again to stop here)...")
    try:
        while not controller.goals_reached() and time.monotonic() < deadline:
            control_step()
        if controller.goals_reached():
            settle_until = time.monotonic() + 0.5
            while time.monotonic() < settle_until:
                control_step()
            print("At home. The bringup can be stopped now (the motors switch off where they are).")
        else:
            print("Warning: home was not reached in time; the arm holds its last command.")
    except KeyboardInterrupt:
        print("\nHoming skipped; the arm holds its last command.")


if __name__ == "__main__":
    main()
    print("Demo completed.")
