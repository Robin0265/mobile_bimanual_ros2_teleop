#!/usr/bin/env python3
"""
Read-only probe for one OpenArm (Damiao) motor on a classic CAN 2.0 bus.

openarm-can-cli's show_param, monitor, enable and disable only speak CAN-FD (openarm_can
1.3.4), so they get no answer on this project's classic 1 Mbps buses. This reads the same
registers and the live state through openarm_can in classic mode. It only sends read
requests: it never enables the motor or writes a register.

Stop any ROS bringup on the bus first. Examples:
    ./scripts/openarm_motor_probe.py -i can0 --id 8                  # right gripper
    ./scripts/openarm_motor_probe.py -i can1 --id 8 --watch 60       # left gripper, 60 s
    ./scripts/openarm_motor_probe.py -i can0 --id 4 --type DM4340 --watch 0
"""

import argparse
import math
import socket
import struct
import sys
import threading
import time

import openarm_can as oa

REGISTERS = ["CTRL_MODE", "PMAX", "VMAX", "TMAX", "TIMEOUT"]
CTRL_MODES = {1: "MIT", 2: "POS_VEL", 3: "VEL", 4: "POS_FORCE"}
GRIPPER_ID = 8
# openarm_hardware's parallel gripper mapping: motor 0 rad = closed, -1.0472 rad = 0.044 m open
GRIPPER_OPEN_RAD = -1.0472
GRIPPER_OPEN_M = 0.044


class ReplyCounter:
    """Counts frames from the motor's reply ID on a separate raw CAN socket."""

    def __init__(self, interface, recv_id):
        self.count = 0
        self._sock = socket.socket(socket.AF_CAN, socket.SOCK_RAW, socket.CAN_RAW)
        self._sock.setsockopt(
            socket.SOL_CAN_RAW, socket.CAN_RAW_FILTER, struct.pack("=II", recv_id, 0x7FF)
        )
        self._sock.bind((interface,))
        self._sock.settimeout(0.2)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        while not self._stop.is_set():
            try:
                self._sock.recv(16)
                self.count += 1
            except socket.timeout:
                pass

    def close(self):
        self._stop.set()
        self._thread.join()
        self._sock.close()


def get_motor(arm):
    # get_motors() returns copies: fetch again after every recv_all() to see new data
    return arm.get_arm().get_motors()[0]


def read_registers(arm, counter):
    """Print the configuration registers; False if the motor did not answer."""
    arm.set_callback_mode_all(oa.CallbackMode.PARAM)
    replies_before = counter.count
    for name in REGISTERS:
        rid = getattr(oa.MotorVariable, name).value
        for _ in range(3):
            arm.query_param_all(rid)
            arm.recv_all(20000)
            value = get_motor(arm).get_param(rid)
            if value != -1:  # -1 until a reply has been decoded
                break
        mode = ""
        if name == "CTRL_MODE" and math.isfinite(value):
            mode = f" ({CTRL_MODES.get(int(value), 'unknown')})"
        print(f"  {name:9s} = {value:g}{mode}")
    time.sleep(0.05)
    return counter.count > replies_before


def watch_state(arm, counter, seconds, is_gripper):
    """Print live position/torque/temperatures and the range of positions seen."""
    arm.set_callback_mode_all(oa.CallbackMode.STATE)
    print(f"Live state for {seconds:.0f} s (Ctrl-C to stop). The motor stays disabled; "
          "move it by hand.")
    q_min, q_max = math.inf, -math.inf
    end = time.monotonic() + seconds
    try:
        while time.monotonic() < end:
            arm.refresh_all()
            arm.recv_all(20000)
            motor = get_motor(arm)
            q = motor.get_position()
            q_min, q_max = min(q_min, q), max(q_max, q)
            # Short enough for \r to overwrite it instead of wrapping in an 80-column terminal
            print(f"\r  pos {q:+7.3f} rad {math.degrees(q):+7.1f} deg  "
                  f"seen [{q_min:+.3f}, {q_max:+.3f}]  tau {motor.get_torque():+5.2f}  "
                  f"{'ENABLED' if motor.is_enabled() else 'disabled'}  ",
                  end="", flush=True)
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    print()
    if math.isfinite(q_min):
        travel = q_max - q_min
        print(f"  Range seen: {q_min:+.3f} .. {q_max:+.3f} rad (travel {travel:.3f} rad)")
        if is_gripper and travel > 0.1:
            cap = GRIPPER_OPEN_M * min(travel - 0.05, abs(GRIPPER_OPEN_RAD)) / abs(GRIPPER_OPEN_RAD)
            print(f"  If closed reads ~0 and open is negative: --gripper_max_opening {cap:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Read-only probe for one OpenArm motor over classic CAN")
    parser.add_argument("-i", "--interface", default="can0", help="SocketCAN interface (right arm: can0, left: can1)")
    parser.add_argument("--id", default=str(GRIPPER_ID), help="Motor send ID (replies on ID + 0x10); 8 is the gripper")
    parser.add_argument("--type", default="DM4310", choices=["DM8009", "DM4340", "DM4310"], help="Motor type")
    parser.add_argument("--watch", default=30.0, type=float, help="Seconds of live state (0: registers only)")
    args = parser.parse_args()

    send_id = int(args.id, 0)
    recv_id = send_id + 0x10
    try:
        counter = ReplyCounter(args.interface, recv_id)
    except OSError as e:
        print(f"Error: cannot open {args.interface}: {e}")
        return 1

    try:
        arm = oa.OpenArm(args.interface, False)
        # With no control modes, init_arm_motors registers the motor without writing to it
        # (init_gripper_motor would write the control-mode register).
        arm.init_arm_motors([getattr(oa.MotorType, args.type)], [send_id], [recv_id])

        print(f"Motor 0x{send_id:X} ({args.type}, replies on 0x{recv_id:X}) on {args.interface}, "
              "classic CAN")
        if not read_registers(arm, counter):
            print("  [!] No reply from the motor: check its power, cabling and ID, and that no "
                  "other program is using the bus.")
            return 1
        if args.watch > 0:
            watch_state(arm, counter, args.watch, is_gripper=send_id == GRIPPER_ID)
    finally:
        counter.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
