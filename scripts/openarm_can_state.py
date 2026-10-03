#!/usr/bin/env python3
"""
Read the state of every OpenArm motor straight from the CAN buses, without ROS.

    ./scripts/openarm_can_state.py            # ROS stopped: ask each motor for its state
    ./scripts/openarm_can_state.py --listen   # ROS running: decode the driver's traffic only
    ./scripts/openarm_can_state.py --watch    # keep refreshing (Ctrl-C to stop)

Right arm on can0, left arm on can1; motor IDs 1-7 are the joints, 8 is the gripper. Only
read requests are sent (the same "report state" request the driver uses), and none at all
with --listen or when the bus is already busy, so it never adds traffic to a running driver.
"""

import argparse
import math
import socket
import struct
import time

BUSES = {"can0": "right", "can1": "left"}
MOTORS = {1: "DM8009", 2: "DM8009", 3: "DM4340", 4: "DM4340",
          5: "DM4310", 6: "DM4310", 7: "DM4310", 8: "DM4310"}
# Encoding ranges (pMax rad, vMax rad/s, tMax Nm), as in openarm_can's MOTOR_LIMIT_PARAMS
LIMITS = {"DM8009": (12.5, 45.0, 54.0), "DM4340": (12.5, 10.0, 28.0), "DM4310": (12.5, 30.0, 10.0)}
STATUS = {0: "disabled", 1: "enabled", 0x8: "FAULT over-voltage", 0x9: "FAULT under-voltage",
          0xA: "FAULT over-current", 0xB: "FAULT MOS over-temp", 0xC: "FAULT rotor over-temp",
          0xD: "FAULT lost comms", 0xE: "FAULT overload"}
# Gripper motor angle at full opening (0.044 m): stock -1.0472 rad; the lab's left gripper is
# mirrored (gripper_open_sign = +1 in the driver patch)
GRIPPER_OPEN_RAD = {"right": -1.0472, "left": 1.0472}
GRIPPER_OPEN_M = 0.044


def open_bus(interface):
    """Raw CAN socket that receives only motor replies (IDs 0x10-0x1F)."""
    sock = socket.socket(socket.AF_CAN, socket.SOCK_RAW, socket.CAN_RAW)
    sock.setsockopt(socket.SOL_CAN_RAW, socket.CAN_RAW_FILTER, struct.pack("=II", 0x10, 0x7F0))
    sock.bind((interface,))
    sock.settimeout(0.01)
    return sock


def to_float(value, span, bits):
    return value / ((1 << bits) - 1) * 2 * span - span


def decode(frame):
    """(motor id, state) from a reply frame, or None for anything that is not a state reply."""
    can_id, dlc, data = struct.unpack("=IB3x8s", frame)
    motor_id = can_id - 0x10
    if dlc != 8 or motor_id not in MOTORS or data[2] in (0x33, 0x55):  # 0x33/0x55: register replies
        return None
    p_max, v_max, t_max = LIMITS[MOTORS[motor_id]]
    return motor_id, {
        "q": to_float((data[1] << 8) | data[2], p_max, 16),
        "dq": to_float((data[3] << 4) | (data[4] >> 4), v_max, 12),
        "tau": to_float(((data[4] & 0xF) << 8) | data[5], t_max, 12),
        "t_mos": data[6],
        "t_rotor": data[7],
        "status": data[0] >> 4,
    }


def collect(sock, seconds, states):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        try:
            decoded = decode(sock.recv(16))
        except socket.timeout:
            continue
        if decoded:
            states[decoded[0]] = decoded[1]


def request_states(sock, states, attempts=3):
    # Re-ask motors that did not answer: an occasional reply goes missing in a burst
    for _ in range(attempts):
        missing = [motor_id for motor_id in MOTORS if motor_id not in states]
        if not missing:
            return
        for motor_id in missing:
            # Standard "report state" request, as sent by openarm_can's refresh
            sock.send(struct.pack("=IB3x8s", 0x7FF, 8, bytes([motor_id, 0, 0xCC, 0, 0, 0, 0, 0])))
        collect(sock, 0.03, states)


def format_table(states_by_bus, listening):
    lines = [f"{'arm':6s} {'joint':8s} {'motor':7s} {'position':>22s} {'vel rad/s':>10s} "
             f"{'torque Nm':>10s} {'temp mos/rotor':>15s}  status"]
    for interface, side in BUSES.items():
        states = states_by_bus.get(interface)
        if states is None:
            lines.append(f"{side:6s} ({interface} not available)")
            continue
        for motor_id, motor in MOTORS.items():
            joint = "gripper" if motor_id == 8 else f"joint{motor_id}"
            state = states.get(motor_id)
            if state is None:
                lines.append(f"{side:6s} {joint:8s} {motor:7s} {'no reply':>22s}")
                continue
            position = f"{math.degrees(state['q']):+7.1f} deg"
            if motor_id == 8:
                opening = GRIPPER_OPEN_M * state["q"] / GRIPPER_OPEN_RAD[side]
                position = f"({opening * 1000:+5.1f} mm) " + position
            status = STATUS.get(state["status"], f"status {state['status']:#x}")
            lines.append(f"{side:6s} {joint:8s} {motor:7s} {position:>22s} {state['dq']:+10.2f} "
                         f"{state['tau']:+10.2f} {state['t_mos']:>9d}/{state['t_rotor']:<3d} C  {status}")
    lines.append("(listening to the driver's traffic)" if listening else "(queried directly)")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Read all OpenArm motor states from the CAN buses")
    parser.add_argument("--listen", action="store_true", help="Only decode existing traffic (use while ROS runs)")
    parser.add_argument("--watch", action="store_true", help="Keep refreshing every 0.5 s")
    args = parser.parse_args()

    sockets = {}
    for interface in BUSES:
        try:
            sockets[interface] = open_bus(interface)
        except OSError as e:
            print(f"{interface}: {e}")

    listening = args.listen
    if not listening:
        # If motors are already replying, a driver is running: listen instead of adding requests
        busy = {}
        for sock in sockets.values():
            collect(sock, 0.1, busy)
        if busy:
            print("The bus is busy (a bringup is running?); listening instead of querying.")
            listening = True

    try:
        while True:
            states_by_bus = {}
            for interface, sock in sockets.items():
                states = {}
                if listening:
                    collect(sock, 0.1, states)
                else:
                    request_states(sock, states)
                states_by_bus[interface] = states
            if args.watch:
                print("\033[H\033[J", end="")  # clear the terminal
            print(format_table(states_by_bus, listening))
            if not args.watch:
                break
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        for sock in sockets.values():
            sock.close()


if __name__ == "__main__":
    main()
