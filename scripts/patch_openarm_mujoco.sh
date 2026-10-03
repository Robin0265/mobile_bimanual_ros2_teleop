#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

MUJOCO_ROOT="$ROOT/src/openarm_mujoco"
FINGER_PATCH="$ROOT/patches/openarm-mujoco-left-finger-position-actuators.patch"

if [[ ! -d "$MUJOCO_ROOT" ]]; then
    echo "Missing: $MUJOCO_ROOT" >&2
    exit 1
fi

# Upstream drives the left fingers with <motor> actuators in the motor_finger
# class, which inherit its position-actuator ctrlrange (0-0.044): at most
# 0.044 N, opening only. Use position actuators like the right hand.
if git -C "$MUJOCO_ROOT" apply --check "$FINGER_PATCH" 2>/dev/null; then
    git -C "$MUJOCO_ROOT" apply "$FINGER_PATCH"
    echo "Made the left finger actuators position actuators"
elif git -C "$MUJOCO_ROOT" apply --reverse --check "$FINGER_PATCH" 2>/dev/null; then
    echo "Left finger actuator patch already applied"
else
    echo "openarm_mujoco no longer matches $FINGER_PATCH" >&2
    exit 1
fi
