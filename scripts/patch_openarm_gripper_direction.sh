#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# The OpenArm driver assumes every gripper opens toward negative motor angles.
# The lab's left gripper is mirrored and opens toward positive angles, so opening
# commands drove it into its closed stop. The driver patch adds a per-arm
# "gripper_open_sign" hardware parameter; the description patch sets it to +1 for
# the left arm. Run after patch_openarm_can20.sh (it edits the same file).
apply_patch() {
    local repo="$1" patch_file="$2" description="$3"

    if [[ ! -d "$repo" ]]; then
        echo "Missing: $repo" >&2
        exit 1
    fi

    # Check "already applied" first: the left and right hardware blocks share the
    # same context lines, so a forward apply would otherwise patch the right one.
    if git -C "$repo" apply --reverse --check "$patch_file" 2>/dev/null; then
        echo "Already applied: $description"
    elif git -C "$repo" apply --check "$patch_file" 2>/dev/null; then
        git -C "$repo" apply "$patch_file"
        echo "Applied: $description"
    else
        echo "$repo no longer matches $patch_file" >&2
        exit 1
    fi
}

apply_patch "$ROOT/src/openarm_ros2" \
    "$ROOT/patches/openarm-hardware-gripper-open-sign.patch" \
    "per-arm gripper opening direction in openarm_hardware"
apply_patch "$ROOT/src/openarm_description" \
    "$ROOT/patches/openarm-description-left-gripper-mirrored.patch" \
    "left gripper opens toward positive motor angles"
