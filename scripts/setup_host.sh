#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

sudo apt update
sudo apt install -y \
    software-properties-common \
    jq \
    can-utils \
    python3-vcstool \
    ros-dev-tools \
    ros-humble-ros2-control \
    ros-humble-ros2-controllers \
    ros-humble-mujoco-ros2-control \
    ros-humble-xacro \
    ros-humble-foxglove-bridge \
    ros-humble-rosbag2-storage-mcap

if ! grep -Rqs '^deb .*ppa.launchpadcontent.net/openarm/main' /etc/apt/sources.list /etc/apt/sources.list.d 2>/dev/null; then
    sudo add-apt-repository -y ppa:openarm/main
    sudo apt update
fi

sudo apt install -y libopenarm-can-dev openarm-can-utils

if [[ ! -f /etc/ros/rosdep/sources.list.d/20-default.list ]]; then
    sudo rosdep init
fi
rosdep update

# Python 3.10 venv on the system interpreter, with system site-packages so
# rclpy, colcon and the other apt-installed ROS packages stay importable.
# Python dependencies are declared in pyproject.toml.
if [[ ! -f "$ROOT/.venv/pyvenv.cfg" ]]; then
    uv venv --python /usr/bin/python3.10 --system-site-packages "$ROOT/.venv"
elif ! grep -q '^include-system-site-packages = true' "$ROOT/.venv/pyvenv.cfg"; then
    echo "$ROOT/.venv exists without system site-packages; remove it and rerun." >&2
    exit 1
fi
touch "$ROOT/.venv/COLCON_IGNORE"
(cd "$ROOT" && uv sync)

echo "Host dependencies installed."
echo "Run 'source scripts/env.sh' in each bash shell to use ROS 2 Humble and .venv."
