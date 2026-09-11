# Set up the current shell for ROS 2 Humble with the workspace's uv venv:
#
#   source scripts/env.sh               # ROS 2 Humble + install/ overlay + .venv
#   source scripts/env.sh --no-overlay  # skip install/ (used when building)
#
# Source this file from bash; don't execute it. scripts/setup_host.sh creates
# .venv.

_mobile_bimanual_env() {
    local root overlay=1 nounset=0 status=0
    root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
    [[ "${1:-}" == "--no-overlay" ]] && overlay=0

    # ROS setup files and venv activation reference unset variables.
    if [[ $- == *u* ]]; then
        nounset=1
        set +u
    fi

    if [[ ! -f /opt/ros/humble/setup.bash ]]; then
        echo "env.sh: ROS 2 Humble not found at /opt/ros/humble" >&2
        status=1
    elif [[ ! -f "$root/.venv/bin/activate" ]]; then
        echo "env.sh: missing $root/.venv; run scripts/setup_host.sh" >&2
        status=1
    else
        # Ignore ~/.local site-packages so system Python tools (ros2 CLI,
        # message generation) use apt's numpy 1.x, which Humble is built
        # against, and the venv stays isolated from user-installed packages.
        export PYTHONNOUSERSITE=1
        source /opt/ros/humble/setup.bash
        if ((overlay)) && [[ -f "$root/install/setup.bash" ]]; then
            source "$root/install/setup.bash"
        fi
        # Activate last so the venv's python3 comes first on PATH.
        source "$root/.venv/bin/activate"
    fi

    ((nounset)) && set -u
    return "$status"
}

_mobile_bimanual_env "$@" || { unset -f _mobile_bimanual_env; return 1; }
unset -f _mobile_bimanual_env
