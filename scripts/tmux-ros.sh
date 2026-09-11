#!/bin/bash

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Name of the tmux session
SESSION="ros_workspace"

ROS_SETUP="source '$ROOT/scripts/env.sh'; clear"

# Start a new session in bash (env.sh is bash-only), but don't attach to it yet
tmux new-session -d -s $SESSION -c "$ROOT" bash

# Split into 4 quadrants
tmux split-window -h -c "$ROOT" bash
tmux split-window -v -c "$ROOT" bash
tmux select-pane -t 1
tmux split-window -v -c "$ROOT" bash

# Send the ROS source command to all 4 panes
tmux send-keys -t 1 "$ROS_SETUP" C-m
tmux send-keys -t 2 "$ROS_SETUP" C-m
tmux send-keys -t 3 "$ROS_SETUP" C-m
tmux send-keys -t 4 "$ROS_SETUP" C-m

# Attach to the newly created session
tmux attach-session -t $SESSION
