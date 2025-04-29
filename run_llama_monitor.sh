#!/bin/bash
#####################################################
# run_llama_monitor.sh
#
# Description: This script runs tornado-llama-opencl with GPU monitoring
# It opens separate terminal windows with nvtop and htop for monitoring
# system resources while the model runs.
#
# Usage:
#   ./run_llama_monitor.sh --gpu                          # Run with GPU and default prompt
#   ./run_llama_monitor.sh --java                         # Run without GPU (Java mode)
#   ./run_llama_monitor.sh --gpu --prompt "custom prompt" # Run with GPU and custom prompt
#   ./run_llama_monitor.sh --java --prompt "custom prompt" # Run without GPU and custom prompt
#   ./run_llama_monitor.sh --gpu -s 10                    # Run with 10 second sleep intervals
#   ./run_llama_monitor.sh --gpu -e                       # Run without verbose output
#
# Options:
#   --gpu                Use GPU acceleration
#   --java               Run without GPU (default)
#   --prompt "text"      Specify custom prompt
#   -s, --sleep N        Set sleep interval to N seconds (default: 5)
#   -e, --echo-off       Disable verbose output
#
# Requirements:
#   - nvtop and htop for monitoring
#   - A terminal emulator (xterm, gnome-terminal, konsole, or terminator)
#   - tornado-llama-opencl and model files
#
# Author: Claude
# Date: April 29, 2025
#####################################################

# Default values
GPU_MODE=false  # Default to not using GPU
CUSTOM_PROMPT="hello world in java give me an example code"
VERBOSE=true    # Default to show messages
SLEEP_TIME=5    # Default sleep interval in seconds

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --gpu)
      GPU_MODE=true
      shift
      ;;
    --java)
      GPU_MODE=false  # Explicitly disable GPU when --java is used
      shift
      ;;
    -e|--echo-off)
      VERBOSE=false
      shift
      ;;
    -s|--sleep)
      if [[ $# -gt 1 && ! $2 == --* && $2 =~ ^[0-9]+$ ]]; then
        SLEEP_TIME=$2
        shift 2
      else
        echo "Error: --sleep requires a numeric argument"
        exit 1
      fi
      ;;
    --prompt|--propmt)  # Handle both correct and common misspelling
      if [[ $# -gt 1 && ! $2 == --* ]]; then
        CUSTOM_PROMPT="$2"
        shift 2
      else
        echo "Error: --prompt requires an argument"
        exit 1
      fi
      ;;
    *)
      # Unknown option
      shift
      ;;
  esac
done

# Function for conditional echo
echo_verbose() {
  if [ "$VERBOSE" = true ]; then
    echo "$@"
  fi
}

# GPU option
if [ "$GPU_MODE" = true ]; then
  GPU_FLAG="--gpu"
else
  GPU_FLAG=""
fi

# Function to check if a command exists
check_cmd() {
  command -v "$1" &> /dev/null
}

# Detect if we're in an X environment
if [ -n "$DISPLAY" ]; then
  HAS_X=true
else
  HAS_X=false
  echo_verbose "Warning: No X display detected. Using fallback monitoring mode."
fi

# Simple approach: launch two separate terminal windows
launch_simple_monitors() {
  echo_verbose "Launching separate monitoring windows..."

  # Try different terminal emulators based on what's available
  if check_cmd xterm; then
    TERM_CMD="xterm"
    TERM_ARGS_NVTOP="-T NVTOP -e"
    TERM_ARGS_HTOP="-T HTOP -e"
  elif check_cmd gnome-terminal; then
    TERM_CMD="gnome-terminal"
    TERM_ARGS_NVTOP="--"
    TERM_ARGS_HTOP="--"
  elif check_cmd konsole; then
    TERM_CMD="konsole"
    TERM_ARGS_NVTOP="-e"
    TERM_ARGS_HTOP="-e"
  elif check_cmd terminator; then
    TERM_CMD="terminator"
    TERM_ARGS_NVTOP="-e"
    TERM_ARGS_HTOP="-e"
  else
    echo "No suitable terminal emulator found."
    return 1
  fi

  # Launch nvtop
  if check_cmd nvtop; then
    $TERM_CMD $TERM_ARGS_NVTOP "nvtop" &
    NVTOP_PID=$!
    echo $NVTOP_PID > /tmp/nvtop_pid
    echo_verbose "Launched nvtop with PID $NVTOP_PID"
  else
    echo_verbose "nvtop not found. Please install it."
  fi

  # Small delay between launches
  sleep 1

  # Launch htop
  if check_cmd htop; then
    $TERM_CMD $TERM_ARGS_HTOP "htop" &
    HTOP_PID=$!
    echo $HTOP_PID > /tmp/htop_pid
    echo_verbose "Launched htop with PID $HTOP_PID"
  else
    echo_verbose "htop not found. Please install it."
  fi

  return 0
}

# Check if we can launch monitoring
if [ "$HAS_X" = true ]; then
  echo "Launching monitoring tools..."
  launch_simple_monitors
else
  echo "Cannot launch graphical terminals without X. Running model only."
fi

  # Wait a moment for terminals to open
sleep 5  # Increased sleep to allow time to rearrange windows

# Run the Llama model
echo "Executing command: tornado-llama-opencl $GPU_FLAG --model Llama-3.2-1B-Instruct-Q8_0.gguf --prompt \"$CUSTOM_PROMPT\""
tornado-llama-opencl $GPU_FLAG --model Llama-3.2-1B-Instruct-Q8_0.gguf --prompt "$CUSTOM_PROMPT"

# Give user time to view results before closing the monitoring windows
echo "Llama model execution completed. Monitoring windows will close in 5 seconds..."
sleep 5

# Cleanup - only kill if we actually need to
if [ -f /tmp/nvtop_pid ]; then
  NVTOP_PID=$(cat /tmp/nvtop_pid)
  if ps -p $NVTOP_PID > /dev/null; then
    echo "Cleaning up nvtop process ($NVTOP_PID)"
    kill $NVTOP_PID 2>/dev/null
  fi
  rm -f /tmp/nvtop_pid
fi

if [ -f /tmp/htop_pid ]; then
  HTOP_PID=$(cat /tmp/htop_pid)
  if ps -p $HTOP_PID > /dev/null; then
    echo_verbose "Cleaning up htop process ($HTOP_PID)"
    kill $HTOP_PID 2>/dev/null
  fi
  rm -f /tmp/htop_pid
fi

if [ "$VERBOSE" = true ]; then
  echo "All done! Thank you for using run_llama_monitor.sh"
  echo ""
  echo "================================================================"
  echo "USAGE GUIDE:"
  echo "================================================================"
  echo "Basic usage with GPU: ./run_llama_monitor.sh --gpu"
  echo "Basic usage without GPU: ./run_llama_monitor.sh --java"
  echo "Custom prompt: ./run_llama_monitor.sh --gpu --prompt \"Your prompt here\""
  echo "Custom prompt without GPU: ./run_llama_monitor.sh --java --prompt \"Your prompt here\""
  echo "Quiet mode (no messages): ./run_llama_monitor.sh --gpu -e"
  echo "Custom sleep time: ./run_llama_monitor.sh --gpu -s 10"
  echo ""
  echo "Tips:"
  echo "- Make sure nvtop and htop are installed for proper monitoring"
  echo "- You can rearrange the monitoring windows during the startup delay"
  echo "- The monitoring windows will remain open for a delay after completion"
  echo "- If xterm doesn't work, try installing gnome-terminal or terminator"
  echo "- Use --gpu for GPU acceleration or --java to run without GPU"
  echo "- Use -s NUMBER to change the default sleep time (${SLEEP_TIME} seconds)"
  echo "- Use -e to disable verbose output messages"
  echo "================================================================"
fi