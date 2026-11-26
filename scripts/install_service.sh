#!/usr/bin/env bash
# Run the monitor as a service on any Linux machine with systemd (a Jetson, a
# Raspberry Pi, any edge board, a PC or server): started at boot, restarted
# within 10 s if it ever stops. Install the project first: ./setup.sh
#
#   scripts/install_service.sh              # install and start
#   scripts/install_service.sh --dry-run    # print the service file only
#   scripts/install_service.sh --uninstall  # stop and remove
#
# What it plays comes from .env: SOURCE=Reception (a CCTV camera named there),
# SOURCE=0 (USB camera) or a video file; empty = the first demo clip.
# Afterwards:
#   systemctl --user status action-monitor       is it running?
#   journalctl --user -u action-monitor -f       its output, live
#   systemctl --user restart action-monitor      after changing .env or training
# A user service: no sudo needed, except once to keep it running when nobody
# is logged in (the command is printed at the end).
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$PWD"
NAME=action-monitor
UNIT="$HOME/.config/systemd/user/$NAME.service"

if [ "${1:-}" = "--uninstall" ]; then
  systemctl --user disable --now "$NAME.service" 2>/dev/null || true
  rm -f "$UNIT"
  systemctl --user daemon-reload
  echo "Removed the $NAME service."
  exit 0
fi

command -v systemctl >/dev/null || { echo "This system has no systemd: start ./run.sh from your init system instead."; exit 1; }
[ -x "$ROOT/.venv/bin/python" ] || { echo "Not installed yet: run ./setup.sh first."; exit 1; }
START="$ROOT/run.sh"

SERVICE="[Unit]
Description=Human Action & Behavior Monitor
After=network-online.target
Wants=network-online.target

[Service]
WorkingDirectory=$ROOT
ExecStart=$START
Restart=always
RestartSec=10
# Models take up to a minute to load on a small board.
TimeoutStartSec=180
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=default.target
"

if [ "${1:-}" = "--dry-run" ]; then
  echo "Would write $UNIT:"
  echo
  echo "$SERVICE"
  exit 0
fi

mkdir -p "$(dirname "$UNIT")"
printf '%s' "$SERVICE" > "$UNIT"
systemctl --user daemon-reload
systemctl --user enable --now "$NAME.service"
echo "The monitor runs as the '$NAME' service: http://$(hostname -I 2>/dev/null | awk '{print $1}'):${API_PORT:-8000}"
if [ "$(loginctl show-user "$USER" -p Linger --value 2>/dev/null)" != "yes" ]; then
  echo
  echo "To start it at boot even when nobody logs in, run once:"
  echo "    sudo loginctl enable-linger $USER"
fi
