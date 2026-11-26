#!/usr/bin/env bash
# Granular Music Visualizer (GMV) on a Steam Deck: install, check, remove.
#
#   bash gmv-deck.sh install          install GMV, or update it to the tested build
#   bash gmv-deck.sh check            what this Deck has for GMV (changes nothing)
#   bash gmv-deck.sh smoke            self-test: the window opens and closes by itself (1-2 min)
#   bash gmv-deck.sh listen [SECONDS] measure what the Deck is playing (default 15 s)
#   bash gmv-deck.sh remove           remove GMV and its menu entry (your settings stay)
#
# Everything goes into one folder in your home, ~/granular-music-visualizer, plus a menu
# entry and its icon in ~/.local/share. Nothing needs root or a password, nothing touches
# SteamOS's read-only system, so SteamOS updates leave GMV alone. The guide is
# docs/STEAM_DECK_DEMO.md in the repository; the notes for a Claude session on the Deck are
# docs/STEAM_DECK_FOR_CLAUDE.md.
#
# Options for install:
#   --recognition   also install ONNX Runtime, for a context recognition model (.json)
#   --ref REF       install another branch, tag or commit instead of the tested build;
#                   its download cannot be checked against a known checksum
#
# GMV_DECK_HOME names another install folder. Written for SteamOS 3.8 and newer on x86-64;
# works the same on other x86-64 Linux desktops with glibc 2.34 or newer.
#
# Copyright (C) 2026 Ralkero. Licensed under the GNU General Public License, version 3 or
# later (see LICENSE in the repository).

set -euo pipefail

REPOSITORY=Ralkero/granular-audio-visualizer
# The build installed by default: one commit of main, downloaded from GitHub and checked
# against its SHA-256, as the Arch package does (packaging/arch/PKGBUILD). It is the code
# validated on CachyOS; see docs/LINUX.md. To move it: set the full commit hash, download
# https://github.com/$REPOSITORY/archive/COMMIT.tar.gz and put its sha256sum here.
GMV_COMMIT=cdc75e8758bdfb62bf747cc9470e53602c9d2abc
GMV_SHA256=2794269a742825d77175b651cd2f7c399f74c34711722c00f76868543191d50f
# uv sets up Python and GMV's libraries from the repository's uv.lock. A copy of its own is
# kept in the install folder; the SHA-256 is the one uv publishes with the release.
UV_VERSION=0.13.10
UV_SHA256=bf65ac1a0af10f92c522d91ac096c7c0a39657647b155577e7836c2cbbb4526c
PYTHON_VERSION=3.14
APP_ID=granular-music-visualizer
MARKER="X-GMV-Installed-By=gmv-deck.sh"

ROOT=${GMV_DECK_HOME:-$HOME/granular-music-visualizer}
case ${XDG_DATA_HOME:-} in
  /*) DATA_HOME=$XDG_DATA_HOME ;;
  *) DATA_HOME=$HOME/.local/share ;;  # unset, empty or relative: the specification's default
esac
DESKTOP_FILE=$DATA_HOME/applications/$APP_ID.desktop
ICON_SIZES="16 22 24 32 48 64 128 256 512"

say() { printf '%s\n' "$*"; }
step() { printf '\n== %s\n' "$*"; }
die() {
  printf '\nGMV: %s\n' "$*" >&2
  exit 1
}

TEMPORARY=""
cleanup() { if [ -n "$TEMPORARY" ]; then rm -rf "$TEMPORARY"; fi; }
trap cleanup EXIT

usage() { sed -n '2,8p' "$0" | sed 's/^# \{0,1\}//'; }

fetch() {  # URL FILE
  curl --fail --location --silent --show-error --retry 3 --connect-timeout 20 --output "$2" "$1" ||
    die "Could not download $1. Is the Deck online? Run the same command again to retry."
}

verify() {  # FILE SHA256
  printf '%s  %s\n' "$2" "$1" | sha256sum --check --status ||
    die "The download $(basename "$1") is not the expected file (checksum differs). Nothing was installed. Try again later."
}

ours() {  # whether ROOT was made by this script
  [ -f "$ROOT/BUILD" ] && grep -q '^installed by: gmv-deck.sh' "$ROOT/BUILD"
}

installed() {
  ours && [ -x "$ROOT/venv/bin/gmv-visualizer" ] && [ -d "$ROOT/app/src/gmv" ]
}

need_installed() {
  installed || die "GMV is not installed in $ROOT yet. Run: bash gmv-deck.sh install"
}

uv_environment() {
  # Python, the libraries, uv's cache and the environment all stay in the install folder,
  # and no settings of uv elsewhere on the machine are read.
  export UV_PROJECT_ENVIRONMENT="$ROOT/venv"
  export UV_PYTHON_INSTALL_DIR="$ROOT/python"
  export UV_PYTHON_BIN_DIR="$ROOT/python/bin"
  export UV_CACHE_DIR="$ROOT/cache"
  export UV_PYTHON_PREFERENCE=only-managed
  export UV_NO_CONFIG=1
}

# -- install -------------------------------------------------------------------------------

install_uv() {
  local uv=$ROOT/uv/uv
  if [ -x "$uv" ] && [ "$("$uv" --version 2>/dev/null | cut -d' ' -f2)" = "$UV_VERSION" ]; then
    say "uv $UV_VERSION is already there."
    return
  fi
  local archive=uv-x86_64-unknown-linux-gnu.tar.gz
  fetch "https://github.com/astral-sh/uv/releases/download/$UV_VERSION/$archive" "$TEMPORARY/$archive"
  verify "$TEMPORARY/$archive" "$UV_SHA256"
  tar -xzf "$TEMPORARY/$archive" -C "$TEMPORARY"
  mkdir -p "$ROOT/uv"
  mv -f "$TEMPORARY/uv-x86_64-unknown-linux-gnu/uv" "$uv"
  say "uv $UV_VERSION downloaded and checked."
}

download_gmv() {  # REF ("" for the tested build); leaves the source in $TEMPORARY/app
  local url=https://github.com/$REPOSITORY/archive/${1:-$GMV_COMMIT}.tar.gz
  fetch "$url" "$TEMPORARY/gmv.tar.gz"
  if [ -z "$1" ]; then
    verify "$TEMPORARY/gmv.tar.gz" "$GMV_SHA256"
    COMMIT=$GMV_COMMIT
    say "GMV ${GMV_COMMIT:0:7} downloaded and checked."
  else
    # GitHub writes the commit into the archive's header.
    COMMIT=$( (gzip -dc "$TEMPORARY/gmv.tar.gz" | head -c 4096 | tr -d '\0' |
      grep -ao 'comment=[0-9a-f]\{40\}' | cut -d= -f2) 2>/dev/null || true)
    say "GMV '$1' (commit ${COMMIT:-unknown}) downloaded. It is not the tested build and has no known checksum."
  fi
  mkdir "$TEMPORARY/app"
  tar -xzf "$TEMPORARY/gmv.tar.gz" -C "$TEMPORARY/app" --strip-components=1
  [ -f "$TEMPORARY/app/uv.lock" ] && [ -d "$TEMPORARY/app/src/gmv" ] ||
    die "The download does not look like GMV's repository. Nothing was installed."
}

write_launcher() {
  cat >"$ROOT/launch.sh" <<'LAUNCHER'
#!/bin/sh
# Starts Granular Music Visualizer as gmv-deck.sh installed it. Started from the menu or from
# Steam there is no terminal, so what the program says goes to last-launch.log beside this
# file (replaced at each start). Started from a terminal, it is shown there instead.
here=$(dirname "$(readlink -f "$0")")
program="$here/venv/bin/gmv-visualizer"

# Steam can start its shortcuts with the library path of its own runtime, whose older
# libraries are not the ones GMV and the system expect. Put back the path from before Steam.
reset=""
case ${LD_LIBRARY_PATH-} in
  *steam-runtime* | *ubuntu12_32* | *ubuntu12_64*)
    LD_LIBRARY_PATH=${SYSTEM_LD_LIBRARY_PATH-}
    if [ -n "$LD_LIBRARY_PATH" ]; then export LD_LIBRARY_PATH; else unset LD_LIBRARY_PATH; fi
    reset="yes (Steam's runtime library path was set)"
    ;;
esac

if [ -t 2 ]; then
  exec "$program" "$@"
fi
{
  echo "GMV started $(date '+%Y-%m-%d %H:%M:%S')"
  for name in XDG_SESSION_TYPE XDG_CURRENT_DESKTOP WAYLAND_DISPLAY DISPLAY \
    GAMESCOPE_WAYLAND_DISPLAY SteamGameId QT_QPA_PLATFORM GMV_QT_PLATFORM; do
    printf '%s=%s\n' "$name" "$(printenv "$name" || echo '(not set)')"
  done
  echo "library path reset: ${reset:-no}"
  echo "--- what the program said"
} >"$here/last-launch.log"
exec "$program" "$@" >>"$here/last-launch.log" 2>&1
LAUNCHER
  chmod 755 "$ROOT/launch.sh"
}

install_menu_entry() {
  local size source
  for size in $ICON_SIZES; do
    if [ "$size" = 512 ]; then
      source=$ROOT/app/packaging/icon.png
    else
      source=$ROOT/app/packaging/arch/icons/$size.png
    fi
    if [ -f "$source" ]; then
      mkdir -p "$DATA_HOME/icons/hicolor/${size}x$size/apps"
      cp -f "$source" "$DATA_HOME/icons/hicolor/${size}x$size/apps/$APP_ID.png"
    fi
  done
  # The entry of the Arch package, started through the launcher. Its file name is the name
  # the program gives itself on Wayland, so the taskbar shows its name and icon.
  mkdir -p "$DATA_HOME/applications"
  sed "s|^Exec=.*|Exec=$ROOT/launch.sh|" "$ROOT/app/packaging/arch/$APP_ID.desktop" >"$DESKTOP_FILE"
  printf '%s\n' "$MARKER" >>"$DESKTOP_FILE"
  if command -v update-desktop-database >/dev/null; then
    update-desktop-database "$DATA_HOME/applications" >/dev/null 2>&1 || true
  fi
  if command -v kbuildsycoca6 >/dev/null; then kbuildsycoca6 >/dev/null 2>&1 || true; fi
}

cmd_install() {
  local ref="" extras=()
  while [ $# -gt 0 ]; do
    case $1 in
      --ref)
        [ $# -ge 2 ] || die "--ref needs a branch, tag or commit."
        ref=$2
        shift 2
        ;;
      --recognition)
        extras=(--extra recognition)
        shift
        ;;
      *) die "Unknown option for install: $1" ;;
    esac
  done
  [ "$(id -u)" -ne 0 ] || die "Run this as yourself, without sudo: GMV installs in your home folder."
  [ "$(uname -s)" = Linux ] && [ "$(uname -m)" = x86_64 ] || die "This is for x86-64 Linux (a Steam Deck is)."
  case $ROOT in
    /*) ;;
    *) die "The install folder must be an absolute path: $ROOT" ;;
  esac
  case $ROOT in
    *[!A-Za-z0-9._/-]*) die "The install folder's path may only hold letters, digits and . _ - /: $ROOT" ;;
  esac
  local program
  for program in curl tar gzip sha256sum; do
    command -v "$program" >/dev/null || die "The program $program is missing; it is part of SteamOS."
  done
  if [ -e "$ROOT" ] && ! ours && [ -n "$(ls -A "$ROOT" 2>/dev/null)" ]; then
    die "$ROOT exists and was not made by this script. Move it away, or set GMV_DECK_HOME to another folder."
  fi
  local updating=""
  installed && updating=yes

  mkdir -p "$ROOT"
  TEMPORARY=$(mktemp -d "$ROOT/.download.XXXXXX")
  say "Installing Granular Music Visualizer into $ROOT"
  say "This downloads about 200 MB and takes a few minutes. Leave this window open."

  step "1/5  uv, the tool that sets up Python"
  install_uv

  step "2/5  GMV itself"
  download_gmv "$ref"
  rm -rf "$ROOT/app.previous"
  if [ -d "$ROOT/app" ]; then mv "$ROOT/app" "$ROOT/app.previous"; fi
  mv "$TEMPORARY/app" "$ROOT/app"

  step "3/5  Python $PYTHON_VERSION and GMV's libraries (the longest step)"
  uv_environment
  if ! "$ROOT/uv/uv" sync --project "$ROOT/app" --locked --python "$PYTHON_VERSION" \
    --compile-bytecode --quiet ${extras[@]+"${extras[@]}"}; then
    if [ -d "$ROOT/app.previous" ]; then
      rm -rf "$ROOT/app"
      mv "$ROOT/app.previous" "$ROOT/app"
    fi
    die "Setting up Python and the libraries failed (see the lines above). Check the Deck is online and run the same command again."
  fi
  rm -rf "$ROOT/app.previous"

  step "4/5  Launcher, menu entry and icon"
  write_launcher
  cp -f "$0" "$ROOT/gmv-deck.sh" 2>/dev/null || true
  chmod 755 "$ROOT/gmv-deck.sh" 2>/dev/null || true
  install_menu_entry
  {
    echo "installed by: gmv-deck.sh"
    echo "when: $(date '+%Y-%m-%d %H:%M:%S')"
    if [ -z "$ref" ]; then
      echo "build: the tested build, commit $GMV_COMMIT (checksum checked)"
    else
      echo "build: '$ref', commit ${COMMIT:-unknown} (not the tested build; no checksum)"
    fi
    echo "python: $("$ROOT/venv/bin/python" --version 2>&1)"
    echo "uv: $UV_VERSION"
    echo "recognition runtime: $([ ${#extras[@]} -gt 0 ] && echo yes || echo no)"
  } >"$ROOT/BUILD"
  say "Launcher: $ROOT/launch.sh"
  say "Menu entry: $DESKTOP_FILE"

  step "5/5  A quick check"
  QT_QPA_PLATFORM=offscreen "$ROOT/venv/bin/python" - <<'CHECK' || die "GMV was installed but does not import (see above)."
import sys

import PySide6
from PySide6.QtCore import qVersion

from gmv import paths
from gmv.app.window import platform_plugin_problem

print(f"Python {sys.version.split()[0]}, PySide6 {PySide6.__version__}, Qt {qVersion()}")
assert paths.checkout(), "GMV did not find its own folder"
for name in ("wayland", "xcb"):
    problem = platform_plugin_problem(name)
    print(f"Qt's {name} display plugin: {problem or 'loads'}")
CHECK

  say ""
  say "Done: GMV is ${updating:+updated and }ready. Your settings and sessions are kept apart from the program."
  say "Start it from the application menu (Multimedia > Granular Music Visualizer),"
  say "or here with: $ROOT/launch.sh"
}

# -- check ---------------------------------------------------------------------------------

library() {  # NAME -> path or "missing"
  local ldconfig found
  ldconfig=$(command -v ldconfig || echo /usr/bin/ldconfig)
  # awk reads to the end (no early exit), so ldconfig is never cut off mid-write.
  found=$("$ldconfig" -p 2>/dev/null |
    awk -v n="$1" '$1 == n && /x86-64/ && !seen {print $NF; seen = 1}') || true
  echo "${found:-missing}"
}

cmd_check() {
  step "This machine"
  local name="unknown" version=""
  if [ -r /etc/os-release ]; then
    name=$(. /etc/os-release && echo "${PRETTY_NAME:-${NAME:-unknown}}")
    version=$(. /etc/os-release && echo "${VERSION_ID:-}${BUILD_ID:+ (build $BUILD_ID)}")
  fi
  say "System: $name $version, $(uname -m), kernel $(uname -r)"
  say "C library: $(getconf GNU_LIBC_VERSION 2>/dev/null || echo unknown) (GMV's Qt needs glibc 2.34 or newer)"
  say "Free space in the home folder: $(df -h --output=avail "$HOME" 2>/dev/null | tail -n1 | tr -d ' ')"

  step "Display (Desktop Mode is normally Wayland; Game Mode is Gamescope)"
  local variable
  for variable in XDG_SESSION_TYPE XDG_CURRENT_DESKTOP WAYLAND_DISPLAY DISPLAY GAMESCOPE_WAYLAND_DISPLAY; do
    say "$variable=$(printenv "$variable" || echo '(not set)')"
  done

  step "Libraries GMV's Qt uses from the system"
  for name in libxcb-cursor.so.0 libxkbcommon.so.0 libxkbcommon-x11.so.0 libwayland-client.so.0 \
    libEGL.so.1 libGL.so.1 libfontconfig.so.1 libdbus-1.so.3; do
    say "$name: $(library "$name")"
  done

  step "Sound (System audio needs pactl, parec and PipeWire's PulseAudio service)"
  for name in pactl parec; do
    say "$name: $(command -v "$name" || echo missing)"
  done
  if command -v pactl >/dev/null && LC_ALL=C pactl info >/dev/null 2>&1; then
    say "Sound server: $(LC_ALL=C pactl info | sed -n 's/^Server Name: //p')"
    local sink
    sink=$(LC_ALL=C pactl get-default-sink 2>/dev/null || true)
    if [ -z "$sink" ]; then
      say "Default output: none found"
    elif LC_ALL=C pactl list short sources | awk '{print $2}' | grep -Fx "$sink.monitor" >/dev/null; then
      say "Default output: found, with the monitor GMV records from"
    else
      say "Default output: found, but without a monitor"
    fi
  else
    say "Sound server: did not answer"
  fi

  step "GMV"
  if installed; then
    sed 's/^/  /' "$ROOT/BUILD"
    [ -f "$DESKTOP_FILE" ] && say "Menu entry: $DESKTOP_FILE" || say "Menu entry: missing"
    if [ -f "$ROOT/last-launch.log" ]; then
      say "Last start from the menu or Steam ($ROOT/last-launch.log):"
      sed 's/^/  /' "$ROOT/last-launch.log" | tail -n 25
    fi
  else
    say "Not installed in $ROOT."
  fi
  local user_models=$DATA_HOME/GranularMusicVisualizer/models
  local models
  models=$(ls "$user_models" 2>/dev/null | tr '\n' ' ') || true
  say "Your model files ($user_models): ${models:-none}"
}

# -- smoke and listen ----------------------------------------------------------------------

cmd_smoke() {
  need_installed
  say "The GMV window will open and close by itself a few times, for about two minutes."
  say "It analyses generated test sounds, records and replays them, and shows Radiant."
  say "It uses scratch folders: your own settings and sessions are not touched, and nothing is recorded."
  say "Leave it alone until it prints how many checks passed."
  TEMPORARY=$(mktemp -d)
  (cd "$TEMPORARY" && "$ROOT/venv/bin/python" "$ROOT/app/scripts/validate_linux.py")
}

cmd_listen() {
  local seconds=${1:-15}
  case $seconds in
    '' | *[!0-9]*) die "listen takes a number of seconds, for example: bash gmv-deck.sh listen 20" ;;
  esac
  need_installed
  say "Play some music on the Deck now. GMV listens to what the Deck plays for $seconds seconds."
  say "It measures the sound only: nothing is played, saved or changed."
  TEMPORARY=$(mktemp -d)
  local status=0
  (cd "$TEMPORARY" && "$ROOT/venv/bin/python" "$ROOT/app/scripts/capture_probe_linux.py" \
    --seconds "$seconds" --every 3 --report "$TEMPORARY/listen.json") || status=$?
  [ -f "$TEMPORARY/listen.json" ] || die "The measurement did not run (see above)."
  "$ROOT/venv/bin/python" - "$TEMPORARY/listen.json" <<'VERDICT'
import json
import sys

report = json.load(open(sys.argv[1], encoding="utf-8"))
problem = report["environment"].get("problem")
print()
if problem:
    print(f"GMV cannot listen here: {problem}")
    sys.exit()
frames = report.get("frames_analysed", 0)
level = (report.get("level") or {}).get("rms_dbfs_median")
silent = report.get("silent_frames", 0)
print(f"GMV heard {frames} analysis frames; median level {level} dBFS; {silent} of them silent.")
if not report.get("passed"):
    print("Some checks failed (marked FAIL above).")
elif level is not None and level > -70 and silent < frames / 2:
    print("It hears the music: System audio works on this Deck.")
else:
    print("It hears only silence. Was music playing, and not muted in the player?")
VERDICT
  return "$status"
}

# -- remove --------------------------------------------------------------------------------

cmd_remove() {
  local answer=""
  if [ "${1:-}" = --yes ]; then
    answer=yes
  elif [ -t 0 ]; then
    say "This removes $ROOT (the program, its Python and libraries),"
    say "the menu entry and the icon. Your settings, sessions and model files stay."
    read -r -p "Type yes to remove GMV: " answer || true
  fi
  [ "$answer" = yes ] || die "Nothing was removed."
  if [ -e "$ROOT" ]; then
    ours || die "$ROOT was not made by this script; it was left alone."
    rm -rf "$ROOT"
    say "Removed $ROOT"
  fi
  if [ -f "$DESKTOP_FILE" ] && grep -qx "$MARKER" "$DESKTOP_FILE"; then
    rm -f "$DESKTOP_FILE"
    say "Removed the menu entry"
  fi
  local size
  for size in $ICON_SIZES; do
    rm -f "$DATA_HOME/icons/hicolor/${size}x$size/apps/$APP_ID.png"
  done
  if command -v kbuildsycoca6 >/dev/null; then kbuildsycoca6 >/dev/null 2>&1 || true; fi
  say "Kept: settings in ${XDG_CONFIG_HOME:-$HOME/.config}/GranularMusicVisualizer,"
  say "sessions and model files in $DATA_HOME/GranularMusicVisualizer."
  say "A shortcut added to Steam stays in the Steam library until you remove it there."
}

main() {
  local command=${1:-help}
  [ $# -gt 0 ] && shift
  case $command in
    install) cmd_install "$@" ;;
    check) cmd_check ;;
    smoke) cmd_smoke ;;
    listen) cmd_listen "$@" ;;
    remove) cmd_remove "$@" ;;
    help | -h | --help) usage ;;
    *)
      usage
      exit 2
      ;;
  esac
}

# The tests (tests/test_steamdeck_helper.py) read this file with GMV_DECK_SOURCED=1 to reach
# single functions; run normally, it does what its command says.
if [ -z "${GMV_DECK_SOURCED:-}" ]; then
  main "$@"
fi
