# Arch package (CachyOS, Arch Linux)

This folder builds Granular Music Visualizer (GMV) as a native Pacman package. Installed, GMV
is an ordinary desktop application: it is in the application menu with its icon, the
commands `gmv-visualizer` and `gmv` are in `/usr/bin`, it runs on the system's own Python and
Qt, and Pacman updates and removes it. The package is built on your own machine; it is not on
the AUR or in any repository.

| File | What it is |
|---|---|
| `PKGBUILD` | The recipe `makepkg` follows |
| `granular-music-visualizer.desktop` | The application menu entry |
| `icons/` | The icon at 16 to 256 pixels (the 512 one is `packaging/icon.png`) |
| `make_icons.py` | Makes `icons/` from `packaging/icon.png`; run it only when the icon changes |

## Build and install

You need `base-devel` (CachyOS has it) and `git`. Run these as yourself, **never as root**:

```sh
git clone https://github.com/Ralkero/granular-audio-visualizer.git
cd granular-audio-visualizer/packaging/arch
makepkg --syncdeps --cleanbuild
sudo pacman -U granular-music-visualizer-*.pkg.tar.zst
```

- `makepkg` downloads the commit named in the `PKGBUILD` from GitHub and refuses to go on if
  its SHA-256 differs from the one written there. It builds from that download, not from the
  files of your checkout, so local changes are not in the package.
- `--syncdeps` installs what the build needs from the Arch repositories (`python-build`,
  `python-installer`, `python-setuptools`, and GMV's own dependencies), asking for your
  password and for a yes first. `--cleanbuild` starts from an empty build folder.
- The build makes a wheel with the system's Python, installs it into a scratch folder and
  checks that copy (it must find all of Radiant's shaders and import its commands) before
  the package is made.
- `pacman -U` asks for your password and installs the file. `makepkg --syncdeps --install`
  does both steps in one.

The fish shell runs these commands as they are; the `PKGBUILD` itself is read by Bash, as
every `PKGBUILD` is.

## Start it

- From the application menu: *Multimedia → Granular Music Visualizer* in KDE Plasma.
- From a terminal, in any folder: `gmv-visualizer`. The command line is `gmv`
  (`gmv visualizer --replay SESSION`, `gmv file track.flac`, `gmv capture --seconds 30`,
  `gmv replay SESSION/messages.jsonl`, `gmv devices`). `gmv-control` opens the same window
  under its older name.

System audio needs a PulseAudio-compatible sound server: `pipewire-pulse`, which CachyOS
runs already, or `pulseaudio`. The package installs `libpulse` for the `pactl` and `parec`
programs it uses, but never a sound server. Without one, GMV still starts, analyses audio
files and replays sessions.

## Your files

| What | Where |
|---|---|
| Settings | `~/.config/GranularMusicVisualizer/` |
| Recorded sessions | `~/.local/share/GranularMusicVisualizer/sessions/` |
| Your own model files | `~/.local/share/GranularMusicVisualizer/models/` |
| Cache | `~/.cache/GranularMusicVisualizer/` |

(These follow `XDG_CONFIG_HOME`, `XDG_DATA_HOME` and `XDG_CACHE_HOME` when they are set.) The
package installs none of them and has no model weights. Updating or removing it leaves them
as they are.

## Update

Pull the repository and build again; install the new file with `sudo pacman -U`. Pacman
replaces the program and keeps your settings, sessions and models.

## Remove

```sh
sudo pacman -R granular-music-visualizer
```

Your settings, sessions and models stay. To remove them too, delete the three
`GranularMusicVisualizer` folders listed above yourself.

## For the maintainer: moving the recipe to a newer commit

The `PKGBUILD` builds one commit, named in `_commit`, so a package is the same whoever builds
it. To package a newer one:

1. Push the commit, so GitHub can serve it.
2. Set `_commit` to its full hash, and `pkgver` to `git describe --long --tags <commit>`
   with the leading `v` dropped and dashes turned into dots and `r`/`g`: `v0.2.0-7-g5271e33`
   becomes `0.2.0.r7.g5271e33`. A release tag such as `v0.2.1` gives `0.2.1` for that commit
   itself. Set `pkgrel=1`.
3. Run `updpkgsums` (from `pacman-contrib`) or `makepkg -g` and put the SHA-256 it prints in
   `sha256sums`. Never use `SKIP`.
4. Build and install it as above.

The recipe cannot name the commit that contains it, so the commit it builds is always an
earlier one: the packaging files it installs (menu entry, icons) come from that commit.
