# Granular Music Visualizer on a Steam Deck (for demos)

This guide puts the current tested build of Granular Music Visualizer (GMV) on a Steam Deck,
so you can show it to people. Development stays on the PC; the Deck only runs the program.

> **Status (2026-11-26): not yet tried on a real Steam Deck.** The installer and every step
> below were run on the CachyOS PC, in a stand-in home folder, as a Deck would run them.
> Steps the PC cannot check are marked *untried on a Deck*. The list at the end says
> what to look out for the first time.

**How it is installed.** One helper script puts GMV and its own copy of Python into one
folder in your home folder, `granular-music-visualizer`, and adds GMV to the application
menu. It asks for no password and changes nothing in SteamOS itself, so SteamOS updates leave
GMV alone. Your settings and recordings are kept in separate folders, so updating or removing
GMV keeps them. The build installed is the same program that runs on your CachyOS PC
(`main` at commit `cdc75e8`), and the download is checked to be exactly that.

If you have Claude Code on the Deck, it can do all of this for you: see
[Using Claude Code on the Deck](#using-claude-code-on-the-deck).

## What you need

- A Steam Deck (LCD or OLED) with SteamOS 3.8 or newer, charged or plugged in.
- Wi-Fi. The install downloads about 200 MB and needs about 1 GB of free space.
- About 10 minutes.
- Something to play music on the Deck: Spotify (from the Discover store), a web browser with
  a music site, or music files in a media player. GMV shows whatever the Deck is playing.

Handy on the Deck: **Steam + X** opens the on-screen keyboard. The touchscreen works, the right
trackpad moves the pointer, **R2** clicks and **L2** right-clicks.

## 1. Switch to Desktop Mode

Press the **Steam** button, choose **Power**, then **Switch to Desktop**.

## 2. Install GMV

It is easiest to open this page on the Deck in a web browser (Firefox or Chrome, from
Discover) so you can copy the two commands instead of typing them.

1. Open the application launcher (bottom-left corner), then **System → Konsole**.
2. Paste the first line into Konsole and press **Enter**. To paste in Konsole use
   **Ctrl+Shift+V**, or right-click and choose **Paste**.

   ```bash
   curl -fLo gmv-deck.sh https://raw.githubusercontent.com/Ralkero/granular-audio-visualizer/main/packaging/steamdeck/gmv-deck.sh
   ```

   If it says `404`, this guide has not been merged into `main` yet. Use this line instead:

   ```bash
   curl -fLo gmv-deck.sh https://raw.githubusercontent.com/Ralkero/granular-audio-visualizer/docs/steam-deck-demo/packaging/steamdeck/gmv-deck.sh
   ```

3. Paste the second line and press **Enter**:

   ```bash
   bash gmv-deck.sh install
   ```

4. Wait until it says **Done: GMV is ready**. It counts its steps 1/5 to 5/5. On the PC
   this took under a minute; on the Deck's Wi-Fi expect a few minutes.

If it stops with a message, it says what went wrong. Most often the Deck went offline: run
the same line again, and it carries on.

## 3. Start GMV

Open the application launcher and choose **Multimedia → Granular Music Visualizer**, or type
`Granular` in its search box. The first start takes a few seconds.

## 4. Make it react to music

1. Click **Settings** (top right). Under **Audio source**, choose **System audio (PipeWire)**.
   A new install already has it selected.
2. Click **Start** (top bar).
3. Play music on the Deck in any app. Within a second the picture moves with the music. The
   status line at the top says what GMV is doing.
4. Choose how it looks under **Settings → Visualizer**:
   - **Layout**: *Spectrum bars* or *Radial*.
   - With *Radial*, **Radial theme**: *Classic* or *Radiant*. Radiant is the glowing one,
     drawn by the graphics chip.
   - **Expressive Mode** makes everything livelier.
5. **Full screen (F11)** is at the bottom of the Visualizer settings. **Close** hides the
   settings panel. To leave full screen, press the button again, or press **Esc** on the
   keyboard.

GMV remembers all of this for next time. It saves its analysis of each session (numbers,
never the sound). You can switch that off: **Settings → Sessions**, *Save analysis logs*.

**How to tell it is really listening:** play a song and watch the bars or rings follow the
beat; pause the song and within a second or two the picture goes quiet. If the picture stays still
while music plays, see [If something goes wrong](#if-something-goes-wrong).

## 5. Optional checks

Two checks you can run in Konsole. Neither records sound or touches your settings:

- With music playing:

  ```bash
  bash ~/granular-music-visualizer/gmv-deck.sh listen
  ```

  After 15 seconds it should end with **It hears the music: System audio works on this
  Deck.**
- The full self-test:

  ```bash
  bash ~/granular-music-visualizer/gmv-deck.sh smoke
  ```

  The GMV window opens and closes by itself a few times for about two minutes; leave it
  alone. It ends with *N of N checks passed*. It uses scratch folders, so your settings and
  sessions are not touched.

## 6. Optional: Game Mode (*untried on a Deck*)

GMV is a desktop program. It is expected to run in Game Mode too, but that has not been tried
yet. Desktop Mode is the safe choice for a demo.

1. In Desktop Mode, set GMV up the way you want to show it: System audio, layout, theme,
   colours. Game Mode starts with the same settings.
2. In the application launcher, right-click **Granular Music Visualizer** and choose
   **Add to Steam**. If that option is missing, open Steam, then **Add a Game → Add a
   Non-Steam Game**, tick Granular Music Visualizer and click **Add Selected Programs**.
3. Do the same for your music app (for example Spotify) if you want to start it from Game
   Mode.
4. Double-click **Return to Gaming Mode** on the desktop.
5. Start the music first. Then in the Library, under **Non-Steam**, start **Granular Music
   Visualizer** and press **Start** (touch it, or use the trackpad).
6. The **Steam** button switches between GMV and the music app.

In Game Mode, avoid buttons that open a second window: **Choose…**, **Replay a session…**,
**Export logs…**, the colour pickers and the *Show* menu. Game Mode may hide such windows
behind GMV. Do those in Desktop Mode.

If GMV does not appear in Game Mode, or the screen stays black: in the Library select GMV,
open the gear menu, then **Properties**. In **Launch Options** type
`GMV_QT_PLATFORM=xcb %command%`, then try again. Either way, what GMV said when it started is
in `~/granular-music-visualizer/last-launch.log`; that file is what helps to find the cause.

## Optional: the instrument detectors

GMV works fully without any extra files: all three looks react to the music. What extra files
add is the confirmation of *which* instrument is playing. Without them, the drum, bass, vocal
and guitar layers are drawn pale and dashed. That means "something like this is playing",
not confirmed. **Settings → Analysis** says which detectors are loaded.

- **Drums, bass and vocals** (`activity_v2.npz`). There are two places to get it:
  - your Windows PC: the GMV folder, in `.model-cache` (copy it with a USB stick);
  - the [0.1.1 release](https://github.com/Ralkero/granular-audio-visualizer/releases/tag/v0.1.1):
    download the Windows zip on the Deck (157 MB), open it, and take the file from its
    `models` folder. Whether the 0.1.1 files work with this newer build is *untried*.

  Then, in Dolphin on the Deck:
  1. Open your home folder and press **Ctrl+H** to show hidden folders.
  2. Go into `.local/share/GranularMusicVisualizer/models`. Make the `models` folder if it is
     missing.
  3. Put the file there and restart GMV.
- **Guitar** (`guitar_mn10_v1`: a `.json` file with its `.onnx` file, from the same places).
  - It needs an extra part: run `bash gmv-deck.sh install --recognition`.
  - Put both files in the same models folder, then choose the `.json` file in
    **Settings → Analysis → Guitar → Choose…**.
  - It reports about a second late and is experimental. It has not been tried on any Linux
    system yet.

These files are published for non-commercial use only (licence CC BY-NC-SA 4.0; the details
are in `MODEL-LICENCES.txt` in the release's `models` folder).

## Your settings and files

| What | Where (in your home folder) |
|---|---|
| The program, its Python and libraries | `granular-music-visualizer/` (safe to delete and install again) |
| What GMV said at its last start from the menu or Steam | `granular-music-visualizer/last-launch.log` |
| Settings | `.config/GranularMusicVisualizer/Granular Music Visualizer.conf` |
| Saved sessions (analysis only, never sound) | `.local/share/GranularMusicVisualizer/sessions/` |
| Your detector files | `.local/share/GranularMusicVisualizer/models/` |

Folders starting with a dot are hidden; **Ctrl+H** in Dolphin shows them.

## Updating and removing

- **Update** to the build this guide names: run the two lines of
  [step 2](#2-install-gmv) again. Your settings, sessions and detector files stay.
- **Remove** GMV:

  ```bash
  bash ~/granular-music-visualizer/gmv-deck.sh remove
  ```

  Type `yes` when asked. It removes the program, its menu entry and its icon. Your settings,
  sessions and detector files stay. A shortcut added to Steam stays in the Steam library
  until you remove it there.

## If something goes wrong

| What you see | What to do |
|---|---|
| The install says *Could not download* | The Deck is offline, or GitHub is slow. Run the same line again. |
| GMV is not in the menu | Run `~/granular-music-visualizer/launch.sh` in Konsole. It starts GMV and prints any problem. Logging out and in refreshes the menu. |
| GMV does not open | Run `~/granular-music-visualizer/launch.sh` in Konsole and read the last lines it prints. |
| Pressing **Start** says GMV *needs the programs pactl and parec*, or *No PulseAudio-compatible sound server answered* | SteamOS is expected to have both, so this would be a surprise. Music files still work: choose **Audio file** under Audio source. Ask Claude (below) to look at it. |
| The picture does not move with the music | Is the source **System audio (PipeWire)**, and did you press **Start**? Is the music playing on the Deck itself, and not muted in the player? Then run the `listen` check from step 5. |
| Radiant stutters or the Deck gets hot | Under **Settings → Visualizer**, set Radiant's **Effects** to *Off*, or use *Classic* or *Spectrum bars*. Plug in the charger for long demos. |
| The sound comes from a TV, dock or Bluetooth speaker | GMV follows whichever output the Deck is using, by itself (*untried on a Deck*). |
| Game Mode problems | See [Game Mode](#6-optional-game-mode-untried-on-a-deck). |

## What has not been tried on a real Deck yet

Everything here was run on the CachyOS PC only. The PC has the same kind of software as the
Deck (an Arch-based system with KDE Plasma, PipeWire and an AMD graphics chip), but it is
not a Deck. The first time on the Deck, these are worth watching:

- that the install finishes, and that GMV opens from the menu in Desktop Mode;
- that **System audio** reacts to music (the `listen` check says so);
- how smooth **Radiant** is on the Deck's much smaller graphics chip;
- Game Mode: whether GMV appears, whether it reacts while the music app runs in the
  background, and whether the controls can be reached;
- a TV, dock or Bluetooth speaker as the output;
- the guitar detector on Linux.

## Using Claude Code on the Deck

If Claude Code is installed on the Deck, start it in Konsole in your home folder and give it
this message:

> Read https://github.com/Ralkero/granular-audio-visualizer/blob/main/docs/STEAM_DECK_FOR_CLAUDE.md
> and use it to install, start and check Granular Music Visualizer on this Steam Deck.

(Before this guide is merged, replace `main` in that link with `docs/steam-deck-demo`.)
