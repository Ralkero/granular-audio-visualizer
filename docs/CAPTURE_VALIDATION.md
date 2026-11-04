# Guided capture validation matrix

Every check must be labeled **observed**, **simulated**, **failed**, or **unrun**. Never change playback/device settings automatically. The user performs manual actions; analysis captures only metrics with raw recording off.

Start `scripts/validate_capture.py --pid PID --pair-system --seconds 600 --output local-evidence/probe.json` after discovering the intended app. It records separate process-tree/system feature metrics and parent/helper resources. `--advanced` enables the current delayed HPSS worker. Use `--device "EXACT LOOPBACK NAME"` for an explicit output. Inspect `.latest.json` during capture; `.samples.jsonl` records one-second observations. `--mark "user confirmed pause" --output ...` appends an observation marker. Markers are user reports, not precisely timed automatic playback actions. The harness never switches a selected app to system capture on failure.

| Check | Procedure / acceptance | Phase 1.1 execution status |
|---|---|---|
| Spotify already playing | Launch Spotify and play normally; start system and process capture; confirm activity | Observed: both produced nonzero frames |
| Spotify pause/resume with browser audible | User pauses Spotify ~20 s, leaves browser playing, then resumes. Process signal should go silent while system remains active | Observed user-assisted: zero Spotify peak during pause, nonzero system peak; Spotify signal resumed |
| Start while paused | User pauses; start a NEW process capture; wait, then resume | Unrun |
| Seek and track transition | User seeks then skips; ensure capture continues without process/format changes or invented gaps | Awaiting user-assisted observation |
| Quit/restart | User chooses Spotify Exit/Quit (window close may only minimize), relaunches and resumes; source must reconnect only to an unambiguous same-executable root | Awaiting user-assisted observation |
| Multiple roots / PID reuse | Inject ambiguous roots and creation-time mismatch; never select arbitrarily | Planned fault injection |
| Repeated lifecycle | Start/stop several short sessions and check owned threads/children after shutdown | Planned automated integration |
| Default output switch | User switches default output while playing; capture should reopen with a new segment | Physical test unrun; simulated change tested in Phase One |
| Explicit device disconnection | User disconnects selected USB output; source must wait for that device, not broaden to another | Physical test unrun |
| USB reconnect | User reconnects; verify timeline gap, segment reset, no synthetic transient | Physical test unrun |
| 10-minute soak off/on | Repeat equivalent source/device/logging/UI settings, note content/load differences | Baseline paired run in progress |
| Failures | Inject logger writes, helper exit, slow subscriber and advanced overload; preserve boundedness | Planned bounded tests |

Capture isolation is supported by the actual Spotify pause/resume contrast, not by an inactive unrelated process yielding silence. This does not establish zero leakage under every Windows routing configuration. No calibrated physical sound/display latency is measured.

Local import manifest example (store it under ignored local-evidence):
```json
{"audio":"licensed.wav","sha256":"ACTUAL_SHA256","license":"ACTUAL_PERMISSION_OR_LICENSE",
 "provenance":"origin and permission record","permission_verified":true,
 "onsets":[0.5,1.0],"annotation_method":"manual attack annotations",
 "uncertainty_seconds":0.02,"source_recording_id":"unique-recording-id","split":"development"}
```
Run `python -m gmv.evaluation.run --import-manifest local-evidence/import.json --output local-evidence/import-scores.json`. All attacks in this minimal format are full-mix onsets; instrument reference labels require separately verified annotations. Do not redistribute local recordings or metadata just because the manifest validates.
