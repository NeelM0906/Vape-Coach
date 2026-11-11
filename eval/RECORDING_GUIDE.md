# Recording guide: vape-coach eval clips (10-15 min total)

Goal: a small labeled set that tells us precision, recall, false positives per hour, and count error.
Clips stay on your laptop. Nothing is uploaded.

## Setup (2 min)

- Use the same laptop webcam you'll demo with. Sit at your normal desk distance (head and shoulders
  plus both hands in frame when raised to your mouth). Don't zoom or crop.
- Record at 720p or 640x480, 30 fps. Keep each clip under ~100 MB (1080p HEVC phone video runs ~1 MB/s):
  record at 720p or export a smaller copy if you share clips through Drive. On Windows the **Camera** app works (Video mode). Saves MP4.
- Keep a phone stopwatch visible to you (not the camera) or say "mark" out loud: it makes labeling faster.
- Windows camera privacy: Settings > Privacy & security > Camera > turn on *Camera access*,
  *Let apps access your camera* and *Let desktop apps access your camera*.

## Shot list

Record each block as its own clip. Act naturally: type, read, look at the second monitor between actions.
Leave at least 15 s of normal work between actions.

| Clip | Length | What to do | Count |
|---|---|---|---|
| `vc_01_desk_day_hits` | 3 min | Normal hits with a visible exhale. Vary the hand (left/right) and how long you hold it at your mouth (1-4 s). | 6-8 hits |
| `vc_02_desk_day_stealth` | 2 min | Stealth hits: short inhale, exhale down into your shirt, to the side, or through your nose. A couple with your head turned 30-45°. | 5-6 hits |
| `vc_03_desk_day_lookalikes` | 3 min | No vaping. Coffee/water sips (mug and bottle), hand on chin while reading (20 s+), scratching nose/face, snacking (chips, a bite of a sandwich), a 30 s phone call, yawning with hand over mouth, pen to lips while thinking. | 2-3 of each |
| `vc_04_mixed_dim` | 3 min | Mixed hits + look-alikes in **dim light** (lamp only, or blinds closed) and backlit (window behind you if you can). | 4 hits + 4 look-alikes |
| `vc_05_mixed_far` | 2 min | Mixed hits + sips with the laptop pushed back ~0.5 m, or you leaning back in the chair. | 3 hits + 3 sips |
| `vc_06_jacks` (optional) | 1 min | Step back so head + arms are in frame, do 15 jumping jacks, then 10 more with feet out of frame. Note the true count. | 25 reps |

File names: `vc_<NN>_<place>_<light>_<what>.mp4`, lowercase, no spaces (e.g. `vc_04_desk_dim_mixed.mp4`).
Put them in one folder, e.g. `C:\Users\<you>\vape-clips\`.

## Labeling (5-10 min)

Open the clip in any player that shows seconds (VLC: View > Status bar). Fill one row per action
in `labels.csv` (copy `labels_template.csv`):

```
clip,start_s,end_s,type,notes
vc_01_desk_day_hits.mp4,12.0,16.5,hit,left hand, long hold
```

- `start_s`: hand reaches your mouth. `end_s`: exhale finished (or hand back down if no visible exhale).
  Half-second precision is plenty; scoring allows +/- 2 s.
- `type`: `hit`, `stealth_hit` (positives); `sip`, `chin`, `snack`, `phone`, `scratch`, `other` (look-alikes);
  `unsure` for anything you can't call (scored as don't-care).
- Label **every** hit. Labeling look-alikes is optional but lets the report say *why* a false positive happened.
- For `vc_06_jacks`, write the true rep count in `notes` on one `other` row.

## Run the eval

```
hermes vape setup
for each clip:   hermes vape watch --replay vc_01_desk_day_hits.mp4 --out vc_01_desk_day_hits.json
hermes vape eval labels.csv vc_*.json
hermes vape reps --replay vc_06_jacks.mp4 --exercise jumping_jacks --debug
```

PowerShell loop: `Get-ChildItem vc_*.mp4 | % { hermes vape watch --replay $_.FullName --out ($_.BaseName + ".json") }`

Add `--debug` to watch the overlay (state machine, hand-mouth distance, exhale score), or
`--debug-out overlay.mp4` to save it for the demo video. `--always-active` runs the full model set on
every frame, which gives the best possible recall to compare against the 2 fps idle mode.
