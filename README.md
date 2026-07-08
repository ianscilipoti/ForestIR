# Forest Impulse Response / Sound Simulation

**Paper (arXiv):** 
https://arxiv.org/abs/2607.06299

Simulate forest impulse responses (IRs) and render microphone-array recordings of bird sounds under realistic acoustic, environmental, and noise conditions.

**Dependencies:** Python 3.9+, NumPy, SciPy, soundfile (`pip install -r requirements.txt`).

**Paths:** unless absolute, file arguments are resolved relative to the repository root.

---

## Quick Start

```bash
pip install -r requirements.txt

python code/render_batch.py \
  --mic finland_konnevesi \
  --bird-wav dataset/input/birdnet_birdsound_plus/BirdNET_01_XC169082.wav \
  --noise-model real_forest --noise-source konnevesi --noise-level 0.30
```

Outputs go to `output/` by default: multichannel WAVs, per-microphone WAVs, and `manifest.csv`.

Run `python code/render_batch.py --help` for the full argument list.

---

## Repository Layout

```text
.
├── code/
│   ├── render_batch.py         # CLI — batch render
│   ├── ForestReverb.py         # core IR simulator
│   ├── SignalProcessing.py     # signal processing & noise models
│   ├── Constants.py
│   └── Wav.py
├── dataset/
│   ├── mic_locations/
│   ├── source_locations/
│   ├── background_noise/
│   ├── treeloc_finland.csv     # measured tree positions (--tree-csv finland)
│   └── input/                  # bird WAVs, ESS recordings
├── requirements.txt
└── output/                     
```

---

## CLI Overview

| Group | Key flags | Notes |
|---|---|---|
| Geometry | `--mic`, `--src`, `--mic-csv`, `--src-csv`, `--tree-csv` | Built-in presets for Konnevesi and Konginkangas field layouts; or pass custom CSV paths |
| Source audio | `--bird-wav` | **Required** — clean bird WAV to convolve |
| Environment | `--temp-c`, `--rh-percent`, `--pressure-hpa`, `--ground-type` | Affect speed of sound and air absorption |
| Forest scene | `--tree-csv`, `--apply-branch-leaf`, `--nTrees`, `--forest-range-x/y` | `finland` uses measured trees; `random` / `none` also available |
| Noise | `--noise-model`, `--noise-source`, `--noise-level`, `--noise-path` | `real_forest` + `nriver` or `konnevesi` uses dataset recordings |
| Output | `--output-dir`, `--fs`, `--quiet`, `--print-every` | Default output dir: `output` |

Built-in presets:

| `--mic` | CSV |
|---|---|
| `finland_konnevesi` (default) | `dataset/mic_locations/micloc_konnevesi.csv` |
| `finland_konginkangas_11` | `dataset/mic_locations/mic_konginkangas.csv` |
| `finland_konginkangas_11_1.2m` | same, filtered to z = 1.2 m |

| `--src` | CSV |
|---|---|
| `custom` (default) | `dataset/source_locations/src_custom.csv` |
| `finland_konnevesi` | `dataset/source_locations/srcloc_konnevesi.csv` |
| `finland_konginkangas` | `dataset/source_locations/src_konginkangas.csv` |

---

## Examples

### Konnevesi (Feb 2026)

```bash
python code/render_batch.py \
  --mic finland_konnevesi --src finland_konnevesi \
  --bird-wav dataset/input/birdnet_birdsound_plus/BirdNET_01_XC169082.wav \
  --tree-csv finland \
  --noise-model real_forest --noise-source konnevesi --noise-level 0.4 \
  --temp-c -5 --rh-percent 85 --pressure-hpa 1005 \
  --apply-branch-leaf \
  --output-dir output_konnevesi
```

### Konginkangas (Oct 2025)

```bash
python code/render_batch.py \
  --mic finland_konginkangas_11 --src finland_konginkangas \
  --bird-wav dataset/input/birdnet_birdsound_plus/BirdNET_01_XC169082.wav \
  --noise-model real_forest --noise-source nriver --noise-level 0.4 \
  --temp-c 12.15 --rh-percent 90 --pressure-hpa 1016 \
  --apply-branch-leaf \
  --output-dir output_konginkangas
```

### Custom CSV paths

`--mic-csv` / `--src-csv` / custom `--tree-csv` override the built-in presets (default `--src custom` already uses `src_custom.csv`):

```bash
python code/render_batch.py \
  --mic-csv dataset/mic_locations/mic_custom.csv \
  --tree-csv dataset/treeloc_finland.csv \
  --bird-wav dataset/input/birdnet_birdsound_plus/BirdNET_01_XC169082.wav \
  --noise-model pink --noise-level 0.2 \
  --output-dir output_custom
```

---

## CSV formats

**Microphones** (`--mic-csv` or built-in `--mic` presets): header row with `device`, `x`, `y`, `z` (metres). Row order is channel order; `device` labels per-channel output files (`{tag}_{device}.wav`). 

**Sources** (`--src-csv` or built-in `--src` presets): `x`, `y`, `z` (metres). One row per source position.

---

## Notes

**Noise mixing:** noise is added after the clean render. `--noise-level` is the noise RMS relative to the signal RMS (`0` = off, `0.5` ≈ half, `1.0` ≈ equal).
