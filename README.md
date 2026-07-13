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
| Geometry | `--mic`, `--src`, `--mic-csv`, `--src-csv` | Field layouts below, or custom CSVs |
| Source audio | `--bird-wav` | **Required** |
| Environment | `--temp-c`, `--rh-percent`, `--pressure-hpa`, `--ground-type` | Speed of sound and air absorption |
| Forest scene | `--tree-csv`, `--apply-branch-leaf`, `--nTrees`, `--forest-range-x/y` | Measured, random, or no trees |
| Noise | `--noise-model`, `--noise-source`, `--noise-level`, `--noise-path` | Synthetic or recorded forest noise |
| Output | `--output-dir`, `--fs` | Default dir: `output` |

**Built-in presets** (`--mic-csv` / `--src-csv` override these):

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

## Tunable Variables

Full argument reference. Mic/source preset paths are in **Built-in presets** above; defaults match `code/render_batch.py`.

| Meaning | Argument | Type / choices | Default |
|---|---|---|---|
| Mic array layout | `--mic` | preset name (see above); ignored if `--mic-csv` is set | `finland_konnevesi` |
| Custom mic positions | `--mic-csv` | CSV path (`device`, `x`, `y`, `z`) | `None` |
| Source positions | `--src` | preset name (see above); ignored if `--src-csv` is set | `custom` |
| Custom source positions | `--src-csv` | CSV path (`x`, `y`, `z`) | `None` |
| Clean bird waveform | `--bird-wav` | WAV path | **required** |
| Output directory | `--output-dir` | folder path (created if missing) | `output` |
| Sample rate | `--fs` | Hz | `384000` |
| Air temperature | `--temp-c` | °C | `20` |
| Relative humidity | `--rh-percent` | % RH | `50.0` |
| Atmospheric pressure | `--pressure-hpa` | hPa | `1013.25` |
| Tree geometry | `--tree-csv` | `default`/`finland` (measured), `random`, `none`, or CSV path | `default` (= `finland`) |
| Random tree count | `--nTrees` | int; used with `--tree-csv random` | `0` |
| Random tree sampler | `--tree-sampling-algo` | `uniform` \| `repulsive` | `uniform` |
| Min tree spacing | `--min-tree-spacing` | metres; for `repulsive` | `1.5` |
| Forest x-bounds | `--forest-range-x` | `X_MIN X_MAX` (m) | `0 100` |
| Forest y-bounds | `--forest-range-y` | `Y_MIN Y_MAX` (m) | `-100 0` |
| Tree cylinder height | `--tree-height` | metres | `20.0` |
| Min trunk diameter | `--trunk-diameter-min` | metres | `0.5` |
| Max trunk diameter | `--trunk-diameter-max` | metres | `0.5` |
| Ground reflection preset | `--ground-type` | `default`, `concrete`, `water`, `ice`, `grass`, `snow` | `default` |
| Floor reflection override | `--floor-reflection-coef` | float; `None` → use `--ground-type` | `None` |
| Branch/leaf scattering | `--apply-branch-leaf` | flag | off |
| Branch/leaf scatterer count | `--n-branch-leaf-scatterers` | int | `5000` |
| Branch/leaf z-min | `--branch-leaf-z-min` | metres | `15.0` |
| Branch/leaf z-max | `--branch-leaf-z-max` | metres | `25.0` |
| Noise vs signal RMS | `--noise-level` | float; `0` = off, `1` ≈ equal RMS | `0.30` |
| Noise model | `--noise-model` | `white`, `pink`, `real_forest`, `brown`, `bandlimited` | `real_forest` |
| Real-forest noise set | `--noise-source` | `default`, `nriver`, `konnevesi` | `default` |
| Custom noise path | `--noise-path` | WAV/FLAC file or directory; overrides `--noise-source` | `None` |
| Geometry RNG seed | `--scene-seed` | int | `1234` |
| Noise RNG seed | `--noise-seed` | int; per source: seed + index | `1234` |

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
