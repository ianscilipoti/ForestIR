__author__ = "Colin/Jennifer"

import argparse
import contextlib
import csv
import hashlib
import os

import numpy as np

import ForestReverb as FR


BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET = os.path.join(BASE, "dataset")

MIC_CSV_KONGINKANGAS = os.path.join(DATASET, "mic_locations", "mic_konginkangas.csv")
MIC_CSV_KONNEVESI = os.path.join(DATASET, "mic_locations", "micloc_konnevesi.csv")

SRC_CSV_CUSTOM = os.path.join(DATASET, "source_locations", "src_custom.csv")
SRC_CSV_KONNEVESI = os.path.join(DATASET, "source_locations", "srcloc_konnevesi.csv")
SRC_CSV_KONGINKANGAS = os.path.join(DATASET, "source_locations", "src_konginkangas.csv")

TREE_CSV_FINLAND = os.path.join(DATASET, "treeloc_finland.csv")

NOISE_SOURCES = {
    "nriver":    os.path.join(DATASET, "background_noise", "nriver"),
    "konnevesi": os.path.join(DATASET, "background_noise", "konnevesi"),
}

NOISE_MODEL_BY_CANON = {
    "white": 1,
    "pink": 2,
    "real_forest": 3,
    "brown": 4,
    "bandlimited": 5,
}


def noise_model_arg(s: str) -> str:
    """Normalize user input to canonical noise-model name for choices / manifest."""
    raw = s.strip().lower()
    if raw.replace("_", "").replace("-", "").replace(" ", "") == "realforest":
        return "real_forest"
    t = raw.replace("-", " ").replace("_", " ")
    if t in ("band limited",):
        t = "bandlimited"
    if t not in NOISE_MODEL_BY_CANON:
        allowed = ", ".join(sorted(NOISE_MODEL_BY_CANON))
        raise argparse.ArgumentTypeError(f"invalid noise model {s!r}; choose one of: {allowed}")
    return t


def load_positions_csv(path, delimiter=","):
    """
    Returns float64 array of shape [N,3] from a CSV that has at least 3 numeric columns.
    Tries header names x,y,z (case-insensitive), otherwise uses first 3 columns.
    Handles single-row CSVs by reshaping to [1,3].
    """
    try:
        data = np.genfromtxt(path, delimiter=delimiter, names=True, dtype=None, encoding=None)
        data = np.atleast_1d(data)
        if data.dtype.names:
            names = [n.lower() for n in data.dtype.names]
            if all(k in names for k in ("x", "y", "z")):
                x = np.asarray(data[data.dtype.names[names.index("x")]], dtype=float)
                y = np.asarray(data[data.dtype.names[names.index("y")]], dtype=float)
                z = np.asarray(data[data.dtype.names[names.index("z")]], dtype=float)
                arr = np.stack([x, y, z], axis=-1)
                if arr.ndim == 1:
                    arr = arr.reshape(1, -1)
                return arr.astype(np.float64)
    except Exception:
        pass

    arr = np.loadtxt(
        path,
        delimiter=delimiter,
        usecols=(0, 1, 2),
        ndmin=2,
        dtype=float,
        comments=None,
        skiprows=1,
    )
    return arr.astype(np.float64)


def load_source_extras_csv(path, n_rows, delimiter=","):
    """
    Read optional per-source columns used by --mix-sources (case-insensitive headers):
      wav     : dry source WAV (absolute, or relative to the repo root); blank -> --bird-wav
      gain    : linear gain before mixing; blank -> 1.0
      onset_s : start time in the mix, seconds; blank -> 0.0
    Returns a list of n_rows dicts with keys "wav" (None if unset), "gain", "onset_s".
    Rows are matched to load_positions_csv() output by order (blank lines skipped).
    """
    extras = [{"wav": None, "gain": 1.0, "onset_s": 0.0} for _ in range(n_rows)]
    with open(path, "r", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        if not reader.fieldnames:
            return extras
        lower = {name.strip().lower(): name for name in reader.fieldnames if name}
        if not any(k in lower for k in ("wav", "gain", "onset_s")):
            return extras
        rows = [r for r in reader if any(str(v or "").strip() for v in r.values())]

    if len(rows) != n_rows:
        raise ValueError(
            f"Source CSV {path!r}: found {len(rows)} data rows but {n_rows} positions."
        )
    for i, row in enumerate(rows):
        def cell(key):
            col = lower.get(key)
            return "" if col is None else str(row.get(col) or "").strip()

        if cell("wav"):
            extras[i]["wav"] = resolve_cli_path(cell("wav"))
        if cell("gain"):
            extras[i]["gain"] = float(cell("gain"))
        if cell("onset_s"):
            extras[i]["onset_s"] = float(cell("onset_s"))
    return extras


def rename_per_mic_files(out_per, tag, mic_names):
    for ch, mic_label in enumerate(mic_names or []):
        old_path = os.path.join(out_per, f"{tag}_mic{ch:02d}.wav")
        new_path = os.path.join(out_per, f"{tag}_{mic_label}.wav")
        if os.path.exists(old_path):
            os.replace(old_path, new_path)


def load_mic_positions_csv(path, delimiter=","):
    """Return (positions [N,3], device_names [N]) in CSV row order."""
    data = np.genfromtxt(path, delimiter=delimiter, names=True, dtype=None, encoding=None)
    if data.dtype.names is None:
        raise ValueError(f"CSV {path!r} requires a header row.")

    data = np.atleast_1d(data)
    dtype_names = data.dtype.names
    names_lower = [n.lower() for n in dtype_names]
    for col in ("device", "x", "y", "z"):
        if col not in names_lower:
            raise ValueError(f"Missing '{col}' column in {path!r}. Columns: {dtype_names}")

    device_col = dtype_names[names_lower.index("device")]
    x_col = dtype_names[names_lower.index("x")]
    y_col = dtype_names[names_lower.index("y")]
    z_col = dtype_names[names_lower.index("z")]

    devices = []
    positions = []
    seen = set()
    for row in data:
        device = str(row[device_col]).strip()
        if not device:
            continue
        if device in seen:
            raise ValueError(f"Duplicate device {device!r} in {path!r}")
        seen.add(device)
        devices.append(device)
        positions.append([float(row[x_col]), float(row[y_col]), float(row[z_col])])

    if not devices:
        raise ValueError(f"No microphones found in {path!r}")

    return np.array(positions, dtype=np.float64), devices


def resolve_cli_path(user_path):
    """Absolute path as-is; otherwise resolved relative to the repository root."""
    user_path = os.path.expanduser(user_path)
    if os.path.isabs(user_path):
        return os.path.abspath(user_path)
    return os.path.abspath(os.path.join(BASE, user_path))


def resolve_builtin_mic_csv(mic_choice):
    if mic_choice in ("finland_konginkangas_11", "finland_konginkangas_11_1.2m"):
        return MIC_CSV_KONGINKANGAS
    if mic_choice == "finland_konnevesi":
        return MIC_CSV_KONNEVESI
    raise ValueError(f"Unknown mic choice: {mic_choice}")


def resolve_builtin_src_csv(src_choice):
    if src_choice == "custom":
        return SRC_CSV_CUSTOM
    if src_choice == "finland_konnevesi":
        return SRC_CSV_KONNEVESI
    if src_choice == "finland_konginkangas":
        return SRC_CSV_KONGINKANGAS
    raise ValueError(f"Unknown src choice: {src_choice}")


def normalize_tree_mode(tree_csv_arg):
    tree_csv_lower = tree_csv_arg.strip().lower()
    if tree_csv_lower == "default":
        tree_csv_lower = "finland"
    if tree_csv_lower == "finland":
        return "finland", TREE_CSV_FINLAND
    if tree_csv_lower == "random":
        return "random", None
    if tree_csv_lower == "none":
        return "none", None
    return "custom", resolve_cli_path(tree_csv_arg)


def csv_row_count(path):
    rows = []
    with open(path, "r", newline="") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if any(str(cell).strip() for cell in row):
                rows.append(row)

    if not rows:
        return 0

    try:
        [float(cell) for cell in rows[0] if str(cell).strip()]
        has_header = False
    except ValueError:
        has_header = True

    return len(rows) - (1 if has_header else 0)


def make_src_uid(src_csv_path, src_idx, pos_src):
    token = "|".join(
        [
            os.path.abspath(src_csv_path),
            str(int(src_idx)),
            f"{float(pos_src[0]):.6f}",
            f"{float(pos_src[1]):.6f}",
            f"{float(pos_src[2]):.6f}",
        ]
    )
    digest = hashlib.sha1(token.encode("utf-8")).hexdigest()[:12]
    return f"src_{digest}"


def build_parser():
    p = argparse.ArgumentParser(
        description="Run forest IR rendering for all source locations, saving outputs to 'output/'"
    )
    p.add_argument(
        "--mic",
        choices=[
            "finland_konginkangas_11",
            "finland_konginkangas_11_1.2m",
            "finland_konnevesi",
        ],
        default="finland_konnevesi",
        help=(
            "Built-in microphone layout. "
            "finland_konginkangas_11 / _1.2m: Konginkangas (mic_konginkangas.csv). "
            "finland_konnevesi: Konnevesi BPH (micloc_konnevesi.csv). "
            "Ignored if --mic-csv is set."
        ),
    )
    p.add_argument(
        "--mic-csv",
        type=str,
        default=None,
        help=(
            "Custom microphone locations CSV path (absolute, or relative to the repo root). "
            "When provided, overrides --mic. Default: None."
        ),
    )
    p.add_argument(
        "--src-csv",
        type=str,
        default=None,
        help=(
            "Custom source locations CSV path (absolute, or relative to the repo root). "
            "Overrides --src. Default: None."
        ),
    )
    p.add_argument(
        "--src",
        choices=["custom", "finland_konnevesi", "finland_konginkangas"],
        default="custom",
        help=(
            "Built-in source layout. "
            "custom: src_custom.csv; "
            "finland_konnevesi: srcloc_konnevesi.csv; "
            "finland_konginkangas: src_konginkangas.csv. "
            "Ignored if --src-csv is set."
        ),
    )
    p.add_argument(
        "--bird-wav",
        default=None,
        help=(
            "Path to clean bird WAV to convolve (mono or stereo ok). Required, unless "
            "--mix-sources is set and every source row has its own 'wav' column."
        ),
    )
    p.add_argument(
        "--mix-sources",
        action="store_true",
        default=False,
        help=(
            "Render every row of the source CSV into ONE scene (same forest) and mix them "
            "into a single multichannel WAV, with noise added once to the mix. Optional "
            "source CSV columns: wav, gain, onset_s. Default: off (one render per source)."
        ),
    )
    p.add_argument(
        "--output-dir",
        default="output",
        help="Directory to write all outputs. Default: output.",
    )
    p.add_argument(
        "--print-every",
        type=int,
        default=10,
        help="Print batch progress every N sources (last source always). Use 1 for every source. Default: 10.",
    )
    p.add_argument(
        "--quiet",
        action="store_true",
        default=False,
        help="Suppress per-source progress lines (mic/source load lines still print). Default: off.",
    )

    p.add_argument("--fs", type=int, default=384000, help="IR/sample rate. Default: 384000.")
    p.add_argument("--temp-c", type=float, default=20, help="Temperature in Celsius. Default: 20.")
    p.add_argument(
        "--rh-percent",
        type=float,
        default=50.0,
        help="Relative humidity in percent. Default: 50.0.",
    )
    p.add_argument(
        "--pressure-hpa",
        type=float,
        default=1013.25,
        help="Atmospheric pressure in hPa. Default: 1013.25.",
    )
    p.add_argument(
        "--nTrees",
        type=int,
        default=0,
        help="Number of scattering trees (only with --tree-csv random). Default: 0.",
    )
    p.add_argument(
        "--tree-sampling-algo",
        choices=["uniform", "repulsive"],
        default="uniform",
        help="Sampling algorithm for random trees. Default: uniform.",
    )
    p.add_argument(
        "--min-tree-spacing",
        type=float,
        default=1.5,
        help="Minimum spacing (m) for repulsive random-tree sampling. Default: 1.5.",
    )
    p.add_argument(
        "--tree-csv",
        default="default",
        help=(
            "Trees: 'default' or 'finland' -> dataset/treeloc_finland.csv; "
            "'random' / 'none' / or a custom CSV path (absolute, or relative to the repo root). "
            "Default: default (same as finland)."
        ),
    )
    p.add_argument(
        "--forest-range-x",
        type=float,
        nargs=2,
        default=[0.0, 100.0],
        metavar=("X_MIN", "X_MAX"),
        help=(
            "x bounds (m) for random trees and for branch_leaf scatterer horizontal extent when none is set "
            "from tree CSV. Default: 0 100."
        ),
    )
    p.add_argument(
        "--forest-range-y",
        type=float,
        nargs=2,
        default=[-100.0, 0.0],
        metavar=("Y_MIN", "Y_MAX"),
        help=(
            "y bounds (m) for random trees and for branch_leaf scatterer horizontal extent when none is set "
            "from tree CSV. Default: -100 0."
        ),
    )

    p.add_argument(
        "--tree-height",
        type=float,
        default=20.0,
        help="Height of finite tree cylinders in metres. Default: 20.0.",
    )
    p.add_argument(
        "--trunk-diameter-min",
        type=float,
        default=0.5,
        help="Minimum tree trunk diameter (m). Default: 0.5.",
    )
    p.add_argument(
        "--trunk-diameter-max",
        type=float,
        default=0.5,
        help="Maximum tree trunk diameter (m). Default: 0.5.",
    )
    p.add_argument(
        "--ground-type",
        type=str,
        choices=list(FR.GROUND_REFLECTION_PRESETS.keys()),
        default="default",
        help="Ground reflection preset for the floor-reflection path. Default: default.",
    )
    p.add_argument(
        "--floor-reflection-coef",
        type=float,
        default=None,
        help="Optional manual override for the ground reflection coefficient. Default: None (use preset).",
    )
    p.add_argument(
        "--apply-branch-leaf",
        action="store_true",
        default=False,
        help="Enable branches/leaves scattering simulation. Default: False (off).",
    )
    p.add_argument(
        "--n-branch-leaf-scatterers",
        type=int,
        default=5000,
        help="Number of diffuse branch/leaf scatterers. Default: 5000.",
    )
    p.add_argument(
        "--branch-leaf-z-min",
        type=float,
        default=15.0,
        help="Min height of branch/leaf scatterer volume (m). Default: 15.0.",
    )
    p.add_argument(
        "--branch-leaf-z-max",
        type=float,
        default=25.0,
        help="Max height of branch/leaf scatterer volume (m). Default: 25.0.",
    )

    p.add_argument(
        "--noise-level",
        type=float,
        default=0.30,
        help="Noise gain vs signal RMS (see README). Default: 0.30.",
    )
    p.add_argument(
        "--noise-model",
        type=noise_model_arg,
        choices=list(NOISE_MODEL_BY_CANON.keys()),
        default="real_forest",
        metavar="NAME",
        help=(
            "Noise: white, pink, real_forest, brown, bandlimited. "
            "real_forest uses recorded background when --noise-level > 0 "
            "(needs --noise-path or --noise-source nriver|konnevesi). Default: real_forest."
        ),
    )
    p.add_argument(
        "--noise-source",
        choices=["default", "nriver", "konnevesi"],
        default="default",
        help=(
            "When --noise-model is real_forest: 'nriver' or 'konnevesi' sets dataset/background_noise/... "
            "unless --noise-path is set. 'default' means do not auto-set path. Default: default."
        ),
    )
    p.add_argument(
        "--noise-path",
        default=None,
        help="Explicit path to noise WAV or directory for real_forest (overrides --noise-source). Default: None.",
    )
    p.add_argument(
        "--scene-seed",
        type=int,
        default=1234,
        help="Base seed for scene geometry randomness (random trees/branch_leaf). Default: 1234.",
    )
    p.add_argument(
        "--noise-seed",
        type=int,
        default=1234,
        help="Base seed for noise; per-source seed is this + src_idx. Default: 1234.",
    )
    return p


def main():
    args = build_parser().parse_args()

    if not args.bird_wav and not args.mix_sources:
        raise ValueError("--bird-wav is required.")

    fr_x = list(args.forest_range_x)
    fr_y = list(args.forest_range_y)
    if fr_x[0] >= fr_x[1]:
        raise ValueError(f"--forest-range-x requires min < max, got {fr_x[0]} {fr_x[1]}")
    if fr_y[0] >= fr_y[1]:
        raise ValueError(f"--forest-range-y requires min < max, got {fr_y[0]} {fr_y[1]}")
    if args.trunk_diameter_min <= 0 or args.trunk_diameter_max <= 0:
        raise ValueError("--trunk-diameter-min and --trunk-diameter-max must be > 0.")
    if args.trunk_diameter_min > args.trunk_diameter_max:
        raise ValueError(
            f"--trunk-diameter-min must be <= --trunk-diameter-max, got "
            f"{args.trunk_diameter_min} {args.trunk_diameter_max}"
        )
    if args.min_tree_spacing <= 0:
        raise ValueError("--min-tree-spacing must be > 0.")

    if args.mic_csv is not None:
        mic_choice = "custom"
        mic_csv_path = resolve_cli_path(args.mic_csv)
    else:
        mic_choice = args.mic
        mic_csv_path = resolve_builtin_mic_csv(mic_choice)

    if args.src_csv is not None:
        src_choice = "custom"
        src_csv_path = resolve_cli_path(args.src_csv)
    else:
        src_choice = args.src
        src_csv_path = resolve_builtin_src_csv(src_choice)

    bird_wav_path = resolve_cli_path(args.bird_wav) if args.bird_wav else None
    if args.noise_path:
        noise_path = resolve_cli_path(args.noise_path)
    elif args.noise_source and args.noise_source != "default":
        noise_path = NOISE_SOURCES[args.noise_source]
    else:
        noise_path = None
    tree_mode, tree_csv_path = normalize_tree_mode(args.tree_csv)

    model_id = NOISE_MODEL_BY_CANON[args.noise_model]

    mic_poss, mic_names = load_mic_positions_csv(mic_csv_path)
    if mic_choice == "finland_konginkangas_11_1.2m":
        keep = np.isclose(mic_poss[:, 2], 1.2)
        mic_poss = mic_poss[keep]
        mic_names = [name for name, k in zip(mic_names, keep) if k]
        if mic_poss.shape[0] == 0:
            raise ValueError("No microphones with z == 1.2 m in mic CSV.")

    src_poss = load_positions_csv(src_csv_path)

    if mic_poss.ndim != 2 or mic_poss.shape[1] != 3:
        raise ValueError(f"Mic CSV {mic_csv_path} did not parse as [N,3]. Got {mic_poss.shape}")
    if src_poss.ndim != 2 or src_poss.shape[1] != 3:
        raise ValueError(f"Source CSV {src_csv_path} did not parse as [N,3]. Got {src_poss.shape}")

    src_extras = None
    if args.mix_sources:
        src_extras = load_source_extras_csv(src_csv_path, src_poss.shape[0])
        for i, ex in enumerate(src_extras):
            if ex["wav"] is None:
                if bird_wav_path is None:
                    raise ValueError(
                        f"Source row {i} has no 'wav' value and --bird-wav was not given."
                    )
                ex["wav"] = bird_wav_path
            if not os.path.exists(ex["wav"]):
                raise FileNotFoundError(f"Source row {i}: WAV not found: {ex['wav']}")

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Loaded {mic_poss.shape[0]} microphones from {mic_csv_path}")
    print(f"Loaded {src_poss.shape[0]} source positions from {src_csv_path}")
    print(f"Writing results to: {os.path.abspath(args.output_dir)}")

    forest_kwargs = {
        "rh_percent": args.rh_percent,
        "pressure_hpa": args.pressure_hpa,
        "treeHeight": args.tree_height,
        "trunkDiameterRange_m": (args.trunk_diameter_min, args.trunk_diameter_max),
        "groundType": args.ground_type,
        "floorReflectionCoef": args.floor_reflection_coef,
        "applyBranchLeafScattering": args.apply_branch_leaf,
        "nBranchLeafScatterers": args.n_branch_leaf_scatterers,
        "branch_leaf_bounds_z": (args.branch_leaf_z_min, args.branch_leaf_z_max),
    }

    if tree_mode == "finland":
        forest_kwargs["tree_csv_path"] = tree_csv_path
        forest_kwargs["tree_csv_kwargs"] = {
            "x_col": "x",
            "y_col": "y",
            "has_header": True,
            "z_val": 0.0,
        }
    elif tree_mode == "random":
        if args.nTrees > 0 or args.apply_branch_leaf:
            forest_kwargs["forestRange_x"] = fr_x
            forest_kwargs["forestRange_y"] = fr_y
            forest_kwargs["samplingAlgo"] = args.tree_sampling_algo
            forest_kwargs["minTreeSpacing_m"] = args.min_tree_spacing
    elif tree_mode == "custom":
        forest_kwargs["tree_csv_path"] = tree_csv_path

    if args.apply_branch_leaf and ("forestRange_x" not in forest_kwargs or "forestRange_y" not in forest_kwargs):
        forest_kwargs["forestRange_x"] = fr_x
        forest_kwargs["forestRange_y"] = fr_y

    noise_kwargs = {}
    if model_id == 3 and args.noise_level > 0.0:
        if not noise_path:
            raise ValueError(
                "real_forest noise with --noise-level > 0 requires --noise-path or "
                "--noise-source nriver|konnevesi (not --noise-source default)."
            )
        noise_kwargs["noise_dir"] = noise_path

    if tree_mode == "random":
        render_n_trees = args.nTrees
        manifest_n_trees = args.nTrees
    elif tree_mode == "none":
        render_n_trees = 0
        manifest_n_trees = 0
    else:
        render_n_trees = 0
        manifest_n_trees = csv_row_count(tree_csv_path) if tree_csv_path else 0

    batch = os.path.basename(os.path.abspath(args.output_dir))
    manifest_path = os.path.join(args.output_dir, "manifest.csv")
    manifest_fieldnames = [
        "batch",
        "src_uid",
        "tag",
        "mc_wav",
        "per_mic_dir",
        "src_idx",
        "src_x",
        "src_y",
        "src_z",
        "mic_cfg",
        "mic_csv",
        "src_cfg",
        "src_csv",
        "noise_model",
        "noise_level",
        "noise_path",
        "temp_c",
        "rh_percent",
        "pressure_hpa",
        "tree_csv",
        "n_trees",
        "tree_height",
        "tree_sampling_algo",
        "min_tree_spacing_m",
        "trunk_diameter_min_m",
        "trunk_diameter_max_m",
        "apply_branch_leaf",
        "n_branch_leaf_scatterers",
        "branch_leaf_z_min",
        "branch_leaf_z_max",
        "ground_type",
        "floor_reflection_coef",
        "fs",
        "bird_wav",
        "tree_mode",
        "forest_range_x",
        "forest_range_y",
        "scene_seed",
        "noise_seed",
        "mix_sources",
        "n_sources_in_mix",
        "src_gain",
        "src_onset_s",
    ]

    def manifest_row(tag, out_mc, out_per, idx, pos_src, src_wav, noise_seed, gain=1.0, onset_s=0.0, n_mix=1):
        return {
            "batch": batch,
            "src_uid": make_src_uid(src_csv_path, idx, pos_src),
            "tag": tag,
            "mc_wav": out_mc,
            "per_mic_dir": out_per,
            "src_idx": idx,
            "src_x": float(pos_src[0]),
            "src_y": float(pos_src[1]),
            "src_z": float(pos_src[2]),
            "mic_cfg": mic_choice,
            "mic_csv": mic_csv_path,
            "src_cfg": src_choice,
            "src_csv": src_csv_path,
            "noise_model": args.noise_model,
            "noise_level": args.noise_level,
            "noise_path": noise_path,
            "temp_c": args.temp_c,
            "rh_percent": args.rh_percent,
            "pressure_hpa": args.pressure_hpa,
            "tree_csv": tree_csv_path,
            "n_trees": manifest_n_trees,
            "tree_height": args.tree_height,
            "tree_sampling_algo": args.tree_sampling_algo if tree_mode == "random" else "n/a",
            "min_tree_spacing_m": args.min_tree_spacing if tree_mode == "random" else "",
            "trunk_diameter_min_m": args.trunk_diameter_min,
            "trunk_diameter_max_m": args.trunk_diameter_max,
            "apply_branch_leaf": args.apply_branch_leaf,
            "n_branch_leaf_scatterers": args.n_branch_leaf_scatterers,
            "branch_leaf_z_min": args.branch_leaf_z_min,
            "branch_leaf_z_max": args.branch_leaf_z_max,
            "ground_type": args.ground_type,
            "floor_reflection_coef": args.floor_reflection_coef,
            "fs": args.fs,
            "bird_wav": src_wav,
            "tree_mode": tree_mode,
            "forest_range_x": f"{fr_x[0]},{fr_x[1]}",
            "forest_range_y": f"{fr_y[0]},{fr_y[1]}",
            "scene_seed": args.scene_seed,
            "noise_seed": noise_seed,
            "mix_sources": bool(args.mix_sources),
            "n_sources_in_mix": n_mix,
            "src_gain": gain,
            "src_onset_s": onset_s,
        }

    common_kwargs = dict(
        micPoss=mic_poss.astype(np.float64),
        fs_ir=args.fs,
        temp_c=args.temp_c,
        nTrees=render_n_trees,
        forest_kwargs=forest_kwargs,
        model_id=model_id,
        snr_db=None,
        noise_level=args.noise_level,
        noise_kwargs=noise_kwargs if noise_kwargs else None,
        scene_seed=args.scene_seed,
    )

    with open(manifest_path, "w", newline="") as manifest_f:
        manifest = csv.DictWriter(manifest_f, fieldnames=manifest_fieldnames)
        manifest.writeheader()

        if args.mix_sources:
            # One scene: every source row rendered into the same forest, mixed, noise added once.
            # One manifest row per source, all sharing the same tag / mc_wav.
            n_src = len(src_poss)
            tag = f"{mic_choice}_{src_choice}_nl{args.noise_level:.2f}_mix{n_src:03d}"
            out_mc = os.path.join(args.output_dir, f"{tag}.wav")
            out_per = os.path.join(args.output_dir, f"per_mic_{tag}")
            os.makedirs(out_per, exist_ok=True)
            noise_seed = args.noise_seed

            sources = [
                {"wav": ex["wav"], "pos": pos, "gain": ex["gain"], "onset_s": ex["onset_s"]}
                for pos, ex in zip(src_poss, src_extras)
            ]
            if not args.quiet:
                print(f"Mixing {n_src} sources into one scene:")
                for i, s in enumerate(sources):
                    print(
                        f"  [{i}] pos={s['pos']} gain={s['gain']} onset_s={s['onset_s']} "
                        f"wav={os.path.basename(s['wav'])}"
                    )

            with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
                _ = FR.synthesize_multi_source_observation(
                    sources=sources,
                    noise_seed=noise_seed,
                    out_multichannel_path=out_mc,
                    out_per_channel_dir=out_per,
                    out_basename=tag,
                    **common_kwargs,
                )
            rename_per_mic_files(out_per, tag, mic_names)

            for idx, s in enumerate(sources):
                manifest.writerow(
                    manifest_row(
                        tag, out_mc, out_per, idx, s["pos"], s["wav"], noise_seed,
                        gain=s["gain"], onset_s=s["onset_s"], n_mix=n_src,
                    )
                )
        else:
            for idx, pos_src in enumerate(src_poss):
                tag = f"{mic_choice}_{src_choice}_nl{args.noise_level:.2f}_src{idx:03d}"
                out_mc = os.path.join(args.output_dir, f"{tag}.wav")
                out_per = os.path.join(args.output_dir, f"per_mic_{tag}")
                os.makedirs(out_per, exist_ok=True)

                pe = max(1, int(args.print_every))
                if not args.quiet and (idx % pe == 0 or idx == len(src_poss) - 1):
                    print(f"[{idx+1}/{len(src_poss)}] Rendering source at {pos_src}")

                noise_seed = None if args.noise_seed is None else (args.noise_seed + idx)

                with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
                    _ = FR.synthesize_array_observation(
                        src_wav_path=bird_wav_path,
                        posSrc=pos_src.astype(np.float64),
                        noise_seed=noise_seed,
                        out_multichannel_path=out_mc,
                        out_per_channel_dir=out_per,
                        out_basename=tag,
                        **common_kwargs,
                    )
                rename_per_mic_files(out_per, tag, mic_names)

                manifest.writerow(
                    manifest_row(tag, out_mc, out_per, idx, pos_src, bird_wav_path, noise_seed)
                )

    print(f"Wrote manifest: {manifest_path}")
    print("Done.")


if __name__ == "__main__":
    main()
