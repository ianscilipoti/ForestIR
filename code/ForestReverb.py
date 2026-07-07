# Based partly on Kaneko & Gamper (2022), JASA Express Letters, DOI: 10.1121/10.0014809

__author__ = "Colin/Jennifer"

import os, time
import numpy as np
from scipy.fft import rfft, irfft
from scipy.special import jn, yn
import Constants as C
import Wav
import SignalProcessing as sigp
from SignalProcessing import (
    addSignal, getImpulse, getFreqsForGivenFilterLength, getIRFromSpectrum_irfft,
    dists, getDelayedSig, floatX
)

na, join, norm, pi = np.newaxis, os.path.join, np.linalg.norm, C.pi

GROUND_REFLECTION_PRESETS = {
    "default": 0.8,
    "concrete": 0.99,
    "water": 0.99,
    "ice": 0.99,
    "grass": 0.80,
    "snow": 0.70,
}

def resolve_floor_reflection_coef(groundType="default", floorReflectionCoef=None):
    ground_key = "default" if groundType is None else str(groundType).strip().lower()
    if ground_key not in GROUND_REFLECTION_PRESETS:
        valid = ", ".join(GROUND_REFLECTION_PRESETS.keys())
        raise ValueError(f"Unknown ground type {groundType!r}. Expected one of: {valid}")
    if floorReflectionCoef is not None:
        return float(floorReflectionCoef)
    return float(GROUND_REFLECTION_PRESETS[ground_key])

def computeAirAbsorptionCoef(freqs, temperature_in_celsius=20.0, pressure_in_kiloPascal=101.325, relativeHumidity_in_percent=50.0):
    T = 273.15 + temperature_in_celsius
    temp_rel = T / 293.15
    T_01 = 273.15 + 0.01
    P_rel = pressure_in_kiloPascal / 101.325
    P_sat_over_P_ref = 10 ** ((-6.8346 * (T_01 / T) ** 1.261) + 4.6151)
    H = relativeHumidity_in_percent * (P_sat_over_P_ref / P_rel)
    Fro = P_rel * (24 + 40400 * H * (0.02 + H) / (0.391 + H))
    Frn = P_rel / np.sqrt(temp_rel) * (9 + 280 * H * np.exp(-4.17 * ((temp_rel ** (-1 / 3)) - 1)))
    Xc = 1.84e-11 / P_rel * np.sqrt(temp_rel)
    f2 = freqs ** 2
    Xo = 0.01275 * np.exp(-2239.1 / T) * 1 / (Fro + f2 / Fro)
    Xn = 0.1068  * np.exp(-3352   / T) * 1 / (Frn + f2 / Frn)
    return 20 * np.log10(np.e) * f2 * (Xc + temp_rel ** (-2.5) * (Xo + Xn))

def getFilterBankForAirAbsorption(
    ntaps, fs, dists_, fftLen,
    temperature_in_celsius=20.0,           
    relativeHumidity_in_percent=50.0,      
    pressure_in_kiloPascal=101.325         
):
    freqs, fh = sigp.getFreqsForGivenFilterLength(ntaps, fs)
    att_dB = computeAirAbsorptionCoef(
        fh,
        temperature_in_celsius=temperature_in_celsius,           
        relativeHumidity_in_percent=relativeHumidity_in_percent, 
        pressure_in_kiloPascal=pressure_in_kiloPascal            
    )[:, na] * dists_[na, :]
    fil = irfft(sigp.getAmplitudeFromdB(-att_dB), axis=0)
    fil = np.roll(fil, ntaps // 2, axis=0)
    return fil, rfft(fil, n=fftLen, axis=0)


def computeGammaTable(N, a, freqs):
    F = len(freqs)
    ks = 2 * pi * freqs / C.soundVel
    kas = ks * a
    ns = np.arange(N + 2)
    jns = np.array([jn(ns, ka) for ka in kas])
    yns = np.array([yn(ns, ka) for ka in kas])
    tanG = np.zeros([F, N + 1], dtype=floatX)
    with np.errstate(divide="ignore", invalid="ignore"):
        for n in range(1, N + 1):
            tanG[:, n] = (jns[:, n - 1] - jns[:, n + 1]) / (yns[:, n - 1] - yns[:, n + 1])
        tanG[:, 0] = jns[:, 1] / yns[:, 1]
    tanG[0, :] = 0
    gamma = np.arctan(tanG)
    s, c = np.sin(gamma), np.cos(gamma)
    se = s * (c + 1j * s)
    se[:, 1:] *= 2
    se[0, :] = 0
    return se

def computeAngleDependentCylinderScatteringFilter(N, a, freqs, angles_rad):
    gtab = computeGammaTable(N, a, freqs)
    ns = np.arange(N + 1)
    cosn = np.cos(ns[na, :] * angles_rad[:, na])
    summ = np.sum(gtab[:, na, :] * cosn[na, :, :], axis=-1)

    inv_sqrt_f = np.zeros_like(freqs)
    mask = freqs > 0.0
    inv_sqrt_f[mask] = freqs[mask] ** -0.5
    const = (2 / pi) ** 0.5 * np.exp(1j * pi * 0.25)
    summ *= inv_sqrt_f[:, na] * const

    summ[0, :] = 1e-8
    return summ

def computeAngleDependentSphereRadiationFilter(maxOrder, a, freqs, angles_rad):
    import scipy.special as sp
    ns = np.arange(maxOrder + 1)
    kas = 2 * np.pi * freqs / C.soundVel * a
    jnps = np.array([sp.spherical_jn(ns, ka) for ka in kas])
    ynps = np.array([sp.spherical_yn(ns, ka) for ka in kas])
    hnps = jnps + 1j * ynps
    P = np.array([sp.lpn(maxOrder, z=np.cos(ang))[0] for ang in angles_rad])
    num = ((2 * ns + 1) * (-1j) ** (ns - 1)) * P
    num_LF = num[na, :, :] * (kas[:, na] ** ns[na, :])[:, na, :]
    den = (kas ** 2)[:, na] * hnps
    den_LF = 1j * (ns + 1) * sp.factorial2(2 * ns - 1)
    summ = num[na, :, :] / den[:, na, :]
    summ_LF = num_LF / den_LF
    for i, ka in enumerate(kas):
        for n in ns:
            if ka <= 0.01 * n:
                summ[i, :, n] = summ_LF[i, :, n]
    return np.sum(summ, axis=-1)

def getFilterBankForAngleDependentCylinderScattering(treeRad=0.25, fs=384000, steps=128, nAngleBins=180, N=50, fftLen=384):
    freqs, fh = getFreqsForGivenFilterLength(steps, fs)
    ang = np.arange(0, nAngleBins + 1) * (pi / nAngleBins)
    H = computeAngleDependentCylinderScatteringFilter(N, treeRad, fh, ang)
    irs = getIRFromSpectrum_irfft(H)
    irs = np.roll(irs, steps // 2, axis=0)
    return irs, rfft(irs, n=fftLen, axis=0)

def getFilterBankForSourceDirectivity(fs, ntaps, nAngleBins=180, birdHeadRad=0.025, maxOrder=80, fftLen=128, doPlot=0):
    freqs, fh = getFreqsForGivenFilterLength(ntaps, fs)
    ang = np.arange(0, nAngleBins + 1) * (pi / nAngleBins)
    H = computeAngleDependentSphereRadiationFilter(maxOrder, birdHeadRad, fh, ang)
    irs = getIRFromSpectrum_irfft(H)
    irs = np.roll(irs, ntaps // 2, axis=0)
    rms0 = sigp.getAweightedRMS(irs[:, 0], fs, energy=1)
    irs = irs / (rms0 ** 0.5)
    return irs, rfft(irs, n=fftLen, axis=0)

def getTreePoss(range_x, range_y, nTrees, algo="uniform", scene_seed=1234, minTreeSpacing_m=1.5):
    algo_l = str(algo).lower()
    rng = np.random.default_rng(scene_seed)
    if algo_l == "uniform":
        tp = rng.uniform(size=[nTrees, 2])
        tp[:, 0] = tp[:, 0] * (range_x[1] - range_x[0]) + range_x[0]
        tp[:, 1] = tp[:, 1] * (range_y[1] - range_y[0]) + range_y[0]
        return np.concatenate([tp, np.zeros([nTrees, 1])], axis=-1)
    if algo_l != "repulsive":
        raise NotImplementedError(f"Unknown tree sampling algo {algo!r}. Expected 'uniform' or 'repulsive'.")
    if minTreeSpacing_m <= 0:
        raise ValueError("minTreeSpacing_m must be > 0 for repulsive sampling.")
    max_attempts = max(1, 30 * int(nTrees))
    x0, x1 = float(range_x[0]), float(range_x[1])
    y0, y1 = float(range_y[0]), float(range_y[1])
    accepted = []
    attempts = 0
    while len(accepted) < int(nTrees) and attempts < max_attempts:
        attempts += 1
        cand = np.array([rng.uniform(x0, x1), rng.uniform(y0, y1)], dtype=np.float64)
        if not accepted:
            accepted.append(cand)
            continue
        arr = np.asarray(accepted, dtype=np.float64)
        d2 = np.sum((arr - cand[na, :]) ** 2, axis=1)
        if np.all(d2 >= float(minTreeSpacing_m) ** 2):
            accepted.append(cand)
    if len(accepted) < int(nTrees):
        print(
            f"Repulsive tree sampling: requested {int(nTrees)}, placed {len(accepted)} "
            f"after {attempts} attempts (min spacing {float(minTreeSpacing_m):.3f} m)."
        )
    tp = np.asarray(accepted, dtype=np.float64)
    if tp.size == 0:
        return np.zeros((0, 3), dtype=np.float64)
    return np.concatenate([tp, np.zeros([len(tp), 1])], axis=-1)

def sampleTreeTrunkRadii(
    n_trees: int,
    trunkDiameterRange_m: tuple = (0.5, 0.5),
    scene_seed: int = 1234,
) -> np.ndarray:
    dmin, dmax = float(trunkDiameterRange_m[0]), float(trunkDiameterRange_m[1])
    if dmin <= 0.0 or dmax <= 0.0 or dmin > dmax:
        raise ValueError(
            f"Invalid trunkDiameterRange_m={trunkDiameterRange_m!r}; expected positive (min, max) with min <= max."
        )
    rng = np.random.default_rng(None if scene_seed is None else (scene_seed + 17))
    diameters = rng.uniform(dmin, dmax, size=int(n_trees))
    return (0.5 * diameters).astype(floatX)

def getRotMat2D(theta):       c, s = np.cos(theta), np.sin(theta); return np.array([[c, -s], [s, c]])
def getRotMat3D_hori(theta):  c, s = np.cos(theta), np.sin(theta); return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
def getCosOf3Pnts(p1, p2, p3): v1, v2 = (p2 - p1)[:2], (p3 - p2)[:2]; return v1.dot(v2) / (norm(v1) * norm(v2))
def getCosOf2Vecs(v1, v2):     return v1.dot(v2) / (norm(v1) * norm(v2))
def getCossOf3Pnts(p1, p2s, p3):
    v1, v2 = (p2s[:, :2] - p1[:2]), (p3[:2] - p2s[:, :2])
    return np.sum(v1 * v2, axis=-1) / (np.sum(v1 ** 2, axis=-1) ** 0.5 * np.sum(v2 ** 2, axis=-1) ** 0.5)
def getCossOf2Vecs_batch(v1, v2s):
    return np.sum(v1 * v2s, axis=-1) / (np.sum(v1 ** 2, axis=-1) ** 0.5 * np.sum(v2s ** 2, axis=-1) ** 0.5)

def getAndStoreMultipleScatteringFilter(filterBank_TxA, angleBins, dicFilter):
    if len(angleBins) == 1:
        k = str(angleBins)
        if k not in dicFilter: dicFilter[k] = filterBank_TxA[:, angleBins[0]]
        return dicFilter[k], dicFilter
    k_prev, k_cur = str(angleBins[:-1]), str(angleBins)
    if k_prev in dicFilter:
        dicFilter[k_cur] = np.convolve(filterBank_TxA[:, angleBins[-1]], dicFilter[k_prev])
        return dicFilter[k_cur], dicFilter
    prev, _ = getAndStoreMultipleScatteringFilter(filterBank_TxA, angleBins[:-1], dicFilter)
    dicFilter[k_cur] = np.convolve(filterBank_TxA[:, angleBins[-1]], prev)
    return dicFilter[k_cur], dicFilter

def getAirFilterIdx(dist, D):         return min(int(dist // 10), D - 1)
def getAirFilterIdx_batch(dists_, Dary): return np.minimum((dists_ // 10).astype(int), (Dary - 1).astype(int))

def getBranchLeafScattererPoss(n_scatterers, bounds_x, bounds_y, bounds_z, scene_seed=4321):
    """Generates random 3D positions for branch/leaf scatterers."""
    rng = np.random.default_rng(scene_seed)
    scatter_poss = rng.uniform(size=[n_scatterers, 3])
    scatter_poss[:, 0] = scatter_poss[:, 0] * (bounds_x[1] - bounds_x[0]) + bounds_x[0]
    scatter_poss[:, 1] = scatter_poss[:, 1] * (bounds_y[1] - bounds_y[0]) + bounds_y[0]
    scatter_poss[:, 2] = scatter_poss[:, 2] * (bounds_z[1] - bounds_z[0]) + bounds_z[0]
    return scatter_poss.astype(floatX)

def getOmnidirectionalScattererFilter(reflectionCoef, ntaps=64, fftLen=384):
    """
    Generates a simple, omnidirectional scattering filter.
    This represents the constant scattering amplitude f0 from Ostashev et al. (2017).
    """
    ir = np.zeros(ntaps, dtype=floatX)
    ir[ntaps // 2] = reflectionCoef
    H = rfft(ir, n=fftLen, axis=0)
    return ir, H

def load_tree_positions_csv(
    path: str,
    *,
    delimiter: str = ",",
    has_header: bool = True,
    x_col=0, y_col=1,    
    z_val: float = 0.0,
    unit_scale: float = 1.0,
    xy_offset: tuple = (0.0, 0.0),
) -> np.ndarray:
    """
    Returns an array of shape [N, 3] with columns x,y,z (meters).
    Accepts either numeric column indices or named columns (has_header=True).
    """
    if has_header and (isinstance(x_col, str) and isinstance(y_col, str)):
        data = np.genfromtxt(path, delimiter=delimiter, names=True, dtype=None, encoding=None)
        if x_col not in data.dtype.names or y_col not in data.dtype.names:
            raise ValueError(f"Could not find columns {x_col!r}, {y_col!r} in header: {data.dtype.names}")
        x = np.asarray(data[x_col], dtype=float) * unit_scale + float(xy_offset[0])
        y = np.asarray(data[y_col], dtype=float) * unit_scale + float(xy_offset[1])
    else:
        try:
            arr = np.loadtxt(path, delimiter=delimiter, usecols=(int(x_col), int(y_col)))
        except Exception:
            arr = np.loadtxt(path, delimiter=delimiter, usecols=(int(x_col), int(y_col)), skiprows=1)
        x = arr[:, 0] * unit_scale + float(xy_offset[0])
        y = arr[:, 1] * unit_scale + float(xy_offset[1])

    z = np.full_like(x, float(z_val), dtype=float)
    out = np.stack([x, y, z], axis=-1).astype(floatX)
    if len(out.shape) != 2 or out.shape[1] != 3:
        raise ValueError(f"Parsed shape {out.shape} from {path!r}; expected [N,3].")
    return out

def simulateForestIR(
    nTrees, posSrc, micPoss, fs, sigLen_in_samples=None, forestRange_x=None, forestRange_y=None,
    reflectionCoef=2.0, treeRadiationExponent=0.7, applyCylindricalScatteringFilter=1, applyAirAbsorption=1,
    maxDist=20000.0, maxReflOrder=1, scene_seed=1234, floorReflectionCoef=None, ntaps_treeScattering=128,
    ntaps_airAbsorption=128, ntaps_delay=128, sourceDirectivity=0, ntaps_sourceDirectivity=128,
    sourceDirectivityVec=None, samplingAlgo="uniform", minTreeSpacing_m: float = 1.5, temp_c: float = None,
    rh_percent: float = 50.0,      
    pressure_hpa: float = 1013.25, 
    tree_csv_path: str = None,
    tree_csv_kwargs: dict = None,
    treePoss: np.ndarray = None,
    treeHeight: float = 20.0,
    trunkDiameterRange_m: tuple = (0.5, 0.5),
    trunkDiameterNBins: int = 8,
    applyBranchLeafScattering: bool = False,
    nBranchLeafScatterers: int = 5000,
    branch_leaf_bounds_z: tuple = (15.0, 25.0),
    nBranchLeafScatterersExtra: int = 0,
    branch_leaf_extra_bounds_z: tuple = (0.0, 0.5),
    branchLeafOmniReflectionCoef: float = 0.05,
    ntaps_branch_leaf: int = 64,
    groundType: str = "default",
):

    C.set_atmosphere(
        temp_c if temp_c is not None else 20.0,
        rh_percent=rh_percent,
        pressure_hPa=pressure_hpa,
    )
    assert maxReflOrder <= 1 and applyAirAbsorption == 1 and applyCylindricalScatteringFilter == 1
    print("simulating forest IR...")
    floorReflectionCoef = resolve_floor_reflection_coef(
        groundType=groundType,
        floorReflectionCoef=floorReflectionCoef,
    )

    if treePoss is not None and nTrees == 0:
        nTrees = int(treePoss.shape[0])
    if tree_csv_path is not None:
        tp = load_tree_positions_csv(tree_csv_path, **(tree_csv_kwargs or {}))
        treePoss = tp.astype(floatX)
        nTrees = int(treePoss.shape[0])
    elif (treePoss is None) and (nTrees > 0):
        treePoss = getTreePoss(
            forestRange_x, forestRange_y, nTrees, algo=samplingAlgo, scene_seed=scene_seed, minTreeSpacing_m=minTreeSpacing_m
        ).astype(floatX)
        nTrees = int(treePoss.shape[0])

    if nTrees == 0:
        treePoss = None
        ntaps_treeScattering = 0
        tree_bin_idx = None
    else:
        tree_radii = sampleTreeTrunkRadii(
            n_trees=nTrees,
            trunkDiameterRange_m=trunkDiameterRange_m,
            scene_seed=scene_seed,
        )
        if int(trunkDiameterNBins) <= 1:
            tree_bin_idx = np.zeros(nTrees, dtype=np.int32)
            radius_bin_centers = np.array([float(np.mean(tree_radii))], dtype=np.float64)
        else:
            n_bins = int(trunkDiameterNBins)
            radius_edges = np.linspace(float(np.min(tree_radii)), float(np.max(tree_radii)), n_bins + 1)
            radius_bin_centers = 0.5 * (radius_edges[:-1] + radius_edges[1:])
            tree_bin_idx = np.digitize(tree_radii, radius_edges[1:-1]).astype(np.int32)

    d_direct = dists(posSrc.reshape([1, 3]), micPoss)[0]

    posSrc_floor = posSrc.copy()
    posSrc_floor[2] *= -1
    d_floor = dists(posSrc_floor.reshape([1, 3]), micPoss)[0]

    if (treePoss is not None) and (nTrees > 0):
        d_src_tree_3d = dists(posSrc.reshape([1, 3]), treePoss)[0]
        d_tree_mic_3d = dists(micPoss, treePoss)
        d_reflect_ = d_src_tree_3d[na, :] + d_tree_mic_3d

        d_src_tree_2d = dists(posSrc.reshape([1, 3])[:, :2], treePoss[:, :2])[0]
        d_tree_mic_2d = dists(micPoss[:, :2], treePoss[:, :2])
        d_mult_ = d_src_tree_2d[na, :] * (d_tree_mic_2d ** treeRadiationExponent)

        src_in_range = posSrc[2] < treeHeight
        mics_in_range = micPoss[:, 2:3] < treeHeight
        valid_tree_mask = src_in_range & mics_in_range
    else:
        d_reflect_ = None
        d_mult_ = None
        valid_tree_mask = None

    branch_leaf_bands: list[tuple] = []
    if applyBranchLeafScattering and nBranchLeafScatterers > 0:
        branch_leaf_seed = None if scene_seed is None else (scene_seed + 1)
        scatterer_poss = getBranchLeafScattererPoss(
            nBranchLeafScatterers, forestRange_x, forestRange_y, branch_leaf_bounds_z, scene_seed=branch_leaf_seed
        )
        d_src_scatter = dists(posSrc.reshape([1, 3]), scatterer_poss)[0]
        d_scatter_mic = dists(micPoss, scatterer_poss)
        branch_leaf_bands.append((
            scatterer_poss,
            d_src_scatter[na, :] + d_scatter_mic,
            d_src_scatter[na, :] * d_scatter_mic,
            nBranchLeafScatterers,
        ))
    if applyBranchLeafScattering and nBranchLeafScatterersExtra > 0:
        extra_seed = None if scene_seed is None else (scene_seed + 2)
        scatterer_poss_extra = getBranchLeafScattererPoss(
            nBranchLeafScatterersExtra,
            forestRange_x,
            forestRange_y,
            branch_leaf_extra_bounds_z,
            scene_seed=extra_seed,
        )
        d_src_scatter_extra = dists(posSrc.reshape([1, 3]), scatterer_poss_extra)[0]
        d_scatter_mic_extra = dists(micPoss, scatterer_poss_extra)
        branch_leaf_bands.append((
            scatterer_poss_extra,
            d_src_scatter_extra[na, :] + d_scatter_mic_extra,
            d_src_scatter_extra[na, :] * d_scatter_mic_extra,
            nBranchLeafScatterersExtra,
        ))

    has_branch_leaf = applyBranchLeafScattering and len(branch_leaf_bands) > 0
    fftLen = ntaps_treeScattering + ntaps_airAbsorption + ntaps_delay + \
             (sourceDirectivity * ntaps_sourceDirectivity) + \
             (has_branch_leaf * ntaps_branch_leaf)
    Bt = 100

    if applyCylindricalScatteringFilter == 1 and nTrees > 0:
        fb_tree_FhxA_by_bin = {}
        for bi in np.unique(tree_bin_idx):
            _, fb_tree_FhxA = getFilterBankForAngleDependentCylinderScattering(
                treeRad=float(radius_bin_centers[int(bi)]),
                fs=fs,
                steps=ntaps_treeScattering,
                fftLen=fftLen,
            )
            fb_tree_FhxA_by_bin[int(bi)] = fb_tree_FhxA * reflectionCoef

    distBins = np.arange(0, maxDist + 10, 10)
    if applyAirAbsorption == 1:
        fb_air_TxD, fb_air_FhxD = getFilterBankForAirAbsorption(
            ntaps_airAbsorption, fs, dists_=distBins, fftLen=fftLen,
            temperature_in_celsius=temp_c if temp_c is not None else 20.0, 
            relativeHumidity_in_percent=rh_percent,                        
            pressure_in_kiloPascal=pressure_hpa / 10.0                     
        )
        D = fb_air_TxD.shape[1]
    
    if sourceDirectivity == 1:
        fb_dir_TxA, fb_dir_FhxA = getFilterBankForSourceDirectivity(fs, ntaps=ntaps_sourceDirectivity, fftLen=fftLen)
    
    if has_branch_leaf:
        _, H_scatter_omni_Fh = getOmnidirectionalScattererFilter(
            reflectionCoef=branchLeafOmniReflectionCoef, ntaps=ntaps_branch_leaf, fftLen=fftLen
        )

    nMic = len(micPoss)
    if sigLen_in_samples is None:
        all_dists = [d_direct, d_floor]
        if d_reflect_ is not None: all_dists.append(d_reflect_)
        for _, d_band_reflect, _, _ in branch_leaf_bands:
            all_dists.append(d_band_reflect)
        d_max = 0
        for d_array in all_dists:
            if d_array is not None:
                d_max = max(d_max, float(np.max(d_array)))
        pad_sec = 0.02
        sigLen_in_samples = int((d_max / C.soundVel + pad_sec) * fs)

    signal = np.zeros([sigLen_in_samples, nMic], dtype=floatX)
    impulse = getImpulse(1)
    T1 = len(signal)

    for m in range(nMic):
        print("computing ir from src to mic", m, "...")
        tic = time.time()

        direct = impulse
        if applyAirAbsorption == 1:
            direct = np.convolve(direct, fb_air_TxD[:, getAirFilterIdx(d_direct[m], D)])
        if sourceDirectivity == 1:
            cosd = np.clip(getCosOf2Vecs(sourceDirectivityVec, micPoss[m] - posSrc), -1, 1)
            dirF = fb_dir_TxA[:, int(np.arccos(cosd) * 180 / pi + 0.5)]
            direct = np.convolve(direct, dirF)
        y, dAdd = getDelayedSig(direct, fs, delayInSec=0.0, distance=d_direct[m])
        addSignal(signal, y, delayToAdd=dAdd, channel=m)

        floor = fb_air_TxD[:, getAirFilterIdx(d_floor[m], D)] if applyAirAbsorption == 1 else impulse
        if sourceDirectivity == 1:
            mic_r = micPoss[m].copy(); mic_r[2] *= -1
            cosd = np.clip(getCosOf2Vecs(sourceDirectivityVec, mic_r - posSrc), -1, 1)
            dirF = fb_dir_TxA[:, int(np.arccos(cosd) * 180 / pi + 0.5)]
            floor = np.convolve(floor, dirF)
        y, dAdd = getDelayedSig(floor, fs, delayInSec=0.0, distance=d_floor[m])
        addSignal(signal, y * floorReflectionCoef, delayToAdd=dAdd, channel=m)

        if nTrees > 0 and valid_tree_mask is not None:
            valid_indices_for_mic_m = np.where(valid_tree_mask[m, :])[0]
            if len(valid_indices_for_mic_m) > 0:
                treePoss_valid = treePoss[valid_indices_for_mic_m, :]
                coss = np.clip(getCossOf3Pnts(posSrc, treePoss_valid, micPoss[m]), -1, 1)
                angBins = (np.arccos(coss) * 180 / pi + 0.5).astype(int)

                if sourceDirectivity == 1:
                    treePoss_valid_dir = treePoss_valid.copy()
                    treePoss_valid_dir[:, 2] = micPoss[m, 2]
                    csd = np.clip(getCossOf2Vecs_batch(sourceDirectivityVec, treePoss_valid_dir - posSrc), -1, 1)
                    angBins_dir = (np.arccos(csd) * 180 / pi + 0.5).astype(int)

                nTrees_valid = len(valid_indices_for_mic_m)
                for tHead in range(0, nTrees_valid, Bt):
                    block_size = min(Bt, nTrees_valid - tHead)
                    idx_slice_local = slice(tHead, tHead + block_size)
                    idx_slice_global = valid_indices_for_mic_m[idx_slice_local]
                    
                    tree_bins_block = tree_bin_idx[idx_slice_global]
                    block_angles = angBins[idx_slice_local]
                    Hs = np.empty(
                        (fb_air_FhxD.shape[0], block_size),
                        dtype=fb_tree_FhxA_by_bin[int(tree_bins_block[0])].dtype
                    )
                    for bi in np.unique(tree_bins_block):
                        msk = tree_bins_block == bi
                        Hs[:, msk] = fb_tree_FhxA_by_bin[int(bi)][:, block_angles[msk]]
                    if applyAirAbsorption == 1:
                        Dary_block = np.ones([block_size], dtype=int) * D
                        Ha = fb_air_FhxD[:, getAirFilterIdx_batch(d_reflect_[m, idx_slice_global], Dary_block)]
                        H = Hs * Ha
                    else:
                        H = Hs
                    if sourceDirectivity == 1:
                        H *= fb_dir_FhxA[:, angBins_dir[idx_slice_local]]

                    ys, dAdds = sigp.getDelayedSig_batch_freqDomain(
                        H, fs, d_reflect_[m, idx_slice_global] / C.soundVel,
                        nIRwithoutDelay=ntaps_delay, fftLen=fftLen
                    )
                    ys /= d_mult_[m, idx_slice_global]
                    
                    T2 = fftLen
                    t1 = np.minimum(T1, dAdds + T2)
                    t2 = np.minimum(T2, T1 - dAdds)
                    idx_valid_delay = np.where(dAdds < T1)[0]
                    for k in idx_valid_delay:
                        signal[dAdds[k]:t1[k], m] += ys[:t2[k], k]

        if has_branch_leaf:
            Bs = 500
            for scatterer_poss, d_band_reflect, d_band_mult, n_band_scatterers in branch_leaf_bands:
                if sourceDirectivity == 1:
                    csd_branch_leaf = np.clip(
                        getCossOf2Vecs_batch(sourceDirectivityVec, scatterer_poss - posSrc), -1, 1
                    )
                    ang_bins_dir_branch_leaf = (np.arccos(csd_branch_leaf) * 180 / pi + 0.5).astype(int)

                for sHead in range(0, n_band_scatterers, Bs):
                    block_size = min(Bs, n_band_scatterers - sHead)
                    idx_slice = slice(sHead, sHead + block_size)

                    Hs_branch_leaf = np.tile(H_scatter_omni_Fh[:, na], (1, block_size))
                    if applyAirAbsorption == 1:
                        Dary_block = np.ones([block_size], dtype=int) * D
                        Ha_branch_leaf = fb_air_FhxD[:, getAirFilterIdx_batch(d_band_reflect[m, idx_slice], Dary_block)]
                        H = Hs_branch_leaf * Ha_branch_leaf
                    else:
                        H = Hs_branch_leaf
                    if sourceDirectivity == 1:
                        H *= fb_dir_FhxA[:, ang_bins_dir_branch_leaf[idx_slice]]

                    ys, dAdds = sigp.getDelayedSig_batch_freqDomain(
                        H, fs, d_band_reflect[m, idx_slice] / C.soundVel,
                        nIRwithoutDelay=ntaps_delay, fftLen=fftLen
                    )
                    ys /= d_band_mult[m, idx_slice]

                    T2 = fftLen
                    t1 = np.minimum(T1, dAdds + T2)
                    t2 = np.minimum(T2, T1 - dAdds)
                    idx_valid_delay = np.where(dAdds < T1)[0]
                    for k in idx_valid_delay:
                        signal[dAdds[k]:t1[k], m] += ys[:t2[k], k]

        print("  time:", time.time() - tic)
    return signal

def getMatDistance(a, b):             d = a - b; return np.sum(d * d.conj()).real ** 0.5
def computeDistancesOfMatrices(dic1, dic2):
    r = np.zeros([len(dic1), len(dic2)], dtype=np.float64)
    for i, (_, v) in enumerate(dic1.items()):
        for j, (_, w) in enumerate(dic2.items()):
            r[i, j] = getMatDistance(v, w)
    return r

def generateSampleForestIR():
    fs = 384000
    ir = simulateForestIR(
        nTrees=1000, posSrc=np.array([50, 50, 1.5]),
        micPoss=np.array([60, 50, 1.5]).reshape([1, 3]),
        fs=fs, forestRange_x=[0,100], forestRange_y=[0,100]
    )
    Wav.writeWav(ir, "sampleForestIR.wav", fs)

from scipy.io import wavfile as _wavfile
from scipy.signal import resample_poly as _resample_poly

def _to_float32(x):
    x = np.asarray(x)
    if np.issubdtype(x.dtype, np.integer):
        x = x.astype(np.float64) / max(1, np.iinfo(x.dtype).max)
    else:
        x = x.astype(np.float64)
    return np.clip(x, -1.0, 1.0).astype(np.float32)

def _resample_mono(x, fs_x, fs_target):
    if fs_x == fs_target: return x.astype(np.float32)
    from math import gcd
    g = gcd(fs_x, fs_target)
    return _resample_poly(x.astype(np.float64), fs_target // g, fs_x // g).astype(np.float32)

def render_array_from_source(
    src_wav_path, posSrc, micPoss, fs_ir=384000, temp_c=None, nTrees=0,
    forest_kwargs=None, scene_seed=1234
):
    forest_kwargs = {} if forest_kwargs is None else dict(forest_kwargs)
    ir_TxC = simulateForestIR(
        nTrees=nTrees, posSrc=posSrc, micPoss=micPoss, fs=fs_ir, temp_c=temp_c,
        scene_seed=scene_seed, **forest_kwargs
    )
    fs_src, s = _wavfile.read(src_wav_path)
    s = _to_float32(s.squeeze());  s = s.mean(axis=-1) if s.ndim > 1 else s
    s = _resample_mono(s, fs_src, fs_ir)
    conv = sigp.convolve(s, ir_TxC, domain="freq")
    return conv[:, 0, :].astype(np.float32), fs_ir, ir_TxC

def apply_noise_and_export(
    mic_TxC, fs,
    model_id: int = 1,
    snr_db: float = None,
    noise_level: float = 0.0,
    noise_kwargs: dict = None,
    noise_seed: int = None,
    out_multichannel_path: str = None,
    out_per_channel_dir: str = None,
    out_basename: str = "array",
):
    mic_TxC = np.asarray(mic_TxC, dtype=np.float32)

    if snr_db is not None:
        noisy = sigp.add_noise(
            mic_TxC, fs,
            model_id=model_id,
            snr_db=snr_db,
            noise_level=1.0,
            noise_seed=noise_seed,
            model_kwargs=(noise_kwargs or {})
        )

    else:
        if noise_level <= 0.0 or model_id is None:
            noisy = mic_TxC.copy()
        else:
            noisy_unit = sigp.add_noise(
                mic_TxC, fs,
                model_id=model_id,
                snr_db=None,
                noise_level=1.0,
                noise_seed=noise_seed,
                model_kwargs=(noise_kwargs or {})
            )

            noise_only = noisy_unit - mic_TxC
            rms_sig = float(np.sqrt(np.mean(mic_TxC ** 2)))
            rms_noise = float(np.sqrt(np.mean(noise_only ** 2)))

            if rms_noise < 1e-12 or rms_sig < 1e-12:
                noisy = noisy_unit
            else:
                scale = (float(noise_level) * rms_sig) / rms_noise
                noisy = mic_TxC + noise_only * scale

    if out_multichannel_path:
        Wav.writeWav(noisy, out_multichannel_path, fs)
    if out_per_channel_dir:
        os.makedirs(out_per_channel_dir, exist_ok=True)
        for ch in range(noisy.shape[1]):
            Wav.writeWav(
                noisy[:, ch],
                os.path.join(out_per_channel_dir, f"{out_basename}_mic{ch:02d}.wav"),
                fs,
            )
    return noisy


def synthesize_array_observation(
    src_wav_path, posSrc, micPoss,
    fs_ir: int = 384000, temp_c: float = None,
    nTrees: int = 0, forest_kwargs: dict = None,
    model_id: int = 1,
    snr_db: float = None,
    noise_level: float = 0.0,
    noise_kwargs: dict = None,
    scene_seed: int = 1234,
    noise_seed: int = 1234,
    out_multichannel_path: str = None,
    out_per_channel_dir: str = None,
    out_basename: str = "array",
):
    mic_TxC, fs_out, ir_TxC = render_array_from_source(
        src_wav_path, posSrc, micPoss,
        fs_ir=fs_ir, temp_c=temp_c, nTrees=nTrees, forest_kwargs=forest_kwargs,
        scene_seed=scene_seed
    )
    noisy = apply_noise_and_export(
        mic_TxC, fs_out,
        model_id=model_id,
        snr_db=snr_db,
        noise_level=noise_level,
        noise_kwargs=noise_kwargs,
        noise_seed=noise_seed,
        out_multichannel_path=out_multichannel_path,
        out_per_channel_dir=out_per_channel_dir,
        out_basename=out_basename,
    )
    return {
        "signals_TxC": noisy,
        "fs": fs_out,
        "ir_TxC": ir_TxC,
        "posSrc": posSrc,
        "micPoss": micPoss,
        "soundVel_mps": C.soundVel,
        "noise": {
            "model_id": model_id,
            "snr_db": snr_db,
            "noise_level": noise_level,
            "noise_seed": noise_seed,
            "noise_kwargs": noise_kwargs or {},
        },
        "scene_seed": scene_seed,
        "forest_kwargs": forest_kwargs or {},
    }

if __name__ == "__main__":
    generateSampleForestIR()
