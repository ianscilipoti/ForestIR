# Based partly on Kaneko & Gamper (2022), JASA Express Letters, DOI: 10.1121/10.0014809

__author__ = "Colin"

import os
import math
import glob
import numpy as np
from scipy.signal import stft as spstft, istft as spistft, firwin, lfilter, resample_poly as _resample_poly
from scipy.fft import rfft, irfft, ifft
from scipy.io import wavfile as _wavfile
import soundfile as sf
import Constants as C

na = np.newaxis
join = os.path.join
floatX = np.float64
pi = np.pi

def stft(sig_TxnCh, fs, winSize, stepSize=None):
    stepSize = winSize // 2 if stepSize is None else stepSize
    noverlap = winSize - stepSize
    assert winSize > stepSize
    freqs, times, spec = spstft(
        sig_TxnCh, fs=fs, window="hann", nperseg=winSize, noverlap=noverlap,
        nfft=None, detrend=False, return_onesided=False, boundary="zeros", padded=True, axis=0
    )
    if sig_TxnCh.ndim == 2:
        spec = np.asarray(spec).transpose(2, 0, 1)
    elif sig_TxnCh.ndim == 1:
        spec = spec.T
    else:
        raise NotImplementedError
    return freqs, times, spec

def istft(spec_TxFxnCh, fs, winSize, stepSize):
    stepSize = winSize // 2 if stepSize is None else stepSize
    noverlap = winSize - stepSize
    assert winSize > stepSize
    _, sig = spistft(
        spec_TxFxnCh, fs=fs, window="hann", nperseg=winSize, noverlap=noverlap,
        nfft=None, input_onesided=False, boundary=True, time_axis=0, freq_axis=1
    )
    return sig

def getZeroPadded(sig, retLen):
    if sig.ndim == 1:
        T = len(sig); assert retLen > T
        return np.concatenate([sig, np.zeros(retLen - T, dtype=sig.dtype)])
    if sig.ndim == 2:
        T, Cn = sig.shape; assert retLen > T
        return np.concatenate([sig, np.zeros([retLen - T, Cn], dtype=sig.dtype)], axis=0)
    raise NotImplementedError

def dist(p1, p2):            return np.sum((p1 - p2) ** 2, axis=-1) ** 0.5
def dist_hori(p1, p2):       return np.sum((p1 - p2)[:2] ** 2, axis=-1) ** 0.5
def dists(ps1, ps2):         return np.sum((ps1[:,na,:] - ps2[na,:,:]) ** 2, axis=-1) ** 0.5

def delayInSec(p1, p2):      return dist(p1, p2) / C.soundVel
def delaysInSec(ps1, ps2):   return dists(ps1, ps2) / C.soundVel
def delayInSamples(p1,p2,fs):return dist(p1, p2) / C.soundVel * fs

def convolve(sig1, sig2, domain="time"):
    s1 = sig1.reshape(-1, 1) if sig1.ndim == 1 else sig1
    s2 = sig2.reshape(-1, 1) if sig2.ndim == 1 else sig2
    if domain == "time":
        out = [[np.convolve(s1[:, c1], s2[:, c2]) for c2 in range(s2.shape[1])] for c1 in range(s1.shape[1])]
        return np.asarray(out).transpose(2, 0, 1)
    if domain == "freq":
        n = len(s1) + len(s2) - 1
        S1 = rfft(s1, n=n, axis=0)
        S2 = rfft(s2, n=n, axis=0)
        Yf = S1[:, :, na] * S2[:, na, :]
        return irfft(Yf, axis=0).real
    raise NotImplementedError

def getDelayedSig(sig, fs, delayInSec=None, distance=None, nIRwithoutDelay=128):
    assert sig.ndim == 1
    if delayInSec is None or delayInSec == 0.0:
        assert distance is not None
        delayInSec = distance / C.soundVel
    distance = 1.0 if (distance is None or distance == 0.0) else distance
    dSamp = delayInSec * fs
    irLen = int(dSamp + nIRwithoutDelay // 2)
    delayToAdd = int(dSamp - nIRwithoutDelay // 2) if dSamp > nIRwithoutDelay // 2 else 0
    ts = np.arange(delayToAdd, irLen)
    x = pi * (ts - dSamp)
    ir = np.sin(x) / (x * distance)
    if irLen > dSamp and dSamp == int(dSamp):
        ir[int(dSamp) - delayToAdd] = 1.0 / distance
    return np.convolve(sig, ir), delayToAdd

def getDelayedSig_batch(sigs, fs, delaysInSec=None, nIRwithoutDelay=128):
    assert sigs.ndim == 2
    T, Cn = sigs.shape
    dSamp = delaysInSec * fs
    dInt = dSamp.astype(int)
    delayToAdd = dInt - int(nIRwithoutDelay // 2)
    delayToAdd[dSamp <= nIRwithoutDelay // 2] = 0
    ts = (np.arange(nIRwithoutDelay).reshape(-1, 1) + delayToAdd)
    x = pi * (ts - dSamp)
    with np.errstate(divide="ignore", invalid="ignore"):
        irs = np.sin(x) / x
    singulars = (x == 0) & (dSamp == dInt)
    irs[singulars] = 1.0
    out = np.array([np.convolve(sigs[:, i], irs[:, i]) for i in range(Cn)]).T
    return out, delayToAdd

def getDelayedSig_batch_freqDomain(sigs, fs, delaysInSec=None, nIRwithoutDelay=128, fftLen=384):
    assert sigs.ndim == 2
    _, Cn = sigs.shape
    dSamp = delaysInSec * fs
    dInt = dSamp.astype(np.int32)
    delayToAdd = dInt - int(nIRwithoutDelay // 2)
    delayToAdd[dSamp <= nIRwithoutDelay // 2] = 0
    ts = (np.arange(nIRwithoutDelay).reshape(-1, 1) + delayToAdd)
    x = pi * (ts - dSamp)
    irs = np.sin(x) / x
    irs[(x == 0) & (dSamp == dInt)] = 1.0
    irs_f = rfft(irs, n=fftLen, axis=0)
    y = irfft(sigs * irs_f, axis=0)
    return y, delayToAdd

def getFreqsForGivenFilterLength(nTaps, fs):
    freqs = np.arange(nTaps) * (fs / nTaps)
    return freqs, freqs[: nTaps // 2 + 1]

def getHPF(nTaps, fs, fcut):
    assert nTaps % 2 == 1
    return firwin(nTaps, cutoff=fcut, fs=fs, pass_zero=False)

def getBPF(nTaps, fs, f_low, f_high):
    assert nTaps % 2 == 1
    return firwin(nTaps, cutoff=[f_low, f_high], fs=fs, pass_zero=False)

def get_dB_from_amplitude(spec, eps=1e-8): return 20 * np.log(np.abs(spec) + eps) / np.log(10)
def getAmplitudeFromdB(dB):               return np.exp(dB * np.log(10) / 20)

def getIRFromSpectrum(spec_FxC):          return ifft(spec_FxC, axis=0).real
def getIRFromSpectrum_irfft(spec_hFxC):   return irfft(spec_hFxC, axis=0).real

def getLongerSignalByRepeating(sig, desiredLen):
    T = len(sig); nRep = int(desiredLen / T) + 1
    rep = np.tile(sig, reps=[nRep, 1]) if sig.ndim == 2 else np.tile(sig, reps=nRep)
    return rep[:desiredLen]

def addSignal(sig1, sig2, delayToAdd, channel):
    assert int(delayToAdd) == delayToAdd and channel is not None
    T1, T2 = len(sig1), len(sig2)
    t1, t2 = min(T1, delayToAdd + T2), min(T2, T1 - delayToAdd)
    if delayToAdd < T1:
        sig1[delayToAdd:t1, channel] += sig2[:t2]

def getSoundPressure_in_Pascal_from_dBSPL(dBSPL):      return getAmplitudeFromdB(dBSPL) * 2e-5
def getSoundPressureLevel_in_dBSPL_from_Pascal(pa):    return 20 * np.log10(pa / 2e-5)

def getSignal_scaled_to_desired_SPL_Aweighted(sig, fs, spl_dB):
    rms_A = getAweightedRMS(sig, fs)
    return sig * (getSoundPressure_in_Pascal_from_dBSPL(spl_dB) / rms_A)

def getSNR(sig, noise, fs, doAweighting=0):
    s = getAweightedRMS(sig, fs) if doAweighting else getRMS(sig)
    n = getAweightedRMS(noise, fs) if doAweighting else getRMS(noise)
    return 20 * np.log10(s / n)

def getRMS(x):     return (np.mean(x ** 2) + 1e-12) ** 0.5
def getEnergy(x):  return np.sum(x ** 2)

def getLinearGainCoefsOfAweighting(freqs):
    import librosa
    return getAmplitudeFromdB(librosa.core.A_weighting(freqs))

def getAweightedRMS(sig_t, fs, energy=0):
    import scipy as sp
    X = sp.fft.rfft(sig_t)
    freqs = sp.fft.rfftfreq(len(sig_t), 1 / fs)
    W = getLinearGainCoefsOfAweighting(freqs)
    xw = sp.fft.irfft(X * W)
    return getEnergy(xw) if energy else getRMS(xw)

def getImpulse(sigLen):
    x = np.zeros(sigLen, dtype=np.float64)
    x[0] = 1
    return x

_SUPPORTED_AUDIO_EXTS = ("*.wav", "*.flac")

def _find_single_audio_in_dir(noise_dir):
    """Find exactly one audio file (.wav or .flac) under *noise_dir* (recursive)."""
    found = []
    for pattern in _SUPPORTED_AUDIO_EXTS:
        found.extend(glob.glob(os.path.join(noise_dir, "**", pattern), recursive=True))
    found = sorted(set(found))
    if len(found) != 1:
        raise ValueError(
            f"Expected exactly one audio file (.wav/.flac) under {noise_dir!r}, found {len(found)}: {found}"
        )
    return found[0]

def _read_audio_file(path):
    """Read .wav or .flac, return (float32 mono ndarray, sample_rate)."""
    n, fs_n = sf.read(path, dtype="float32")
    if n.ndim > 1:
        n = n.mean(axis=-1)
    return n.astype(np.float32), fs_n

def _load_random_segments_per_channel_from_single_wav_in_dir(noise_dir, target_len, fs_target, Cn, rng=None):
    """
    noise_dir: directory that contains EXACTLY ONE audio file (.wav or .flac).
    Returns [Cn, target_len]; each row is an independent random segment (may overlap).
    """
    rng = np.random.default_rng(rng) if not isinstance(rng, np.random.Generator) else rng
    path = _find_single_audio_in_dir(noise_dir)
    n, fs_n = _read_audio_file(path)
    n = resample_mono(n, fs_n, fs_target)
    N = len(n)
    out = np.zeros((Cn, target_len), dtype=np.float32)
    for c in range(Cn):
        if N >= target_len:
            start = int(rng.integers(0, N - target_len + 1))
            out[c] = n[start : start + target_len]
        else:
            reps = int(np.ceil(target_len / N))
            n_long = np.tile(n, reps)
            start = int(rng.integers(0, len(n_long) - target_len + 1))
            out[c] = n_long[start : start + target_len]
    return out


def _to_float32(x):
    x = np.asarray(x)
    if np.issubdtype(x.dtype, np.integer):
        x = x.astype(np.float64) / max(1, np.iinfo(x.dtype).max)
    else:
        x = x.astype(np.float64)
    return np.clip(x, -1.0, 1.0).astype(np.float32)

def resample_mono(x, fs_x, fs_target):
    if fs_x == fs_target: return x.astype(np.float32)
    from math import gcd
    g = gcd(fs_x, fs_target)
    return _resample_poly(x.astype(np.float64), fs_target // g, fs_x // g).astype(np.float32)

def _load_random_bg_clip(noise_dir, target_len, fs_target, rng=None):
    rng = np.random.default_rng(rng) if not isinstance(rng, np.random.Generator) else rng
    found = []
    for pattern in _SUPPORTED_AUDIO_EXTS:
        found.extend(glob.glob(os.path.join(noise_dir, "**", pattern), recursive=True))
    found = sorted(set(found))
    if not found:
        raise FileNotFoundError(f"No audio files (.wav/.flac) under {noise_dir!r}")
    path = found[int(rng.integers(0, len(found)))]
    n, fs_n = _read_audio_file(path)
    n = resample_mono(n, fs_n, fs_target)
    if len(n) >= target_len:
        start = int(rng.integers(0, len(n) - target_len + 1))
        return n[start:start + target_len]
    reps = int(math.ceil(target_len / len(n)))
    return np.tile(n, reps)[:target_len]

def _rms_per_ch(x):  return np.array([getRMS(ch) for ch in (x if x.ndim == 2 else [x])], dtype=np.float64)

def gen_awgn(shape, rng=None):
    rng = np.random.default_rng(rng) if not isinstance(rng, np.random.Generator) else rng
    Cn, T = shape
    return rng.standard_normal((Cn, T)).astype(np.float32)

def gen_pink(shape, rng=None):
    rng = np.random.default_rng(rng) if not isinstance(rng, np.random.Generator) else rng
    Cn, T = shape
    out = np.zeros((Cn, T), dtype=np.float32)
    for c in range(Cn):
        w = rng.standard_normal(T).astype(np.float32)
        W = np.fft.rfft(w.astype(np.float64))
        f = np.fft.rfftfreq(T, d=1.0); f[0] = 1.0
        y = np.fft.irfft(W * (1.0 / np.sqrt(f)), n=T)
        out[c] = (y / getRMS(y)).astype(np.float32)
    return out

def gen_dataset_noise(shape, fs, noise_dir, rng=None, per_channel_desync=True):
    """
    Single-file dataset noise:
      - noise_dir contains EXACTLY ONE audio file (.wav or .flac).
      - For each mic, draw an independent random segment of length T.
      - Returns unit-RMS per channel, shape [C, T].
    """
    Cn, T = shape
    out = _load_random_segments_per_channel_from_single_wav_in_dir(
        noise_dir, target_len=T, fs_target=fs, Cn=Cn, rng=rng
    ).astype(np.float32)
    return (out.T / (_rms_per_ch(out) + 1e-12)).T.astype(np.float32)

def gen_brown(shape, rng=None):
    rng = np.random.default_rng(rng) if not isinstance(rng, np.random.Generator) else rng
    Cn, T = shape
    out = np.zeros((Cn, T), dtype=np.float32)
    for c in range(Cn):
        y = np.cumsum(rng.standard_normal(T)).astype(np.float64)
        y -= y.mean(); out[c] = (y / getRMS(y)).astype(np.float32)
    return out

def gen_bandlimited(shape, fs, f_low=300.0, f_high=3000.0, rng=None):
    rng = np.random.default_rng(rng) if not isinstance(rng, np.random.Generator) else rng
    Cn, T = shape
    n_taps = min(2049, max(129, int(4 * fs / max(1.0, f_low))))
    if n_taps % 2 == 0: n_taps += 1
    b = firwin(n_taps, [f_low, f_high], fs=fs, pass_zero=False)
    out = np.zeros((Cn, T), dtype=np.float32)
    for c in range(Cn):
        w = rng.standard_normal(T + n_taps)
        y = lfilter(b, [1.0], w)[n_taps:]
        y -= y.mean(); out[c] = (y / getRMS(y)).astype(np.float32)
    return out

def gen_noise_model_1_awgn(shape, rng=None):                           return gen_awgn(shape, rng=rng)
def gen_noise_model_2_pink(shape, rng=None):                           return gen_pink(shape, rng=rng)
def gen_noise_model_3_dataset(shape, fs, noise_dir, rng=None, per_channel_desync=True):
    return gen_dataset_noise(shape, fs=fs, noise_dir=noise_dir, rng=rng, per_channel_desync=per_channel_desync)
def gen_noise_model_4_brown(shape, rng=None):                          return gen_brown(shape, rng=rng)
def gen_noise_model_5_bandlimited(shape, fs, f_low=300.0, f_high=3000.0, rng=None):
    return gen_bandlimited(shape, fs=fs, f_low=f_low, f_high=f_high, rng=rng)

def map_noise_level_to_gain(noise_level, min_gain_db=-40.0, max_gain_db=-6.0):
    """
    Map noise_level in [0,1] to a fixed linear gain, independent of signal.
    -40 dBFS (quiet) .. -6 dBFS (loud) by default.
    """
    nl = float(np.clip(noise_level, 0.0, 1.0))
    gain_db = min_gain_db + (max_gain_db - min_gain_db) * nl
    return 10.0 ** (gain_db / 20.0)


_NOISE_MODELS = {
    1: "awgn",
    2: "pink",
    3: "dataset",
    4: "brown",
    5: "bandlimited",
}

def _gen_noise_from_model_id(model_id, shape, fs, rng=None, **kw):
    if   model_id == 1: return gen_noise_model_1_awgn(shape, rng=rng)
    elif model_id == 2: return gen_noise_model_2_pink(shape, rng=rng)
    elif model_id == 3: return gen_noise_model_3_dataset(shape, fs=fs, rng=rng, **kw)
    elif model_id == 4: return gen_noise_model_4_brown(shape, rng=rng)
    elif model_id == 5: return gen_noise_model_5_bandlimited(shape, fs=fs, rng=rng, **kw)
    else: raise ValueError(f"Unknown noise model id: {model_id}")

def add_noise(signals_TxC, fs, model=None, model_id=None, snr_db=None, noise_level=0.0, noise_seed=None, model_kwargs=None):
    """
    Add noise at a FIXED level (same gain for all mics), independent of signal.
    Resulting SNR will vary across mics as a function of propagation/scene.
    - If noise_level <= 0: returns input.
    - snr_db is ignored (kept only for backward compatibility).
    """
    model_kwargs = {} if model_kwargs is None else dict(model_kwargs)
    rng = np.random.default_rng(noise_seed)

    x = signals_TxC.astype(np.float32)
    T, Cn = x.shape
    if float(noise_level) <= 0.0:
        return x

    if model_id is None:
        if model is None:
            model_id = 1
        else:
            m = model.strip().lower()
            inv = {v: k for k, v in _NOISE_MODELS.items()}
            if m not in inv:
                raise ValueError(f"Unknown noise model '{m}'. Valid: {list(inv.keys())} or model_id=1..5")
            model_id = inv[m]

    n_CxT = _gen_noise_from_model_id(model_id, (Cn, T), fs, rng=rng, **model_kwargs)
    g = map_noise_level_to_gain(noise_level)
    n_scaled = (n_CxT * g).astype(np.float32)
    y = (x.T + n_scaled).T
    return np.clip(y, -1.0, 1.0).astype(np.float32)


if __name__ == "__main__":
    pass
