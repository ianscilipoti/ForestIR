# Based partly on Kaneko & Gamper (2022), JASA Express Letters, DOI: 10.1121/10.0014809

import numpy as np
import os
import soundfile as sf

na = np.newaxis
join = os.path.join

def writeWav(ary, path, fs):
    ary_ = np.asarray(ary)
    if ary_.ndim == 1:
        ary_ = ary_.reshape([len(ary_), 1])
    _, ext = os.path.splitext(path)
    sf.write(path, ary_, fs, subtype="FLOAT", format=ext[1:].upper())
    max_amp = float(np.max(np.abs(ary_))) if ary_.size else 1.0
    return 1.0 / max_amp if max_amp > 0 else 1.0

if __name__ == "__main__":
    pass
