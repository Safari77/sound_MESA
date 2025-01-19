"""
Install https://github.com/martini-alessandro/Maximum-Entropy-Spectrum
This script loads an audio file and it computes its PSD.

Loudness can be normalized according to EBU R128, positive values are accepted
but not recommended.
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
# https://python-soundfile.readthedocs.io/en/0.11.0/
sfreq = "0.11.0"
import soundfile as sf
import pyloudnorm as pyln
from memspectrum import MESA
import memspectrum.GenerateTimeSeries as GenerateTimeSeries
from packaging import version
import sys

try:
    if version.parse(sf.__version__) < version.parse(sfreq):
        raise ImportError(f"SoundFile version >= {sfreq} is required, but {sf.__version__} is installed.")
except ImportError as e:
    sys.stderr.write(f"{e}\n")
    sys.exit(1)

# Setup command line argument parsing
parser = argparse.ArgumentParser(description='Analyze audio spectrum with optional loudness normalization.')
parser.add_argument('filename', type=str, 
                    help='Path to the audio file')
parser.add_argument('--verbose', action='store_true',
                    help='Show verbose information about the audio file')
parser.add_argument('--logfreq', action='store_true',
                    help='Use logarithmic frequency scale (maybe you also want --ratestart 20 or so)')
parser.add_argument('--ratestart', type=int, default=0, 
                    help='Start frequency for analysis (default: 0)')
parser.add_argument('--rateend', type=int, default=0, 
                    help='End frequency for analysis, 0 means Nyquist (default: 0)')
parser.add_argument('--normalize', type=float, 
                    help='Target loudness in LUFS for normalization (optional)')

args = parser.parse_args()

# Extract arguments
fname = args.filename
verbose = args.verbose
logfreq = args.logfreq
ratestart = args.ratestart
rateend = args.rateend
target_loudness = args.normalize

plt.rcParams.update({'font.size': 10})
plt.rcParams['font.family'] = 'Iosevka SS08'
plt.rcParams['figure.dpi'] = 300

if verbose:
    print(sf.info(fname))
# Loading data and preparing input to MESA
data, realrate = sf.read(fname, always_2d=True)
if rateend == 0:
    rateend = int(realrate / 2)

# Loudness normalization to EBU R128 if specified
if target_loudness is not None:
    meter = pyln.Meter(realrate)  # create BS.1770 meter
    loudness = meter.integrated_loudness(data)
    print(f"Integrated loudness before normalization: {loudness} LUFS")
    normalized_audio = pyln.normalize.loudness(data, loudness, target_loudness)
    normalized_loudness = meter.integrated_loudness(normalized_audio)
    print(f"Integrated loudness after normalization: {normalized_loudness} LUFS")
else:
    normalized_audio = data

# Calculate the length of the audio file in seconds
t = normalized_audio.shape[0] / realrate  # Total samples divided by sample rate
print(f"Processing \"{fname}\", length {t} seconds, data Nyquist freq {realrate / 2}, Analysis freq {ratestart}-{rateend}")
data_MESA = normalized_audio[:int(t * realrate), 0].astype(np.float64)
dt = 1. / realrate

# Computing PSD with MESA
M = MESA()
P, ak, opt = M.solve(data_MESA, method="Standard", optimisation_method="FPE",
                     m=int(2 * len(data_MESA) / (2 * np.log(len(data_MESA)))))

# Evaluating the spectrum
N_points = 1000000
f_PSD = np.linspace(ratestart, rateend, N_points)
PSD = M.spectrum(dt, f_PSD)

fig, ax = plt.subplots(1, sharex=True)
plt.plot(f_PSD, PSD.real)
plt.yscale('log')
plt.ylabel('PSD')
if logfreq:
    plt.xscale('log')
plt.xlabel("frequency (Hz)")

# Find min and max PSD values
min_idx_y = np.argmin(PSD.real)
max_idx_y = np.argmax(PSD.real)
print(f"Max frequency {f_PSD[max_idx_y]:.2f} Hz")

# Scatter points for min and max
plt.scatter(f_PSD[max_idx_y], PSD.real[max_idx_y], color='red',
            label=f'Max: {PSD.real[max_idx_y]:.4e} @{f_PSD[max_idx_y]:.4f}Hz', s=15)
plt.scatter(f_PSD[min_idx_y], PSD.real[min_idx_y], color='green',
            label=f'Min: {PSD.real[min_idx_y]:.4e} @{f_PSD[min_idx_y]:.4f}Hz', s=15)

plt.title(fname)
plt.tight_layout()  # Automatically adjusts borders
plt.legend()
plt.show()
