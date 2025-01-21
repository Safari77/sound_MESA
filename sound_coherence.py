import argparse
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
from scipy import signal
import sys
from packaging import version

# Check SoundFile version
sfreq = "0.11.0"
try:
    if version.parse(sf.__version__) < version.parse(sfreq):
        raise ImportError(f"SoundFile version >= {sfreq} is required, but {sf.__version__} is installed.")
except ImportError as e:
    sys.stderr.write(f"{e}\n")
    sys.exit(1)

# Setup command line argument parsing
parser = argparse.ArgumentParser(description='Analyze audio spectrum with optional loudness normalization.')
parser.add_argument('filename', type=str, help='Path to the audio file')
parser.add_argument('--window', type=str, default='hann', help='Coherence window function')
parser.add_argument('--nperseg', type=int, default=1024, help='Length of each segment')
parser.add_argument('--verbose', action='store_true', help='Show verbose information about the audio file')
parser.add_argument('--ratestart', type=int, default=0, help='Start frequency for analysis (default: 0)')
parser.add_argument('--rateend', type=int, default=0, help='End frequency for analysis, 0 means Nyquist (default: 0)')
args = parser.parse_args()

# Extract arguments
fname = args.filename
verbose = args.verbose
ratestart = args.ratestart
rateend = args.rateend
window = args.window
nperseg = args.nperseg
if nperseg < 8:
    print(f'nperseg should be >= 8')
    exit(1)

plt.rcParams.update({'font.size': 10})
plt.rcParams['font.family'] = 'Iosevka SS08'
plt.rcParams['figure.dpi'] = 300

if verbose:
    print(sf.info(fname))

# Loading data and preparing input to MESA
data, realrate = sf.read(fname, always_2d=False)
if rateend == 0:
    rateend = int(realrate / 2)

# Calculate the length of the audio file in seconds
t = data.shape[0] / realrate  # Total samples divided by sample rate
print(f"Processing \"{fname}\", length {t} seconds, data Nyquist freq {realrate / 2}, Analysis freq {ratestart}–{rateend}")

# Check if the audio file has at least two channels
if data.ndim == 1 or data.shape[1] < 2:
    sys.stderr.write("Error: Audio file must have at least two channels.\n")
    sys.exit(1)

# Extract the first and second channels
channel_1 = data[:int(t * realrate), 0].astype(np.float64)
channel_2 = data[:int(t * realrate), 1].astype(np.float64)

# Estimate the magnitude squared coherence estimate
f, Cxy = signal.coherence(channel_1, channel_2, fs=realrate, nperseg=nperseg, window=window)

# Filter frequencies and coherence values to the specified range
mask = (f >= ratestart) & (f <= rateend)
f_filtered = f[mask]
Pxy_filtered = Cxy[mask]

# Find min and max values within the filtered range
min_idx_y = np.argmin(Pxy_filtered)
max_idx_y = np.argmax(Pxy_filtered)
min_val = Pxy_filtered[min_idx_y]
max_val = Pxy_filtered[max_idx_y]
min_freq = f_filtered[min_idx_y]
max_freq = f_filtered[max_idx_y]
print(f"Min/Max Coherence {min_val:.3f}@{min_freq:.2f}Hz / {max_val:.3f}@{max_freq:.2f}Hz")

title=f'Coherence for {fname}'
if np.allclose(Cxy, 1.0):
    print("The two channels are identical.")
    title=f'Channels identical in {fname}'

# Plot the results
plt.plot(f_filtered, Pxy_filtered, label='Coherence', linewidth=1)
plt.scatter(f_filtered[min_idx_y], Pxy_filtered[min_idx_y], color='red',
            label=f'Min: {min_val:.2f} @{min_freq:.2f}', zorder=5, s=15)
plt.scatter(f_filtered[max_idx_y], Pxy_filtered[max_idx_y], color='green',
            label=f'Max: {max_val:.2f} @{max_freq:.2f}', zorder=5, s=15)
plt.ylim([-0.05, 1.05])
plt.xlabel(f'Frequency [Hz]')
plt.ylabel(f'Coherence [V²/Hz] ({window})')
plt.title(title)

# Add min/max values to the legend
min_label = f'Min {min_val:.3f}@{min_freq:.2f}Hz'
max_label = f'Max {max_val:.3f}@{max_freq:.2f}Hz'
plt.legend([f'Coherence {ratestart}–{rateend} Hz', min_label, max_label], loc='best')

plt.show()

