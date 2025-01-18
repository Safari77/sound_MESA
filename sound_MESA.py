"""
Install https://github.com/martini-alessandro/Maximum-Entropy-Spectrum
This script loads an audio file and it computes its PSD.
Specify start and end frequencies, and filename.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
#from matplotlib import font_manager
import soundfile as sf
from memspectrum import MESA
import memspectrum.GenerateTimeSeries as GenerateTimeSeries

if len(sys.argv) != 4:
    print("[freqstart] [freqend] [file] (freqend 0=Nyquist)")
    sys.exit(1)
fname = sys.argv[3]
plt.rcParams.update({'font.size': 10})
plt.rcParams['font.family'] = 'Iosevka SS08'
plt.rcParams['figure.dpi'] = 300

#loading data and preparing input to MESA
data, realrate = sf.read(fname)
ratestart = int(sys.argv[1])
rateend = int(sys.argv[2])
if rateend == 0:
    rateend = int(realrate/2)

#data is (N,2): stereophonic sound
# Check if the audio is mono (1D) or stereo (2D)
if data.ndim == 1:
    # Convert mono to stereo by duplicating the channel
    data = np.stack((data, data), axis=-1)
# Calculate the length of the audio file in seconds
t = data.shape[0] / realrate  # Total samples divided by sample rate
print(f"Processing \"{fname}\", length {t} seconds, data Nyquist freq {realrate/2}, Analysis freq {ratestart}-{rateend}")
data_MESA = data[:int(t * realrate), 0].astype(np.float64)
dt = 1./realrate
times_out = np.linspace(0., len(data_MESA)*dt, len(data_MESA))

#computing PSD with MESA
M = MESA()
P, ak, opt = M.solve(data_MESA, method = "Standard", optimisation_method = "FPE",
                     m = int(2*len(data_MESA)/(2*np.log(len(data_MESA)))))

#evaluating the spectrum
N_points = 1000000
f_PSD = np.linspace(ratestart, rateend, N_points)
PSD = M.spectrum(dt, f_PSD)

fig, ax = plt.subplots(1, sharex = True)
plt.plot(f_PSD, PSD.real)
plt.yscale('log')
plt.ylabel('PSD')
plt.xlabel("frequency (Hz)")

# Find min and max PSD values
min_idx_y = np.argmin(PSD.real)
max_idx_y = np.argmax(PSD.real)

# Scatter points for min and max
plt.scatter(f_PSD[max_idx_y], PSD.real[max_idx_y], color='red',
            label=f'Max: {PSD.real[max_idx_y]:.4e} @{f_PSD[max_idx_y]:.4f}Hz', s=15)
plt.scatter(f_PSD[min_idx_y], PSD.real[min_idx_y], color='green',
            label=f'Min: {PSD.real[min_idx_y]:.4e} @{f_PSD[min_idx_y]:.4f}Hz', s=15)

plt.title(fname)
plt.tight_layout() # Automatically adjusts borders
plt.legend()
plt.show()

