""" 
CROSS SYNTHESIS APPLICATION
argument: # of linear coefficients (M)
argument: window size (L)
arguemnt: window function (--w) takes b for bartlett or h for hanning
argument: flag to flatten spectrum (--flatten, optional, default False)
argument: flag to plot the spectrograms (--plot, optional, default False)
"""
#==============Imports and Load Audio===============#
import numpy as np
import os.path
import matplotlib.pyplot as plt
from IPython.display import Audio
from scipy.io import wavfile
from scipy import signal
import argparse
from pathlib import Path 
import simpleaudio as same
import sounddevice as sd

#==============Window Functions===============#

def get_hamming(L):
    """ returns coefficients for hamming window of length L"""
    # TODO: doesn't work, reconstruction sounds clippy
    return 0.54 - (0.46*np.cos(2*np.pi*np.arange(L)/L))

def get_bartlett(L):
    """ returns coefficients for bartlett/triangular window 
        of length L
        window should have 50% 
        (hop size R should be window size M / 2)"""
    half = 2* np.arange(L//2) / L
    return np.concatenate((half, np.flip(half)))
def get_hanning(L):
    """ returns coefficients for hanning window of length L"""
    ''' 50% overlap 
         np.ones(1024), run hanning, run through istft
    '''
    return 0.5* (1.001-np.cos(2*np.pi*np.arange(L)/L))



#==============STFT Functions===============#
def get_windowed_signal(signal, L, R, w=None):
    """
    Partitions signal into windows of length M, separated by R samples
    returns: M x N matrix, where M is the window size, and N is number of windows
    """ 
    xms = np.array([np.zeros(L)]).T  # ini
    signal = np.concatenate([np.zeros(L//2), signal, np.zeros(L//2)])  # for COLA reconstruction
        # TODO: do we need to strip these zeros in istft?
#     print(signal)
    ms = range(0, len(signal), R)
    for m in ms:
#         print(f'frame is {m} and x frame is {signal[m:]}')
        xm = signal[m:m+L].copy()
#         print(f'frame is {m} and x frame is {xm}')
        if len(xm) < L:
            xm = np.concatenate([xm, np.zeros(L-len(xm))])
        if w is not None:  # apply window fn
#             print(f'the window we are multiplying our frame by is {w(L)}')
#             print(f'xm is now {xm}, and signal is {signal[m:]}')

            xm *= w(L)
#             print(f'xm is now {xm}, and signal is {signal[m:]}')
        xm = np.array([xm]).T
        xms = np.hstack([xms,xm])
    return xms[:, 1:]  # discard first column of all zeros


def get_stft(windowed_signal, nfft=None):
    """
    :param windowed_signal: windowed signal matrix
    :param nfft: the size of the dft.
    
    :return 2-D matrix where each column is a single dft frame 
    """
    if nfft is None:
        nfft = windowed_signal.shape[0]
    ms = windowed_signal.shape[1]
    dfts = np.array([np.zeros(nfft)]).T
    for m in range(ms):
        xm = windowed_signal[:, m]
        freq_window = np.array([np.fft.fft(xm, nfft)]).T
        dfts = np.hstack([dfts, freq_window])
    return dfts[:, 1:]


def get_istft(stft, R):
    """ 
    Performs Overlap-Add reconstruction of original signal
    """
    nfft = stft.shape[0]  # size of the dft
    num_frames = stft.shape[1]  # number of dft windows
    signal = np.zeros((R * (num_frames - 1)) + nfft)
    for m in range(num_frames):
        idx = m*R
        windowed_signal = (np.fft.ifft(stft[:, m])).real
        signal[(idx):idx+nfft] += windowed_signal
    return signal * (1/nfft)


def plot_spectrogram(stft, fs, R, idx, title="", colorbar=False):    
    """
    plot spectrogram of stft
    """
    plt.figure(figsize=[10,5])
    L = stft.shape[0]
    num_frames = stft.shape[1]
    # we only look at DFT freq up to nyquist limit fs/2, and normalize out imag components
    stft_db = 20*np.log10(np.abs(stft[:L//2, :])*2)
    plt_spec = plt.imshow(
        stft_db,
        origin='lower'
    )
    
    ## create ylim
    num_yticks = 10
    ks = np.linspace(0, L/2, num_yticks)
    ks_hz = ks * fs // (L)
    plt.yticks(ks,ks_hz)
    plt.ylabel("Frequency (Hz)")

    ## create xlim
    num_xticks = 10
    ts_spec = np.linspace(0, num_frames, num_xticks)
    ts_spec_sec  = ["{:4.2f}".format(i) for i in np.linspace(0, (R*num_frames)/fs, num_xticks)]
    plt.xticks(ts_spec,ts_spec_sec)
    plt.xlabel("Time (sec)")

    plt.title(f"{title} L={L} hopsize={R}, fs={fs} Spectrogram.shape={stft.shape}")
    if colorbar:
        plt.colorbar(None, use_gridspec=True)
    plt.show()
    return(plt_spec)



#==============LPC Functions===============#

def gen_autocorrelates(x, M):
    """
    returns [r_0, r_1, ..., r_M]
    """
    rx = np.zeros(M+1)
    for i in range(M+1):
        rx[i] = np.dot(x[:len(x)-i],x[i:])
    return rx


def gen_toeplitz(rx, M):
    covmatrix = np.zeros((M,M))
    for i in range(0,M):
        for j in range(0,M):
            covmatrix[i, j] = rx[np.abs(i-j)]
    return covmatrix


def gen_lp_coeffs(x, M):
    """
    returns a_0, a_1, ... a_M for a signal x
    """
    rx = gen_autocorrelates(x, M)
    toeplitz = gen_toeplitz(rx, M)          
    coeffs, _, _, _ = np.linalg.lstsq(
        toeplitz,
        -1*rx[1:], 
        rcond=None
    )
    return np.concatenate(([1], coeffs))


def gen_lpc_spec_envs(windowed_modulator, M, nfft):
    """
    :param windowed_modulator: matrix where each column is a windowed signal
    :param M: order of linear predictor
    :param nfft: dft size

    Returns a matrix of spectral envelopes, where column m is spectral envelope for m'th signal frame
    """
    num_frames = windowed_modulator.shape[1]
    spec_envs = np.array([np.zeros(nfft)]).T
    for m in range(num_frames):
        xm = windowed_modulator[:, m]  # get mth column
        coeffs = gen_lp_coeffs(xm, M)
        spec_env = np.array([1/(np.abs(np.fft.fft(coeffs, nfft)))]).T
        spec_envs = np.hstack([
            spec_envs, 
            spec_env
        ])
    return spec_envs[:, 1:]
    

#==============Cross-Synthesis===============#

def cross_synthesize(fs, carrier, modulator, L, R, M, flatten=False, w=None, plot=False):
    """
    :param fs: sample rate
    :param carrier: carrier signal in time
    :param modulator: modulator signal in time
    :param L: window size
    :param R: hop size
    :param M: number of coefficients
    :param flatten: if true, divide carrier spectrum by its own envelope
    :param w: window coefficients
    :param plot: if true, will generate spectrograms
    
    returns stft of cross-synthesized signal, and cross-synthesized audio signal
    """
    # to prevent time-domain aliasing, make nfft size double the window size
    nfft = L*2  # convolution length of two length-L signals, the whitening filter and windowed signal
    
    windowed_carrier = get_windowed_signal(carrier, L, R, w=w)
    windowed_modulator = get_windowed_signal(modulator, L, R, w=w)
    
    carrier_stft = get_stft(windowed_carrier, nfft)
    modulator_stft = get_stft(windowed_modulator, nfft)
    if plot:
        plot_spectrogram(carrier_stft, fs, R, 1, title="original carrier")
        plot_spectrogram(modulator_stft, fs, R, 2, title="modulator")
    
    # Optional: divide spectrum of carrier frame by its own envelope 
    if flatten:
        carrier_spec_envs = gen_lpc_spec_envs(windowed_carrier, M, nfft)
        carrier_stft = carrier_stft / carrier_spec_envs
        if plot: 
            plot_spectrogram(carrier_stft, fs, R, 3, title="flattened carrier")
    
    # Multiply carrier spectral frame by modulator spectral envelops
    modulator_spec_envs = gen_lpc_spec_envs(windowed_modulator, M, nfft)
    cross_synth_stft = carrier_stft * modulator_spec_envs
    if plot: 
        plot_spectrogram(cross_synth_stft, fs, R, 4, title="cross-synthesized carrier")
    return cross_synth_stft, get_istft(cross_synth_stft, R)


if __name__ == "__main__":
    carr_files = os.listdir("./carriers")
    mod_files = os.listdir("./modulators")
#==============Command Line Arguments===============#
    parser = argparse.ArgumentParser()
    parser.add_argument("M", help="number of linear coefficients for LPC (from 1-10)", choices=[f'{n}' for n in range(1,10)])
    parser.add_argument("L", help="window size (choose a power of 2)")
    parser.add_argument("--w", help="choose a window function (b for bartlett--triangle, h for hanning)", choices=['b', 'h'])
    parser.add_argument("--flatten", required=False, help="flattens the carrier spectrum by its envelope", action='store_true')
    parser.add_argument("--plot", required=False, help="show the spectrograms of the carrier, modulator and cross-synthesized signal.", action='store_true')
    args = parser.parse_args()

    print('The carrier files are:')
    for f in range(len(carr_files)):
        if carr_files[f] != '.DS_Store':    
            print(f'({f}) : {carr_files[f]}')
    carrier = carr_files[int(input('Enter the number for the file you would like to choose: '))]
    print('The modulator files are:')
    for f in range(len(mod_files)):
        if mod_files[f] != '.DS_Store':    
            print(f'({f}) : {mod_files[f]}')
    modulator = mod_files[int(input('Enter the number for the file you would like to choose: '))]
    # grabbing arguments from command line
    # modulator = args.modulator_file
    # carrier = args.carrier_file
    M = int(args.M)

    L = int(args.L )
    R = L
    w = args.w 
    ws = {'b': get_bartlett, 'h': get_hanning}
    w_fn = None
    flatten = args.flatten
    plot = args.plot

    if w == 'b' or w == 'h':
        R = L // 2
        w_fn = ws[w]
    fs_mod, modulator = wavfile.read(f'modulators/{modulator}')
    fs_car, carrier = wavfile.read(f'carriers/{carrier}')
    print(w_fn)
    # make sure files are the same sample rate and length

    fs = min(fs_mod, fs_car)
    modulator = signal.resample(modulator, fs)
    carrier = signal.resample(carrier, fs)

    carrier = carrier[:min(len(modulator), len(carrier))]
    modulator = modulator[:min(len(modulator), len(carrier))]

    # Normalize signals
    carrier = carrier/(np.max(carrier))
    modulator = modulator/(np.max(modulator))


    # Cross-synthesize using rectangular window with 0 overlap
    cross_synth_stft, cross_synth_audio = \
        cross_synthesize(
            fs,
            carrier, 
            modulator, 
            L, 
            R, 
            M, 
            flatten=flatten,
            w=w_fn,
            plot=plot
        )
    sd.play(cross_synth_audio, fs // 2)
    sd.wait()
