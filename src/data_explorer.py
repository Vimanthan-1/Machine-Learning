import matplotlib.pyplot as plt
import numpy as np
import scipy.io.wavfile as wavf
from scipy import signal
import os

def plot_waveform(audio_path, title, save_path):
    sr, y = wavf.read(audio_path)
    time = np.linspace(0, len(y) / sr, num=len(y))
    plt.figure(figsize=(10, 3))
    plt.plot(time, y, alpha=0.6, color='blue')
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_spectrogram(audio_path, title, save_path):
    sr, y = wavf.read(audio_path)
    f, t, Sxx = signal.spectrogram(y, sr)
    plt.figure(figsize=(10, 4))
    plt.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-10), shading='gouraud')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

if __name__ == "__main__":
    os.makedirs('outputs', exist_ok=True)
    print("Generating Waveform and Spectrogram visualizations...")
    plot_waveform('dataset/real/sample_01.wav', 'Waveform - Genuine Human Speech', 'outputs/real_waveform.png')
    plot_waveform('dataset/fake/sample_01.wav', 'Waveform - AI Generated Speech (Deepfake)', 'outputs/fake_waveform.png')
    
    plot_spectrogram('dataset/real/sample_01.wav', 'Spectrogram - Genuine Human Speech', 'outputs/real_spectrogram.png')
    plot_spectrogram('dataset/fake/sample_01.wav', 'Spectrogram - AI Generated Speech (Deepfake)', 'outputs/fake_spectrogram.png')
    print("Visualizations saved to outputs/ directory.")
