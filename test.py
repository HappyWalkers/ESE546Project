import torch
import torchaudio
import librosa
from signalProducer import Plot

x = torch.arange(1000)
y = torch.sin(x)

spectrogram = torchaudio.transforms.Spectrogram(n_fft=100)
spec = spectrogram(y)

Plot.plot_signal(signal=y)
Plot.plot_spectrogram(spec)