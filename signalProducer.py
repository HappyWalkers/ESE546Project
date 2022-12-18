import torch
import matplotlib.pyplot as plt
import torchaudio
import librosa


class SignalProducer:
    @staticmethod
    def produce_batch(batch_size: int,
                      seconds: float, points_per_second: int,
                      stable_slot: float, frequency_num: int, frequency_magnitude: float):
        """

        :param batch_size: produce a batch of signals
        :param seconds: length of the signal
        :param points_per_second: seconds * points_per_second = number of points in every signal
        :param stable_slot: in a slot, frequencies do not change
        :param frequency_num: number of frequencies in every slot
        :param frequency_magnitude: magnitude of frequency
        :return: batch_signal_sequence.shape = (batch_size, seconds * points_per_second), frequency_arr.shape = (batch_size, frequency_num, seconds * points_per_second)
        """
        signal_points_num = int(seconds * points_per_second)
        all_points_index_list = torch.arange(signal_points_num).reshape((1, -1)).repeat((1, batch_size))

        signal_slot_num = int(seconds / stable_slot)
        all_slot_num = batch_size * signal_slot_num
        slot_points_num = stable_slot * points_per_second
        frequency_arr = torch.cat(tensors=[(torch.rand(size=(frequency_num, 1)) * frequency_magnitude).expand((frequency_num, slot_points_num)) for _ in range(all_slot_num)], dim=1)

        all_signal_sequence = torch.sin(2 * torch.pi * frequency_arr * all_points_index_list).sum(dim=0) / frequency_num
        batch_signal_sequence = all_signal_sequence.reshape((batch_size, -1))

        frequency_arr = frequency_arr.reshape((frequency_num, batch_size, -1)).permute((1, 0, 2))

        return batch_signal_sequence, frequency_arr


class Plot:
    @staticmethod
    def plot_spectrogram(specgram, title=None, ylabel="freq_bin"):
        fig, axs = plt.subplots(1, 1)
        axs.set_title(title or "Spectrogram (db)")
        axs.set_ylabel(ylabel)
        axs.set_xlabel("frame")
        im = axs.imshow(librosa.power_to_db(specgram), origin="lower", aspect="auto")
        fig.colorbar(im, ax=axs)
        plt.show()

    @staticmethod
    def plot_signal(signal):
        fig, ax = plt.subplots()
        ax.plot(signal.detach().cpu().numpy())
        ax.set_xlabel('t')
        ax.set_ylabel('A')
        ax.set_title('signal')
        plt.show()


if __name__ == "__main__":
    batch_signal, frequency_arr = SignalProducer.produce_batch(batch_size=32, seconds=10, points_per_second=1000, stable_slot=1, frequency_num=10, frequency_magnitude=100)

    spectrogram = torchaudio.transforms.Spectrogram(n_fft=400)
    spec = spectrogram(batch_signal)

    Plot.plot_signal(signal=batch_signal[0])
    Plot.plot_spectrogram(spec[0])







