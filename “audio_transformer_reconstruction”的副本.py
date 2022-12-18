# -*- coding: utf-8 -*-
"""“audio transformer reconstruction”的副本

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1UNYsI4aa2PdxU77yuIz-SVGaR3056wqh
"""

# ! pip install git+https://github.com/openai/whisper.git
# ! pip install jiwer

from locale import normalize
import os
import numpy as np
import torch.nn.functional as F
try:
    import tensorflow  # required in Colab to avoid protobuf compatibility issues
except ImportError:
    pass

import torch
import pandas as pd
import whisper

import torchaudio
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from tqdm.notebook import tqdm

from operator import ne
import torch
import matplotlib.pyplot as plt
import torchaudio
import librosa
import matplotlib.pyplot as plt
import gc
import math

class SignalProducer:
    @staticmethod
    def produce_batch(batch_size: int,
              seconds: float, 
              points_per_second: int,
              stable_slot: float, 
              frequency_num: int, 
              frequency_magnitude: float,
              amptitude: float = 10):
        """

        :param batch_size: produce a batch of signals
        :param seconds: length of the signal
        :param points_per_second: seconds * points_per_second = number of points in every signal
        :param stable_slot: in a slot, frequencies do not change
        :param frequency_num: number of frequencies in every slot
        :param frequency_magnitude: magnitude of frequency
        :return: batch_signal_sequence.shape = (batch_size, seconds * points_per_second), frequency_arr.shape = (batch_size, frequency_num, seconds * points_per_second)
        """

        variable_frequency_signal, variable_frequency_arr = \
            SignalProducer.produce_variable_frequency_signal(batch_size=batch_size, seconds=seconds, points_per_second=points_per_second, stable_slot=stable_slot, frequency_num=frequency_num, frequency_magnitude=frequency_magnitude, amptitude=amptitude)
        fixed_frequency_signal, fixed_frequency_arr = \
            SignalProducer.produce_variable_frequency_signal(batch_size=batch_size, seconds=seconds, points_per_second=points_per_second, stable_slot=seconds, frequency_num=1, frequency_magnitude=frequency_magnitude, amptitude=10*amptitude)
        batch_signal_sequence = variable_frequency_signal + fixed_frequency_signal
        frequency_arr = torch.cat(tensors=[variable_frequency_arr, fixed_frequency_arr], dim=1)
        return batch_signal_sequence, frequency_arr
        

    @staticmethod
    def produce_variable_frequency_signal(batch_size: int,
                          seconds: float, 
                          points_per_second: int,
                          stable_slot: float, 
                          frequency_num: int, 
                          frequency_magnitude: float,
                          amptitude: float):
        signal_points_num = int(seconds * points_per_second)
        all_points_num = batch_size * signal_points_num
        all_points_index_list = torch.arange(all_points_num)

        signal_slot_num = int(seconds / stable_slot)
        all_slot_num = batch_size * signal_slot_num
        slot_points_num = stable_slot * points_per_second
        frequency_arr = torch.cat(tensors=[(torch.rand(size=(frequency_num, 1)) * frequency_magnitude).expand((frequency_num, slot_points_num)) for _ in range(all_slot_num)], dim=1)

        all_signal_sequence = torch.sin(2 * torch.pi * frequency_arr * all_points_index_list).sum(dim=0) / frequency_num * amptitude
        batch_signal_sequence = all_signal_sequence.reshape((batch_size, -1))

        frequency_arr = frequency_arr.reshape((frequency_num, batch_size, -1)).permute((1, 0, 2))

        return batch_signal_sequence, frequency_arr


class Plot:
    @staticmethod
    def plot_spectrogram(specgram, title=None, ylabel="freq_bin"):
        fig, axs = plt.subplots()
        axs.set_title(title or "Spectrogram (db)")
        axs.set_ylabel(ylabel)
        axs.set_xlabel("frame")
        im = axs.imshow(librosa.power_to_db(specgram), origin="lower", aspect="auto")
        fig.colorbar(im, ax=axs)
        return fig

    @staticmethod
    def plot_signal(signal):
        fig, ax = plt.subplots()
        ax.plot(signal.detach().cpu().numpy())
        ax.set_xlabel('t')
        ax.set_ylabel('A')
        ax.set_title('signal')
        return fig

    @staticmethod
    def plot_loss(loss_list):
        fig, ax = plt.subplots()
        ax.plot(loss_list)
        ax.set_xlabel('weight updates')
        ax.set_ylabel('loss')
        ax.set_title('loss vs weight updates')
        return fig


class RandomGeneratedSignals(torch.utils.data.Dataset):
    """
    Torch dataset to generate signals
    """
    def __init__(self, size=10000, points_per_second=198, device=DEVICE):
        split_batch_num = 5
        batch_signal, frequency_arr = SignalProducer.produce_batch(batch_size=size//split_batch_num, seconds=20, points_per_second=points_per_second, stable_slot=1, frequency_num=10, frequency_magnitude=100)
        for _ in range(split_batch_num - 1):
            batch_signal_, frequency_arr_ = SignalProducer.produce_batch(batch_size=size//split_batch_num, seconds=20, points_per_second=points_per_second, stable_slot=1, frequency_num=10, frequency_magnitude=100)
            batch_signal = torch.cat((batch_signal_, batch_signal), dim=0)
            frequency_arr = torch.cat((frequency_arr_, frequency_arr), dim=0)
        spectrogram = torchaudio.transforms.Spectrogram(n_fft=198, return_complex=True, normalized="frame_length")
        spec = spectrogram(batch_signal)
        self.dataset = {idx:(batch_signal[idx], spec[idx], frequency_arr[idx]) for idx in range(size)}
        self.device = device

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        audio, spec, frequency_arr = self.dataset.get(item)
        audio = torch.mean(audio.view(-1, 10), axis=1)
        # spec = F.normalize(spec)
        
        return (spec, audio, frequency_arr)

dataset = RandomGeneratedSignals(40000)
loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True)

s, a, frequency_arr = next(iter(loader))
print(s.shape, a.shape, frequency_arr.shape)
fig = Plot.plot_signal(signal=a[0])
fig.savefig('./signal_example.png')
fig = Plot.plot_spectrogram(s[0])
fig.savefig('./spectrogram_example.png')

s[0].min()

# def sinusoids(length, channels, max_timescale=10000, frequency_magnitude=1000 * 2 * torch.pi):
#     """Returns sinusoids for positional embedding"""
#     den = torch.exp(- torch.arange(0, channels, 2)* math.log(max_timescale) / channels)
#     pos = torch.arange(0, length).reshape(length, 1)
#     pos_embedding = torch.zeros((length, channels))
#     pos_embedding[:, 0::2] = torch.sin(frequency_magnitude * pos * den)
#     pos_embedding[:, 1::2] = torch.cos(frequency_magnitude * pos * den)
#     # pos_embedding = pos_embedding.unsqueeze(-2)
#     return pos_embedding

# P = sinusoids(40, 100)
# # P = getPositionEncoding(40, 100)
# cax = plt.matshow(P)
# plt.gcf().colorbar(cax)
# plt.show()

from dataclasses import dataclass
from typing import Dict
from typing import Iterable, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch import nn

@dataclass
class ModelDimensions:
    n_mels: int
    n_audio_ctx: int
    n_audio_state: int
    n_audio_head: int
    n_audio_layer: int
    n_vocab: int
    n_text_ctx: int
    n_text_state: int
    n_text_head: int
    n_text_layer: int


class LayerNorm(nn.LayerNorm):
    def forward(self, x: Tensor) -> Tensor:
        return super().forward(x.float()).type(x.dtype)


class Linear(nn.Linear):
    def forward(self, x: Tensor) -> Tensor:
        return F.linear(
            x, self.weight.to(x.dtype), None if self.bias is None else self.bias.to(x.dtype)
        )


class Conv1d(nn.Conv1d):
    def _conv_forward(self, x: Tensor, weight: Tensor, bias: Optional[Tensor]) -> Tensor:
        return super()._conv_forward(
            x, weight.to(x.dtype), None if bias is None else bias.to(x.dtype)
        )

# def sinusoids(length, channels, max_timescale=10, frequency_magnitude=1000 * 2 * torch.pi):
#     """Returns sinusoids for positional embedding"""
#     assert channels % 2 == 0
#     log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
#     inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2))
#     scaled_time = torch.arange(length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
#     return torch.cat([torch.sin(frequency_magnitude * scaled_time), torch.cos(frequency_magnitude * scaled_time)], dim=1)

def sinusoids(length, channels, max_timescale=10000, frequency_magnitude=1000 * 2 * torch.pi):
    """Returns sinusoids for positional embedding"""
    den = torch.exp(- torch.arange(0, channels, 2)* math.log(max_timescale) / channels)
    pos = torch.arange(0, length).reshape(length, 1)
    pos_embedding = torch.zeros((length, channels))
    pos_embedding[:, 0::2] = torch.sin(frequency_magnitude * pos * den)
    pos_embedding[:, 1::2] = torch.cos(frequency_magnitude * pos * den)
    # pos_embedding = pos_embedding.unsqueeze(-2)
    return pos_embedding

for A in np.array([1, 10, 100, 200]) * 2 * np.pi:
    # fig, ax = plt.subplots()
    P = sinusoids(length=41, channels=100, max_timescale=10000, frequency_magnitude=A)
    # cax = ax.matshow(P.T)
    # plt.gcf().colorbar(cax)
    fig = Plot.plot_spectrogram(specgram=P.T, title='positional embedding')
    fig.savefig('./positional_embedding' + str(A) + '.png')

    # fig.ax = plt.subplots()
    # cax = ax.matshow(P.T + s[0])
    # plt.gcf().colorbar(cax)
    fig = Plot.plot_spectrogram(specgram=P.T + s[0], title='positional embedding + spectrogram')
    fig.savefig('./positional_embedding_and_spectrogram' + str(A) + '.png')



class MultiHeadAttention(nn.Module):
    def __init__(self, n_state: int, n_head: int):
        super().__init__()
        self.n_head = n_head
        self.query = Linear(n_state, n_state)
        self.key = Linear(n_state, n_state, bias=False)
        self.value = Linear(n_state, n_state)
        self.out = Linear(n_state, n_state)

    def forward(
        self,
        x: Tensor,
        xa: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        kv_cache: Optional[dict] = None,
    ):
        q = self.query(x)

        if kv_cache is None or xa is None or self.key not in kv_cache:
            # hooks, if installed (i.e. kv_cache is not None), will prepend the cached kv tensors;
            # otherwise, perform key/value projections for self- or cross-attention as usual.
            k = self.key(x if xa is None else xa)
            v = self.value(x if xa is None else xa)
        else:
            # for cross-attention, calculate keys and values once and reuse in subsequent calls.
            k = kv_cache[self.key]
            v = kv_cache[self.value]

        wv = self.qkv_attention(q, k, v, mask)
        return self.out(wv)

    def qkv_attention(self, q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor] = None):
        n_batch, n_ctx, n_state = q.shape
        scale = (n_state // self.n_head) ** -0.25
        q = q.view(*q.shape[:2], self.n_head, -1).permute(0, 2, 1, 3) * scale
        k = k.view(*k.shape[:2], self.n_head, -1).permute(0, 2, 3, 1) * scale
        v = v.view(*v.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)

        qk = q @ k
        if mask is not None:
            qk = qk + mask[:n_ctx, :n_ctx]

        w = F.softmax(qk.float(), dim=-1).to(q.dtype)
        return (w @ v).permute(0, 2, 1, 3).flatten(start_dim=2)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, n_state: int, n_head: int, cross_attention: bool = False):
        super().__init__()

        self.attn = MultiHeadAttention(n_state, n_head)
        self.attn_ln = LayerNorm(n_state)

        self.cross_attn = MultiHeadAttention(n_state, n_head) if cross_attention else None
        self.cross_attn_ln = LayerNorm(n_state) if cross_attention else None

        n_mlp = n_state * 4
        self.mlp = nn.Sequential(Linear(n_state, n_mlp), nn.GELU(), Linear(n_mlp, n_state))
        self.mlp_ln = LayerNorm(n_state)

    def forward(
        self,
        x: Tensor,
        xa: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        kv_cache: Optional[dict] = None,
    ):
        x = x + self.attn(self.attn_ln(x), mask=mask, kv_cache=kv_cache)
        if self.cross_attn:
            x = x + self.cross_attn(self.cross_attn_ln(x), xa, kv_cache=kv_cache)
        x = x + self.mlp(self.mlp_ln(x))
        return x


class AudioEncoder(nn.Module):
    def __init__(self, n_mels: int, n_ctx: int, n_state: int, n_head: int, n_layer: int, sinusoids_max_timescale: int = 10000, sinusoids_frequency_magnitude: float = 1000 * 2 * torch.pi):
        super().__init__()
        # self.conv1 = Conv1d(n_mels, n_state, kernel_size=3, padding=1)
        # self.conv2 = Conv1d(n_state, n_state, kernel_size=3, stride=2, padding=1)
        self.register_buffer("positional_embedding", sinusoids(n_ctx, n_state, max_timescale=sinusoids_max_timescale, frequency_magnitude=sinusoids_frequency_magnitude))

        self.blocks: Iterable[ResidualAttentionBlock] = nn.ModuleList(
            [ResidualAttentionBlock(n_state, n_head) for _ in range(n_layer)]
        )
        self.ln_post = LayerNorm(n_state)

    def forward(self, x: Tensor):
        """
        x : torch.Tensor, shape = (batch_size, n_mels, n_ctx)
            the mel spectrogram of the audio
        """
        # x = F.gelu(self.conv1(x))
        # x = F.gelu(self.conv2(x))
        x = x.permute(0, 2, 1)
        # print(x.shape, self.positional_embedding.shape)

        assert x.shape[1:] == self.positional_embedding.shape, "incorrect audio shape"
        x = (x + self.positional_embedding).to(x.dtype)

        for block in self.blocks:
            x = block(x)

        x = self.ln_post(x)
        return x


class ReconstructionDecoder(nn.Module):
    def __init__(self, n_vocab: int, n_ctx: int, n_state: int, n_head: int, n_layer: int):
        super().__init__()
        self.expand_mlp = Linear(n_vocab, n_state)
        self.positional_embedding = nn.Parameter(torch.empty(n_ctx, n_state))
        self.blocks: Iterable[ResidualAttentionBlock] = nn.ModuleList(
            [ResidualAttentionBlock(n_state, n_head, cross_attention=True) for _ in range(n_layer)]
        )
        self.ln = LayerNorm(n_state)
        mask = torch.empty(n_ctx, n_ctx).fill_(-np.inf).triu_(1)
        self.register_buffer("mask", mask, persistent=False)
        self.out_mlp = nn.Sequential(Linear(n_state, n_state), nn.GELU(), Linear(n_state, 1))
        # self.out_mlp = Linear(n_state, 1)

    def forward(self, x: Tensor, xa: Tensor, kv_cache: Optional[dict] = None):
        """
        x : torch.LongTensor, shape = (batch_size, <= n_ctx)
            the text tokens
        xa : torch.Tensor, shape = (batch_size, n_mels, n_audio_ctx)
            the encoded audio features to be attended on
        """
        # print("original x", x.shape)
        x = torch.unsqueeze(x, dim=-1)
        # print("unsequz x", x.shape)
        # print("expand", self.expand_mlp(x).shape)
        x = self.expand_mlp(x) + self.positional_embedding[:x.shape[1]].repeat((x.shape[0], 1, 1))
        # print("x and pos", x.shape, self.positional_embedding[:x.shape[1]].repeat((x.shape[0], 1, 1)).shape)
        x = x.to(xa.dtype)

        for block in self.blocks:
            x = block(x, xa, mask=self.mask, kv_cache=kv_cache)

        x = self.ln(x)
        # print("x after layer norm", x.shape)

        out = self.out_mlp(x).squeeze_()
        # print("output", out.shape)

        return out
    


# class ReconstructionDecoder(nn.Module):
#     def __init__(self, n_ctx: int, n_audio_ctx, n_audio_state: int):
#         super().__init__()
#         self.mlp = nn.Sequential(Linear(n_audio_ctx*n_audio_state, n_ctx),
#                                         nn.GELU(),
#                                         Linear(n_ctx, n_ctx),
#                                         nn.GELU(),
#                                         Linear(n_ctx, n_ctx))


#     def forward(self, xa):
#         return self.mlp(xa.view(xa.shape[0], -1))


class Replicator(nn.Module):
    def __init__(self, dims: ModelDimensions, sinusoids_max_timescale: int = 10000, sinusoids_frequency_magnitude: float = 1000 * 2 * torch.pi):
        super().__init__()
        self.dims = dims
        self.encoder = AudioEncoder(
            self.dims.n_mels,
            self.dims.n_audio_ctx,
            self.dims.n_audio_state,
            self.dims.n_audio_head,
            self.dims.n_audio_layer,
            sinusoids_max_timescale = sinusoids_max_timescale, 
            sinusoids_frequency_magnitude = sinusoids_frequency_magnitude
        )
        self.decoder = ReconstructionDecoder(
            self.dims.n_vocab,
            self.dims.n_text_ctx,
            # self.dims.n_audio_ctx,
            # self.dims.n_audio_state,
            self.dims.n_text_state,
            self.dims.n_text_head,
            self.dims.n_text_layer,
        )

    def embed_audio(self, mel: torch.Tensor):
        return self.encoder(mel)

    # def logits(self, tokens: torch.Tensor, audio_features: torch.Tensor):
    #     return self.decoder(audio_features)

    # def forward(self, mel: torch.Tensor, tokens: torch.Tensor) -> Dict[str, torch.Tensor]:
    #     return self.decoder(self.encoder(mel))


    def logits(self, tokens: torch.Tensor, audio_features: torch.Tensor):
        return self.decoder(tokens, audio_features)

    def forward(self, mel: torch.Tensor, tokens: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.decoder(tokens, self.encoder(mel))


    @property
    def device(self):
        return next(self.parameters()).device


dims = ModelDimensions(100, 41, 100, 4, 4, 1, 400, 100, 4, 2)


def train_epoch(model, optimizer, loss_fn, loader):
    model.to(DEVICE)
    model.train()
    losses = 0
    iters = 0
    
    for src, tgt, frequency_arr in loader:
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)
        frequency_arr = frequency_arr.to(DEVICE)
        
        tokens = torch.cat((torch.zeros((tgt.shape[0], 1)).to(DEVICE), tgt[:, :-1]), dim=1)
        # tokens = tgt
        outs = model(src, tokens)
        outs = outs
        # print(tokens.shape,  outs.shape, "___")
        optimizer.zero_grad()

        loss = loss_fn(tgt, outs)
        loss.backward()

        optimizer.step()
        losses += loss.item()
        iters += 1
        # print(loss.item())
        # print(outs.shape)
        if iters%200 == 0:
            print(loss.item())
    # del src, tgt

    return losses / iters

def loss_fn(y_pred, target):
    loss = 2 * (y_pred - target).abs() / (y_pred.abs() + target.abs() + 1e-8)
    return loss.mean()

def validate(dataloader, model):
    model.eval()
    losses = 0
    iters = 0
    for src, tgt, frequency_arr in dataloader:
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)
        tokens = torch.cat((torch.zeros((tgt.shape[0], 1)).to(DEVICE), tgt[:, :-1]), dim=1)

        outs = model(src, tokens)

        loss = loss_fn(tgt, outs)
        iters += 1
        losses += loss.detach().cpu()

    model.train()
    return losses / iters

torch.manual_seed(2022)

model_dictionary = {}
idx = 0
validation_dataset = RandomGeneratedSignals(10000)
validation_dataloader = torch.utils.data.DataLoader(validation_dataset, batch_size=32, shuffle=True)

for sinusoids_max_timescale, sinusoids_frequency_magnitude in zip([10000, 10000, 10000, 10000], np.array([1, 10, 100, 200]) * 2 * np.pi):
  print(idx)
  print(sinusoids_max_timescale)
  print(sinusoids_frequency_magnitude)
  model_dictionary[idx] = {}
  model_dictionary[idx]["sinusoids_max_timescale"] = sinusoids_max_timescale
  model_dictionary[idx]["sinusoids_frequency_magnitude"] = sinusoids_frequency_magnitude

  model = Replicator(dims, sinusoids_max_timescale=sinusoids_max_timescale, sinusoids_frequency_magnitude=sinusoids_frequency_magnitude)
  model.to(DEVICE)
  model_dictionary[idx]["model"] = model
  
  optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.98), eps=1e-9)
  scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-4, last_epoch=-1, verbose=False)

  train_loss_list = []
  validation_loss_list = []

  for epoch in range(3):
    model.train()
    losses = 0
    iters = 0
    
    for src, tgt, frequency_arr in loader:
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)
        frequency_arr = frequency_arr.to(DEVICE)
        
        tokens = torch.cat((torch.zeros((tgt.shape[0], 1)).to(DEVICE), tgt[:, :-1]), dim=1)
        
        outs = model(src, tokens)
        
        # print(tokens.shape,  outs.shape, "___")
        optimizer.zero_grad()

        loss = loss_fn(tgt, outs)
        loss.backward()

        optimizer.step()
        losses += loss.item()
        iters += 1

        train_loss_list.append(loss.item())
        scheduler.step()


    if math.isnan(losses):
        print('reinit')
        model = Replicator(dims, sinusoids_max_timescale=sinusoids_max_timescale,
                           sinusoids_frequency_magnitude=sinusoids_frequency_magnitude)
        model.to(DEVICE)
        model_dictionary[idx]["model"] = model

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, betas=(0.9, 0.98), eps=1e-9)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6, last_epoch=-1, verbose=False)

        train_loss_list = []


    print(losses / iters)
    validation_loss = validate(dataloader=validation_dataloader, model=model)
    validation_loss_list.append(validation_loss)


  model_dictionary[idx]["train_loss_list"] = train_loss_list
  model_dictionary[idx]["validation_loss_list"] = validation_loss_list

  idx += 1

fig, ax = plt.subplots()
for key in model_dictionary.keys():
    A = model_dictionary[key]["sinusoids_frequency_magnitude"]
    ax.plot(model_dictionary[key]["train_loss_list"], label=f"A: {A}")
ax.set_xlabel("Epoch")
ax.set_ylabel("loss")
ax.set_title("training loss vs epoch")
fig.legend()
fig.savefig('./train_loss.png')

fig, ax = plt.subplots()
for key in model_dictionary.keys():
    A = model_dictionary[key]["sinusoids_frequency_magnitude"]
    ax.plot(model_dictionary[key]["validation_loss_list"], label=f"A: {A}")
ax.set_xlabel("Epoch")
ax.set_ylabel("loss")
ax.set_title("validation loss vs epoch")
fig.legend()
fig.savefig('./validation_loss.png')

validation_dataloader = torch.utils.data.DataLoader(validation_dataset, batch_size=1, shuffle=True)

# torch.cuda.memory_summary(device=torch.cuda.current_device(), abbreviated=False)
# del dataset
# del loader
# del s
# gc.collect()

# def size_of_model(model):
#   # print(model)
#   param_size = 0
#   for param in model.parameters():
#       param_size += param.nelement() * param.element_size()
#   buffer_size = 0
#   for buffer in model.buffers():
#       buffer_size += buffer.nelement() * buffer.element_size()

#   size_all_mb = (param_size + buffer_size) / 1024**2
#   print('model size: {:.3f}MB'.format(size_all_mb))

# size_of_model(model=model_dictionary[0]["model"])

for key in model_dictionary.keys():
  model_dictionary[key]["fixed_frequency_in_validation"] = []
  model_dictionary[key]["loss_in_validation"] = []
  model_dictionary[key]["model"].eval()
  
for src, tgt, frequency_arr in validation_dataloader:
  src = src.to(DEVICE)
  tgt = tgt.to(DEVICE)
  tokens = torch.cat((torch.zeros((tgt.shape[0], 1)).to(DEVICE), tgt[:, :-1]), dim=1)

  for key in model_dictionary.keys():
    model = model_dictionary[key]["model"]
    
    outs = model(src, tokens)

    loss = loss_fn(tgt, outs)

    model_dictionary[key]["fixed_frequency_in_validation"].append(frequency_arr[0, -1, 0].detach().cpu().numpy())
    model_dictionary[key]["loss_in_validation"].append(loss.detach().cpu())

def location_of_positional_embedding_given_frequency(fixed_frequency):
  return fixed_frequency

def frequency_of_positional_embedding_at_location(max_timescale, frequency_magnitude, location, dmodel):
  frequency = frequency_magnitude * 1 / np.power(max_timescale, np.array(location) / dmodel)
  return frequency

for key in model_dictionary.keys():
  fixed_frequency = model_dictionary[key]["fixed_frequency_in_validation"]
  loc = location_of_positional_embedding_given_frequency(fixed_frequency=fixed_frequency)

  max_timescale = model_dictionary[key]["sinusoids_max_timescale"]
  frequency_magnitude = model_dictionary[key]["sinusoids_frequency_magnitude"]
  positional_embedding_frequency = frequency_of_positional_embedding_at_location(max_timescale=max_timescale, frequency_magnitude=frequency_magnitude, location=loc, dmodel=201)

  model_dictionary[key]["positional_embedding_frequency_in_validation"] = positional_embedding_frequency

fig = plt.figure(figsize = (16, 9))
ax = plt.axes(projection ="3d")
for key in model_dictionary.keys():
  fixed_frequency = model_dictionary[key]["fixed_frequency_in_validation"]
  positional_embedding_frequency = model_dictionary[key]["positional_embedding_frequency_in_validation"]
  validation_loss = model_dictionary[key]["loss_in_validation"]

  fixed_frequency = np.array(fixed_frequency).reshape((-1, 1))
  positional_embedding_frequency = np.array(positional_embedding_frequency).reshape((-1, 1))
  validation_loss = np.array(validation_loss).reshape((-1, 1))
  # print(fixed_frequency.shape, positional_embedding_frequency.shape, validation_loss.shape)

  A = model_dictionary[key]["sinusoids_frequency_magnitude"]
  ax.scatter(fixed_frequency, positional_embedding_frequency, validation_loss, label=f"A: {A}")

ax.set_xlabel('fixed signal frequency')
ax.set_ylabel('positional embedding_frequency')
ax.set_zlabel('validation loss')
ax.set_title('relationship between positional embedding frequency and signal frequency')
fig.legend()
fig.savefig('./relationship_between_positional_embedding_frequency_and_signal_frequency.png')


for key in model_dictionary.keys():
  print(model_dictionary[key]["sinusoids_max_timescale"], model_dictionary[key]["sinusoids_frequency_magnitude"], np.mean(model_dictionary[key]["loss_in_validation"]))


new_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
mel, audio, frequency_arr = next(iter(new_loader))

fig, ax = plt.subplots(5, figsize=(12, 12))
ax[0].plot(audio[0].cpu().detach().numpy())
ax[0].set_title('True signal')
s = torchaudio.transforms.Spectrogram(n_fft=198)
fig_spec = Plot.plot_spectrogram(s(audio[0].cpu()))
fig_spec.savefig('./spectrogram_truth.png')
idx = 1
for key in model_dictionary.keys():
  r = model_dictionary[key]["model"]
  r.eval()
  r.to(DEVICE)

  embeded_audio = r.embed_audio(mel.to(DEVICE))
  tokens = audio[:, :1].to(DEVICE)

  with torch.no_grad():
      for _ in range(audio.shape[1]-1):
          logits = r.logits(tokens, embeded_audio).unsqueeze(-1)
          tokens = torch.cat([tokens, logits[-1].view(tokens.shape[0], -1)], dim=-1)
      logits = r(mel.to(DEVICE), torch.cat((torch.zeros((audio.shape[0], 1)), audio[:, :-1]), dim=1).to(DEVICE))

  ax[idx].plot(logits.cpu().detach().numpy())
  ax[idx].set_title(f'A: {model_dictionary[key]["sinusoids_frequency_magnitude"]}')

  fig_spec = Plot.plot_spectrogram(s(logits.cpu()))
  fig_spec.savefig('./spectrogram' + str(model_dictionary[key]["sinusoids_frequency_magnitude"]) + '.png')

  idx += 1
fig.savefig('./signal_reconstruction.png')

for key in model_dictionary.keys():
  fixed_frequency = model_dictionary[key]["fixed_frequency_in_validation"]
  positional_embedding_frequency = model_dictionary[key]["positional_embedding_frequency_in_validation"]
  validation_loss = model_dictionary[key]["loss_in_validation"]

  fixed_frequency = np.array(fixed_frequency).reshape((-1, 1))
  positional_embedding_frequency = np.array(positional_embedding_frequency).reshape((-1, 1))
  validation_loss = np.array(validation_loss).reshape((-1, 1))
  break


plt.show()