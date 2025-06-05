from dataclasses import dataclass
import os
import torch
import torchaudio
import matplotlib.pyplot as plt
import noisereduce
import torchaudio.functional as F
import asyncio
from concurrent.futures import ThreadPoolExecutor

@dataclass
class AudioInfo:
    y: torch.Tensor
    sr: int

audios = "audios"
os.makedirs(audios, exist_ok=True)

executor = ThreadPoolExecutor()

async def run_in_thread(fn, *args, **kwargs):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, lambda: fn(*args, **kwargs))


def normalize_audio(info: AudioInfo) -> AudioInfo:
    y = info.y / info.y.abs().max()
    return AudioInfo(y, info.sr)

def denoise_audio(info: AudioInfo) -> AudioInfo:
    y_np = info.y.numpy() if isinstance(info.y, torch.Tensor) else info.y
    y_denoised = noisereduce.reduce_noise(y_np, sr=info.sr, prop_decrease=1.0, stationary=True)
    y_denoised = y_denoised.mean(axis=0)
    y_tensor = torch.tensor(y_denoised, dtype=torch.float32)
    if y_tensor.ndim == 1:
        y_tensor = y_tensor.unsqueeze(0)
    return AudioInfo(y_tensor, info.sr)

def trimm_audio(info: AudioInfo) -> AudioInfo:
    y = torchaudio.transforms.Vad(info.sr)(info.y.clone().detach())
    return AudioInfo(y, info.sr)

def mono_audio(info: AudioInfo) -> AudioInfo:
    y = torch.mean(info.y, dim=0, keepdim=True)
    return AudioInfo(y, info.sr)

def resample_audio(info: AudioInfo, new_sr: int) -> AudioInfo:
    y = torchaudio.functional.resample(info.y, orig_freq=info.sr, new_freq=new_sr)
    return AudioInfo(y, new_sr)

def equalize_audio(info: AudioInfo) -> AudioInfo:
    y = F.equalizer_biquad(info.y, sample_rate=info.sr, center_freq=1000, gain=10.0, Q=0.707)
    return AudioInfo(y, info.sr)

def visualize_audio(info: AudioInfo):
    fig = plt.figure(figsize=(10, 4), facecolor="#0a0a0a")
    ax = fig.add_subplot(1, 1, 1)

    ax.set_facecolor("#1e1e1e")
    ax.plot(info.y.squeeze().numpy(), color='lime', lw=1)

    ax.set_title("Waveform", color='white')
    ax.set_xlabel("Time", color='white')
    ax.set_ylabel("Amplitude", color='white')
    ax.tick_params(colors='white')

    for spine in ax.spines.values():
        spine.set_color('white')
        spine.set_linewidth(1.5)

    plt.show()

def save_audio(info: AudioInfo, uri="output.wav"):
    if info.y.ndim == 1:
        info.y = info.y.unsqueeze(0)
    torchaudio.save(uri, info.y, info.sr)

def load_audio(
    input_uri,
    output_uri="output.wav",
    normalize=True,
    denoise=True,
    trim=True,
    equalize=True,
    mono=True,
    new_sr=22050,
    visualize=False,
    save=True,
):
    y, sr = torchaudio.load(input_uri)
    info = AudioInfo(y, sr)

    if normalize:
        info = normalize_audio(info)
    if denoise:
        info = denoise_audio(info)
    if trim:
        info = trimm_audio(info)
    if mono:
        info = mono_audio(info)
    if new_sr and info.sr != new_sr:
        info = resample_audio(info, new_sr)
    if equalize:
        info = equalize_audio(info)
    if visualize:
        visualize_audio(info)
    if save:
        save_audio(info, output_uri)

    return info

async def async_pipeline_audio(input_uri, output_uri="uploads/output.wav", **kwargs):
    return await run_in_thread(load_audio, input_uri, output_uri, **kwargs)