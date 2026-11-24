import numpy as np
import librosa
import torch
import torch.nn.functional as F
from scipy.signal import butter, sosfiltfilt

SAMPLE_RATE = 44100
TARGET_LENGTH = 220500
N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 128

IMAGENET_MEAN = torch.tensor(
    [0.485, 0.456, 0.406]
).view(1, 3, 1, 1)

IMAGENET_STD = torch.tensor(
    [0.229, 0.224, 0.225]
).view(1, 3, 1, 1)


def standardize_audio(audio):
    audio = np.asarray(audio, dtype=np.float32)

    if len(audio) < TARGET_LENGTH:
        audio = np.pad(
            audio,
            (0, TARGET_LENGTH - len(audio))
        )
    else:
        audio = audio[:TARGET_LENGTH]

    return audio


def audio_crop(audio):
    nonzero = np.flatnonzero(audio != 0)

    if len(nonzero) == 0:
        return np.zeros(TARGET_LENGTH, dtype=np.float32)

    cropped = audio[nonzero[0]:nonzero[-1] + 1]

    if len(cropped) == 0:
        return np.zeros(TARGET_LENGTH, dtype=np.float32)

    repeats = int(np.ceil(TARGET_LENGTH / len(cropped)))
    return np.tile(cropped, repeats)[:TARGET_LENGTH]


def high_pass_filter(audio, cutoff=2048, order=5):
    sos = butter(
        order, cutoff, btype="highpass",
        fs=SAMPLE_RATE, output="sos"
    )
    return sosfiltfilt(sos, audio)


def low_pass_filter(audio, cutoff=512, order=5):
    sos = butter(
        order, cutoff, btype="lowpass",
        fs=SAMPLE_RATE, output="sos"
    )
    return sosfiltfilt(sos, audio)


def apply_condition(audio, condition):
    if condition == "No Filter":
        return audio

    if condition == "Audio Crop":
        return audio_crop(audio)

    if condition == "High Pass":
        return high_pass_filter(audio)

    if condition == "Noise Removal":
        try:
            import noisereduce as nr
        except ImportError as exc:
            raise ImportError(
                "Install noisereduce to use this specialist."
            ) from exc

        return nr.reduce_noise(y=audio, sr=SAMPLE_RATE)

    if condition == "PCEN":
        return audio

    raise ValueError(f"Unsupported condition: {condition}")


def compute_feature(audio, condition):
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        fmin=0,
        fmax=SAMPLE_RATE / 2,
        power=2.0
    )

    if condition == "PCEN":
        feature = librosa.pcen(
            mel,
            sr=SAMPLE_RATE,
            hop_length=HOP_LENGTH,
            gain=0.98,
            bias=2.0,
            power=0.5,
            time_constant=0.4,
            eps=1e-6
        )
    else:
        feature = librosa.power_to_db(
            mel,
            ref=np.max,
            top_db=80.0
        )

    return np.asarray(feature, dtype=np.float32)


def preprocess_audio(audio_path, condition, device):
    audio, _ = librosa.load(
        str(audio_path),
        sr=SAMPLE_RATE,
        mono=True
    )

    audio = standardize_audio(audio)
    audio = apply_condition(audio, condition)
    feature = compute_feature(audio, condition)

    x = torch.from_numpy(feature.copy()).float()

    if condition == "PCEN":
        x_min = x.min()
        x_max = x.max()

        if x_max > x_min:
            x = (x - x_min) / (x_max - x_min)
        else:
            x = torch.zeros_like(x)
    else:
        x = torch.clamp((x + 80.0) / 80.0, 0.0, 1.0)

    x = x[None, None, :, :]
    x = F.interpolate(
        x,
        size=(224, 224),
        mode="bilinear",
        align_corners=False
    )
    x = x.repeat(1, 3, 1, 1)

    mean = IMAGENET_MEAN.to(device)
    std = IMAGENET_STD.to(device)

    x = (x.to(device) - mean) / std
    return x
