import os
import torch
import numpy as np
import soundfile as sf
from torch.utils.data import Dataset
import torchaudio.transforms as T

class MarineDataset(Dataset):
    def __init__(self, root_dir, n_mels=64, target_length=256):
        self.samples = []
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.target_length = target_length
        self.n_mels = n_mels

        for label in self.classes:
            folder = os.path.join(root_dir, label)
            for fname in os.listdir(folder):
                if fname.endswith(".wav"):
                    self.samples.append((os.path.join(folder, fname), self.class_to_idx[label]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        audio, sr = sf.read(path)

        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        waveform = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)

        mel_transform = T.MelSpectrogram(sample_rate=sr, n_mels=self.n_mels)
        db_transform = T.AmplitudeToDB()
        mel = db_transform(mel_transform(waveform))

        if mel.shape[2] < self.target_length:
            mel = torch.nn.functional.pad(mel, (0, self.target_length - mel.shape[2]))
        else:
            mel = mel[:, :, :self.target_length]

        return mel, label