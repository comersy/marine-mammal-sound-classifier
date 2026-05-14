import os
import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset
from transformers import ASTFeatureExtractor
import librosa

extractor = ASTFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")

class MarineDatasetAST(Dataset):
    def __init__(self, root_dir):
        self.samples = []
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

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
        audio = librosa.resample(audio.astype(np.float32), orig_sr=sr, target_sr=16000)
        inputs = extractor(audio, sampling_rate=16000, return_tensors="pt")
        return inputs["input_values"].squeeze(0), label