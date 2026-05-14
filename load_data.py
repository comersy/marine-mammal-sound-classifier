from datasets import load_dataset, Audio
import soundfile as sf
import io
import os
from collections import defaultdict
import random

random.seed(42)

ds = load_dataset("confit/wmms-parquet")
ds = ds.cast_column("audio", Audio(decode=False))

all_samples = list(ds["train"]) + list(ds["test"])
by_species = defaultdict(list)
for sample in all_samples:
    by_species[sample["species"]].append(sample)

os.makedirs("data", exist_ok=True)

for species, samples in by_species.items():
    random.shuffle(samples)
    split_idx = int(len(samples) * 0.8)
    splits = {"train": samples[:split_idx], "test": samples[split_idx:]}

    for split, split_samples in splits.items():
        species_dir = f"data/{split}/{species}"
        os.makedirs(species_dir, exist_ok=True)
        for sample in split_samples:
            audio_array, sr = sf.read(io.BytesIO(sample["audio"]["bytes"]))
            path = f"{species_dir}/{sample['audio']['path']}"
            sf.write(path, audio_array, sr)

print("Done")