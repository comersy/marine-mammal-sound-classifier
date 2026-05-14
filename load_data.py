from datasets import load_dataset, Audio
import soundfile as sf
import io
import os

ds = load_dataset("confit/wmms-parquet")
ds = ds.cast_column("audio", Audio(decode=False))

os.makedirs("data", exist_ok=True)

for split in ["train", "test"]:
    for sample in ds[split]:
        species_dir = f"data/{split}/{sample['species']}"
        os.makedirs(species_dir, exist_ok=True)
        
        audio_array, sr = sf.read(io.BytesIO(sample["audio"]["bytes"]))
        path = f"{species_dir}/{sample['audio']['path']}"
        sf.write(path, audio_array, sr)

print("Done")