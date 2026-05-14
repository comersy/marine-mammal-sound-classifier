import argparse
import os
import torch
import json
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from dataset_cnn import MarineDataset
from dataset_ast import MarineDatasetAST
from model_cnn_scratch import MarineCNN
from model_resnet18 import MarineResNet18
from model_ast import MarineAST

parser = argparse.ArgumentParser()
parser.add_argument("--model", choices=["cnn_scratch", "resnet18", "ast"], required=True)
parser.add_argument("--epochs", type=int, default=50)
args = parser.parse_args()

output_dir = f"outputs/{args.model}"
os.makedirs(output_dir, exist_ok=True)

if args.model == "ast":
    train_dataset = MarineDatasetAST("data/train")
    test_dataset  = MarineDatasetAST("data/test")
else:
    train_dataset = MarineDataset("data/train")
    test_dataset  = MarineDataset("data/test")

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader  = DataLoader(test_dataset,  batch_size=32)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if args.model == "cnn_scratch":
    model = MarineCNN(num_classes=32).to(device)
elif args.model == "resnet18":
    model = MarineResNet18(num_classes=32).to(device)
else:
    model = MarineAST(num_classes=32).to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
N_EPOCHS = args.epochs

for epoch in range(N_EPOCHS):
    model.train()
    train_loss, train_correct, train_total = 0.0, 0, 0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{N_EPOCHS} [train]")
    for x, label in pbar:
        x, label = x.to(device), label.to(device)
        optimizer.zero_grad()
        out  = model(x)
        loss = criterion(out, label)
        loss.backward()
        optimizer.step()
        train_loss    += loss.item() * x.size(0)
        train_correct += (out.argmax(1) == label).sum().item()
        train_total   += x.size(0)
        pbar.set_postfix(loss=loss.item(), acc=train_correct/train_total)

    train_loss /= train_total
    train_acc   = train_correct / train_total

    model.eval()
    val_loss, val_correct, val_total = 0.0, 0, 0
    with torch.no_grad():
        pbar = tqdm(test_loader, desc=f"Epoch {epoch+1}/{N_EPOCHS} [val]  ")
        for x, label in pbar:
            x, label = x.to(device), label.to(device)
            out  = model(x)
            loss = criterion(out, label)
            val_loss    += loss.item() * x.size(0)
            val_correct += (out.argmax(1) == label).sum().item()
            val_total   += x.size(0)
            pbar.set_postfix(loss=loss.item(), acc=val_correct/val_total)

    val_loss /= val_total
    val_acc   = val_correct / val_total

    history["train_loss"].append(train_loss)
    history["train_acc"].append(train_acc)
    history["val_loss"].append(val_loss)
    history["val_acc"].append(val_acc)

    print(f"Epoch {epoch+1}/{N_EPOCHS} | "
          f"train_loss {train_loss:.4f} train_acc {train_acc:.4f} | "
          f"val_loss {val_loss:.4f} val_acc {val_acc:.4f}")

torch.save(model.state_dict(), f"{output_dir}/model.pt")
with open(f"{output_dir}/history.json", "w") as f:
    json.dump(history, f, indent=2)

epochs = range(1, N_EPOCHS + 1)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(epochs, history["train_loss"], label="train")
ax1.plot(epochs, history["val_loss"], label="val")
ax1.set_title("Loss"); ax1.set_xlabel("Epoch"); ax1.legend()
ax2.plot(epochs, history["train_acc"], label="train")
ax2.plot(epochs, history["val_acc"], label="val")
ax2.set_title("Accuracy"); ax2.set_xlabel("Epoch"); ax2.legend()
plt.tight_layout()
plt.savefig(f"{output_dir}/curves.png")
print(f"Saved to {output_dir}/")