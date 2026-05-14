import torch
import json
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import MarineDataset
from model_cnn import MarineCNN

train_dataset = MarineDataset("data/train")
test_dataset  = MarineDataset("data/test")

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader  = DataLoader(test_dataset,  batch_size=32)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = MarineCNN(num_classes=32).to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

history = {
    "train_loss": [],
    "train_acc":  [],
    "val_loss":   [],
    "val_acc":    [],
}

N_EPOCHS = 20

for epoch in range(N_EPOCHS):
    # ── Training ─────────────────────────────────────────────
    model.train()
    train_loss, train_correct, train_total = 0.0, 0, 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{N_EPOCHS} [train]")
    for mel, label in pbar:
        mel, label = mel.to(device), label.to(device)
        optimizer.zero_grad()
        out  = model(mel)
        loss = criterion(out, label)
        loss.backward()
        optimizer.step()

        train_loss    += loss.item() * mel.size(0)
        train_correct += (out.argmax(1) == label).sum().item()
        train_total   += mel.size(0)

        pbar.set_postfix(loss=loss.item(), acc=train_correct/train_total)

    train_loss /= train_total
    train_acc   = train_correct / train_total

    # ── Validation ───────────────────────────────────────────
    model.eval()
    val_loss, val_correct, val_total = 0.0, 0, 0

    with torch.no_grad():
        pbar = tqdm(test_loader, desc=f"Epoch {epoch+1}/{N_EPOCHS} [val]  ")
        for mel, label in pbar:
            mel, label = mel.to(device), label.to(device)
            out  = model(mel)
            loss = criterion(out, label)

            val_loss    += loss.item() * mel.size(0)
            val_correct += (out.argmax(1) == label).sum().item()
            val_total   += mel.size(0)

            pbar.set_postfix(loss=loss.item(), acc=val_correct/val_total)

    val_loss /= val_total
    val_acc   = val_correct / val_total

    # ── Save history ─────────────────────────────────────────
    history["train_loss"].append(train_loss)
    history["train_acc"].append(train_acc)
    history["val_loss"].append(val_loss)
    history["val_acc"].append(val_acc)

    print(f"Epoch {epoch+1}/{N_EPOCHS} | "
          f"train_loss {train_loss:.4f} train_acc {train_acc:.4f} | "
          f"val_loss {val_loss:.4f} val_acc {val_acc:.4f}")

# ── Save model + history ────────────────────────────────────
torch.save(model.state_dict(), "outputs/cnn.pt")
with open("outputs/history.json", "w") as f:
    json.dump(history, f, indent=2)