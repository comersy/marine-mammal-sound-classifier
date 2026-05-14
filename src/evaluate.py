import argparse
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from dataset_cnn import MarineDataset
from dataset_ast import MarineDatasetAST
from model_cnn_scratch import MarineCNN
from model_resnet18 import MarineResNet18
from model_ast import MarineAST

parser = argparse.ArgumentParser()
parser.add_argument("--model", choices=["cnn_scratch", "resnet18", "ast"], required=True)
args = parser.parse_args()

if args.model == "ast":
    train_dataset = MarineDatasetAST("data/train")
    test_dataset  = MarineDatasetAST("data/test")
else:
    train_dataset = MarineDataset("data/train")
    test_dataset  = MarineDataset("data/test")

test_loader = DataLoader(test_dataset, batch_size=32)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if args.model == "cnn_scratch":
    model = MarineCNN(num_classes=32).to(device)
elif args.model == "resnet18":
    model = MarineResNet18(num_classes=32).to(device)
else:
    model = MarineAST(num_classes=32).to(device)

model.load_state_dict(torch.load(f"outputs/{args.model}/model.pt", map_location=device))
model.eval()

all_preds, all_labels = [], []
with torch.no_grad():
    for x, label in test_loader:
        x = x.to(device)
        out = model(x)
        all_preds.extend(out.argmax(1).cpu().tolist())
        all_labels.extend(label.tolist())

print(classification_report(all_labels, all_preds, target_names=train_dataset.classes))

cm = confusion_matrix(all_labels, all_preds, labels=list(range(32)))
plt.figure(figsize=(14, 12))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=train_dataset.classes, yticklabels=train_dataset.classes)
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig(f"outputs/{args.model}/confusion_matrix.png")
print(f"Confusion matrix saved to outputs/{args.model}/confusion_matrix.png")