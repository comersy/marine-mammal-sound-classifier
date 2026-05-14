import torch.nn as nn
from transformers import ASTForAudioClassification

class MarineAST(nn.Module):
    def __init__(self, num_classes=32):
        super().__init__()
        self.model = ASTForAudioClassification.from_pretrained(
            "MIT/ast-finetuned-audioset-10-10-0.4593",
            num_labels=num_classes,
            ignore_mismatched_sizes=True
        )

    def forward(self, x):
        return self.model(x).logits