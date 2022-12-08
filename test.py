import torch

print(torch.__version__)

from torchmetrics import AUROC

preds = torch.tensor([0.45])
target = torch.tensor([1])
auroc = AUROC(pos_label=1)
xx = auroc(preds, target)

print(xx)
