from torchmetrics import AUROC


class RocAuc:
    def __init__(self, *args, **kwargs):
        self.auroc = AUROC()

    def __call__(self, y_pred, y_true):
        return self.auroc(y_pred, y_true)

    def __str__(self):
        return "RocAuc"

    def count(self):
        return self.auroc.tp + self.auroc.fp + self.auroc.tn + self.auroc.fn

    def to(self, device):
        self.auroc.to(device)
        return self

    def update(self, y_pred, y_true):
        self.auroc.update(y_pred, y_true)

    def compute(self):
        return self.auroc.compute()
