import torch


class BinaryBACC:
    def __init__(self, threshold, device="cpu", **kwargs):
        self.threshold = threshold
        self.device = device

        self.tp = torch.tensor(0, device=self.device)
        self.fp = torch.tensor(0, device=self.device)
        self.tn = torch.tensor(0, device=self.device)
        self.fn = torch.tensor(0, device=self.device)

    def to(self, device):
        self.device = device
        return self

    def update(self, probs, labels, **kwargs):
        self.tp += torch.sum(torch.logical_and(labels, probs >= self.threshold))
        self.fp += torch.sum(
            torch.logical_and(torch.logical_not(labels), probs >= self.threshold)
        )
        self.tn += torch.sum(
            torch.logical_and(torch.logical_not(labels), probs < self.threshold)
        )
        self.fn += torch.sum(torch.logical_and(labels, probs < self.threshold))

    def compute(self):
        specificity = torch.div(
            self.tn, self.tn + self.fp, out=torch.zeros_like(self.tn).float()
        )
        recall = torch.div(
            self.tp, self.tp + self.fn, out=torch.zeros_like(self.tp).float()
        )
        return torch.mean(torch.div(specificity + recall, 2))

    def count(self):
        return self.tp + self.fp + self.tn + self.fn

    def reset(self):
        self.tp = torch.tensor(0, device=self.device)
        self.fp = torch.tensor(0, device=self.device)
        self.tn = torch.tensor(0, device=self.device)
        self.fn = torch.tensor(0, device=self.device)


class MultiClassBACC:
    def __init__(self, threshold, num_classes, device="cpu", **kwargs):
        self.threshold = threshold
        self.device = device

        self.tp = torch.tensor([0 for _ in range(num_classes)], device=self.device)
        self.fp = torch.tensor([0 for _ in range(num_classes)], device=self.device)
        self.tn = torch.tensor([0 for _ in range(num_classes)], device=self.device)
        self.fn = torch.tensor([0 for _ in range(num_classes)], device=self.device)

    def to(self, device):
        self.device = device
        return self

    def update(self, probs, labels, **kwargs):
        assert (
            probs.shape == labels.shape
        ), f"Probs and labels must have the same shape: {probs.shape}, {labels.shape}"
        assert (
            len(probs.shape) == 2
        ), f"Probs and labels must be 2D tensors: {probs.shape}"
        assert probs.shape[1] == len(
            self.tp
        ), "Number of classes must match the number of classes in the metric"

        self.tp += torch.sum(torch.logical_and(labels, probs >= self.threshold))
        self.fp += torch.sum(
            torch.logical_and(torch.logical_not(labels), probs >= self.threshold)
        )
        self.tn += torch.sum(
            torch.logical_and(torch.logical_not(labels), probs < self.threshold)
        )
        self.fn += torch.sum(torch.logical_and(labels, probs < self.threshold))

    def compute(self):
        specificity = torch.div(
            self.tn, self.tn + self.fp, out=torch.zeros_like(self.tn).float()
        )
        recall = torch.div(
            self.tp, self.tp + self.fn, out=torch.zeros_like(self.tp).float()
        )
        bacc = (specificity + recall) / 2

        return torch.mean(bacc)

    def count(self):
        return torch.sum(self.tp + self.fp + self.tn + self.fn)

    def reset(self):
        self.tp = torch.tensor([0 for _ in range(len(self.tp))], device=self.device)
        self.fp = torch.tensor([0 for _ in range(len(self.fp))], device=self.device)
        self.tn = torch.tensor([0 for _ in range(len(self.tn))], device=self.device)
        self.fn = torch.tensor([0 for _ in range(len(self.fn))], device=self.device)


class BACC:
    def __init__(self, threshold, task="binary", num_classes=1, device="cpu", **kwargs):
        self.threshold = threshold
        self.device = device

        self.task = task
        assert (
            self.task == "binary" or self.task == "multiclass"
        ), "Task must be binary or multiclass"

        self.num_classes = num_classes
        if self.task == "multiclass":
            assert (
                self.num_classes >= 1
            ), "Number of classes must be greater than 1 for multiclass task"

        self.bacc = (
            BinaryBACC(threshold, device=device)
            if self.task == "binary"
            else MultiClassBACC(threshold, num_classes, device=device)
        )

    def to(self, device):
        self.bacc.to(device)
        return self

    def update(self, probs, labels, **kwargs):
        self.bacc.update(probs, labels, **kwargs)

    def compute(self):
        return self.bacc.compute()

    def count(self):
        return self.bacc.count()

    def reset(self):
        self.bacc.reset()

    def __str__(self):
        return "BACC"
