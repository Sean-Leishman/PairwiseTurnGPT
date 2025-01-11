from torchmetrics import F1Score


class F1:
    def __init__(
        self,
        task="binary",
        num_classes=1,
        threshold=0,
        average=None,
        device="cpu",
        *args,
        **kwargs,
    ):
        self.task = task
        self.num_classes = num_classes
        self.threshold = threshold

        self.device = device

        self.average = average

        if self.task == "binary":
            self.f1 = F1Score(
                task=self.task,
                threshold=self.threshold,
                num_classes=self.num_classes,
                *args,
                **kwargs,
            ).to(device)
        else:
            self.f1 = F1Score(
                task=self.task,
                threshold=self.threshold,
                num_classes=self.num_classes,
                ignore_index=-100,
                *args,
                **kwargs,
            ).to(device)

    def __str__(self):
        return "F1"

    def count(self):
        return self.f1.tp + self.f1.fp + self.f1.tn + self.f1.fn

    def to(self, device):
        self.f1.to(device)
        return self

    def update(self, preds, labels, **kwargs):
        self.f1.update(preds, labels)

    def compute(self):
        return self.f1.compute()

    def reset(self):
        self.f1.reset()
