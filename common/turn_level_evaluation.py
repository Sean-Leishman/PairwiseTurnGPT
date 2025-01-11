import pickle
import os
import torch

from utils import get_logger, get_abs_path

logger = get_logger(__name__)


class InferenceReader:
    def __init__(self, path, device="cuda:0"):
        self.path = path
        if not os.path.exists(path):
            raise FileNotFoundError(f"File {path} not found.")

        self.probs = torch.tensor([])
        self.labels = torch.tensor([])
        self.turn_ids = torch.tensor([])

        self.device = torch.device(device)

    def __call__(self):
        self.read()

    def read(self, split="val"):
        filepath = os.path.join(self.path, f"{split}_inference_0.txt.pkl")
        with open(filepath, "rb") as f:
            return_dict = pickle.load(f)

        self.probs = return_dict["logits"]
        self.labels = return_dict["labels"]
        self.turn_ids = return_dict["turn_ids"]

        print(f"Probs: {self.probs.shape}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--split", type=str, default="test")
    args = parser.parse_args()

    path = get_abs_path(args.path)
    reader = InferenceReader(path, device=args.device)

    reader.read(split=args.split)
