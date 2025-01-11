import torch
import threading
import pickle

from utils import get_logger

logger = get_logger(__name__)


class DefaultInferenceSaver:
    def __init__(self):
        pass

    def start(self, filename=None):
        pass

    def reset(self):
        pass

    def save(self, logits, turn_ids, labels):
        pass

    def save_to_file(self, filename):
        pass

    def write_to_file(self, filename, logits, turn_ids, labels):
        pass

    def shutdown(self):
        pass


class InferenceSaver(DefaultInferenceSaver):
    def __init__(self, save_file, device="cpu"):
        self.device = device

        self.save_file = save_file
        self.running = True

        self.lock = threading.Lock()
        self.cv = threading.Condition(self.lock)

        self.turn_ids = torch.tensor([]).to(device)
        self.labels = torch.tensor([]).to(device)
        self.logits = torch.tensor([]).to(device)

        self.thread = threading.Thread(target=self.save_to_file, args=(self.save_file,))
        self.buffer = 10000

    def start(self, filename=None):
        if filename is not None:
            self.save_file = filename

        self.reset()
        self.thread = threading.Thread(target=self.save_to_file, args=(self.save_file,))
        self.thread.start()

        self.running = True

    def reset(self):
        self.turn_ids = torch.tensor([])
        self.labels = torch.tensor([])
        self.logits = torch.tensor([])

        with open(self.save_file, "w") as file:
            file.write("turn_id,logits,labels\n")

    def save(self, logits, turn_ids, labels):
        self.logits = torch.cat((self.logits, logits.to(self.device)))
        self.turn_ids = torch.cat((self.turn_ids, turn_ids.to(self.device)))
        self.labels = torch.cat((self.labels, labels.to(self.device)))

        if len(self.logits) % self.buffer == 0:
            with self.cv:
                self.cv.notify()

    def save_to_file(self, filename):
        while self.running:
            with self.cv:
                while len(self.logits) < self.buffer and self.running:
                    self.cv.wait(timeout=1.0)

                logits = self.logits.clone()
                turn_ids = self.turn_ids.clone()
                labels = self.labels.clone()

                self.logits = torch.tensor([])
                self.turn_ids = torch.tensor([])
                self.labels = torch.tensor([])

            self.write_to_file(filename, logits, turn_ids, labels)

    def write_to_file(self, filename, logits, turn_ids, labels):
        logger.info(f"Writing to file {filename}")
        with open(filename, "a") as file:
            write_to_file = "\n".join(
                [
                    f"{turn_id},{logit},{label}"
                    for turn_id, logit, label in zip(turn_ids, logits, labels)
                ]
            )
            # file.write(write_to_file)
            logger.info(f"Writing to file {write_to_file[:100]}")

        pickle.dump(
            {
                "turn_ids": turn_ids,
                "logits": logits,
                "labels": labels,
            },
            open(filename + ".pkl", "wb"),
        )

    def shutdown(self):
        self.running = False
        with self.cv:
            self.cv.notify()

        self.thread.join()
        logger.info("Inference Saver shutdown")
