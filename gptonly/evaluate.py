import copy
import json
import logging
import os
from argparse import Namespace
from random import random

import torch
from torch.utils.data import DataLoader

from data import PairwiseGenerationDM
from gptonly import GPT
from gptonly.train import build_parser, get_latest_model
from pairwisegpt.evaluate import EvaluateType, get_abs_path
from utils import pp_pair_dialogs


def load_model(config):
    load_path = get_abs_path(config.load_path)
    logging.getLogger(__name__).info(
        f"model: loading model from {load_path}")

    with open(os.path.join(load_path, "config.json")) as f:
        config = Namespace(**json.load(f))

    logging.getLogger(__name__).info(f"Loaded config: {config}")

    model = GPT(
        **vars(config)
    )

    print("LOAD MODEL from ", load_path)
    load_model_file = get_latest_model(os.path.dirname(
        load_path))
    checkpoint = torch.load(load_model_file)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.cuda()

    config.overwrite = False
    test_ds = PairwiseGenerationDM(
        split="test",
        tokenizer=model.get_tokenizer(),
        basic_mode=config.serialise_data,
        **vars(config),
    )
    test_ds.prepare_data()
    test_dl = DataLoader(
        test_ds,
        batch_size=config.batch_size,
        collate_fn=test_ds.collate_fn,
        num_workers=8,
        shuffle=True
    )
    return model, test_ds, test_dl

class Analyser:
    def __init__(self,
                 base_config,
                 compare_config=None,
                 **kwargs):
        """
        Should only load one model at a time to avoid memory issues
        """
        self.base_config = base_config
        self.new_config = compare_config

    def generate_until_ts(self, device="cuda"):
        self.model1, self.ds1, self.dl1 = load_model(self.base_config)

        for i in range(1000):
            batch_idx = random.randint(0, len(self.ds1))
            seq_idx = random.randint(50, 200)
            seq_length = 50

            speakerA = self.ds1[batch_idx]['speakerA']
            speakerB = self.ds1[batch_idx]['speakerB']

            prior_batches = torch.tensor(
                [batch_idx.item() for _ in range(seq_idx - seq_length, seq_idx + 1)], device=device)
            prior_idx = torch.tensor(
                [idx for idx in range(seq_idx - seq_length, seq_idx + 1)], device=device)

            post_batches = torch.tensor(
                [batch_idx.item() for _ in range(seq_idx + 1, seq_idx + 50)], device=device)
            post_idx = torch.tensor(
                [idx for idx in range(seq_idx + 1, seq_idx + 50)], device=device)

            prior_batch = {
                'speakerA': {
                    k: v[(prior_batches, prior_idx)].unsqueeze(0) for k, v in speakerA.items() if
                    isinstance(v, torch.Tensor)
                },
                'speakerB': {
                    k: v[(prior_batches, prior_idx)].unsqueeze(0) for k, v in speakerB.items() if
                    isinstance(v, torch.Tensor)
                }
            }
            generated = self.model1.generate(context=prior_batch, stop_at_eos=True)
            self.print_dialogs(prior_batch, generated, post_batches)

    def print_dialogs(self, prior, gen, post):
        prior_length = 20
        curr = [40, 52]
        print("PRIOR")
        pp_pair_dialogs(self.model1.tokenizer,
                        prior['input_ids'][0], curr=curr)
        print()

        step = 7
        print("GENERATED")
        for idx in range(0, len(gen['input_idsA'][0]), step):
            pp_pair_dialogs(self.model1.tokenizer, input_ids=gen[0], curr=[
                idx, idx+step], speaker="speakerA")
            print()

        print("--------------------")


if __name__ == "__main__":
    parser = build_parser()
    config = parser.parse_args()

    config.device = "cuda"
    config.load_path = "trained_model/2024-03-31:11-44-53/"

    new_config = copy.deepcopy(config)
    new_config.load_path = "trained_model/2024-03-31:11-44-53/"

    print(config)
    print("----------------")
    print(new_config)

    analyser = Analyser(config, new_config)
    analyser(EvaluateType.OVERLAP_EFFECT)
