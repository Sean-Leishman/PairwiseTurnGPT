import torch
import datasets
import pprint

from data.tokenizer import SpokenDialogTokenizer

from conversational_dm import ConversationalDM
from spoken_dm import SpokenDM
from base import Datasets, DialogDMInterface
from data.utils import retokenize, get_logger

from torch.utils.data import DataLoader, ConcatDataset


datasets.disable_caching()
logger = get_logger(__name__)

CONVERSATIONAL_DATASETS = [
    Datasets.CURIOSITY_DIALOG,
    Datasets.DAILY_DIALOG,
    Datasets.METAWOZ,
    Datasets.MULTIWOZ,
    Datasets.TASKMASTER1,
    Datasets.TASKMASTER2,
    Datasets.TASKMASTER3,
]

PAIRWISE_GENERATION_DATASETS = [Datasets.SWITCHBOARD, Datasets.FISHER, Datasets.EDACC]

MAP_STRING_TO_DATASET = {
    "switchboard": Datasets.SWITCHBOARD,
    "fisher": Datasets.FISHER,
    "edacc": Datasets.EDACC,
    "curiosity_dialog": Datasets.CURIOSITY_DIALOG,
    "daily_dialog": Datasets.DAILY_DIALOG,
    "metawoz": Datasets.METAWOZ,
    "multiwoz": Datasets.MULTIWOZ,
    "taskmaster1": Datasets.TASKMASTER1,
    "taskmaster2": Datasets.TASKMASTER2,
    "taskmaster3": Datasets.TASKMASTER3,
}


def get_dataset(dataset_str):
    if dataset_str in MAP_STRING_TO_DATASET:
        return MAP_STRING_TO_DATASET[dataset_str]
    elif isinstance(dataset_str, Datasets):
        return dataset_str
    else:
        raise ValueError(f"Dataset {dataset_str} not found")


"""
A simple container for the conversation and pairwise generation datasets.
Main processing occurs within the datasets themselves and this class is used
to combine the two datasets into a single dataset for simpler loading

Should be able to handle any combination of conversation and pairwise generation datasets

Raises:
    ValueError: If no data is found in either dataset
"""


class DialogDM(DialogDMInterface):
    def __init__(
        self,
        tokenizer=None,
        datasets=[Datasets.SWITCHBOARD],
        combine_speaker=False,
        remove_speaker_key=False,
        remove_special_tokens=False,
        max_length=256,
        **kwargs,
    ):
        self.tokenizer = tokenizer if tokenizer is not None else SpokenDialogTokenizer()

        datasets = [get_dataset(x) for x in datasets]
        conversation_datasets = [x for x in datasets if x in CONVERSATIONAL_DATASETS]
        pairwise_generation_datasets = [
            x for x in datasets if x in PAIRWISE_GENERATION_DATASETS
        ]

        logger.info(
            f"Initializing DialogDM with {conversation_datasets} conversation datasets and {pairwise_generation_datasets} pairwise generation datasets"
        )

        self.combine_speaker = combine_speaker
        self.remove_speaker_key = remove_speaker_key
        self.remove_special_tokens = remove_special_tokens
        self.max_length = max_length

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.dm = []

        if len(conversation_datasets) > 0:
            self.dm.append(
                ConversationalDM(
                    tokenizer=tokenizer,
                    datasets=conversation_datasets,
                    combine_speaker=self.combine_speaker,
                    remove_special_tokens=self.remove_special_tokens,
                    max_length=self.max_length,
                    **kwargs,
                )
            )
        if len(pairwise_generation_datasets) > 0:
            self.dm.append(
                SpokenDM(
                    tokenizer=tokenizer,
                    datasets=pairwise_generation_datasets,
                    combine_speaker=self.combine_speaker,
                    remove_special_tokens=self.remove_special_tokens,
                    max_length=self.max_length,
                    **kwargs,
                )
            )

        self.dataset = []

        self.key_set = {
            "input_ids",
            "attention_mask",
            "speaker_ids",
            "token_type_ids",
            "other_token_type_ids",
            "turn_ids",
            "conv_id",
            "loss_mask",
        }
        self.warned_key_set = {"time_until_ts", "conv_id"}

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def print_info(self):
        top_10_conv_ids = [self.dataset[x]["speakerA"]["conv_id"] for x in range(10)]
        logger.info(f"Top 10 conversation ids: {top_10_conv_ids}")

    def get_by_conv_id(self, conv_id):
        for i in range(len(self)):
            if self.dataset[i]["speakerA"]["conv_id"] == conv_id:
                return self.dataset[i]

    def prepare_data(self):
        for dm in self.dm:
            dm.prepare_data()

        if sum(len(x) for x in self.dm) == 0:
            raise ValueError("No data found in any dataset")

        datasets = [x for x in self.dm if len(x) > 0]
        self.dataset = ConcatDataset(datasets)

        self.log_info()

    def setup(self):
        for dm in self.dm:
            dm.setup()

    def reset(self):
        for dm in self.dm:
            dm.reset()

    def log_info(self):
        logger.info(f"Dialog DM: {len(self)} samples")

    def collate_fn(self, batch):
        def collate_fn_channel(batch):
            ret = self.tokenizer.pad(
                {"input_ids": [b["input_ids"][: self.max_length] for b in batch]},
                padding="max_length",
                max_length=self.max_length,
            )

            # Add required keys from batch to ret
            for key, value in batch[0].items():
                if key == "input_ids" or key == "attention_mask":
                    continue
                if key == "conv_id":
                    first_conv_id = batch[0][key]
                    ret[key] = [b[key] if key in b else first_conv_id for b in batch]
                    continue
                if not isinstance(value, torch.Tensor) or key not in self.key_set:
                    continue

                ret[key] = self.tokenizer.pad(
                    {"input_ids": [b[key][: self.max_length] for b in batch]},
                    padding="max_length",
                    max_length=self.max_length,
                )["input_ids"]

            # Add missing keys to ret
            for key in self.key_set:
                if key not in ret:
                    if key not in self.warned_key_set:
                        self.warned_key_set.add(key)
                        logger.warning(
                            f"Key {key} not found in batch, setting to torch.zeros_like() with shape {ret['input_ids'].shape}"
                        )

                    if key == "conv_id":
                        ret[key] = "placeholder"
                        continue

                    ret[key] = torch.zeros_like(ret["input_ids"])

            for k, v in ret.items():
                if isinstance(v, torch.Tensor):
                    ret[k] = v.clone().detach()
                else:
                    ret[k] = v

            return ret

        if self.remove_speaker_key:
            return collate_fn_channel([x["speakerA"] for x in batch])

        result = {}
        for speaker in batch[0].keys():
            if "speaker" not in speaker:
                continue

            result[speaker] = collate_fn_channel([x[speaker] for x in batch])

            assert all(
                [key in result[speaker].keys() for key in self.key_set]
            ), f"{pprint.pformat(result[speaker])}"
        return result

    def retokenize(self, tokenizer):
        new_dataset = []
        for idx in range(len(self.dataset)):
            ds_inst = self.dataset[idx]
            new_dataset.append(retokenize(self.tokenizer, tokenizer, ds_inst))

        self.tokenizer = tokenizer
        self.dataset = new_dataset


if __name__ == "__main__":
    import tqdm

    dm = DialogDM(
        include_partial_overlaps=True,
        include_backchannels=True,
        include_overlaps=True,
        end_of_utterance_tokens=["<eint>", "<ebc>"],
        datasets=[
            "switchboard",
            "daily_dialog",
            "taskmaster1",
            "taskmaster2",
            "taskmaster3",
            "multiwoz",
            "metawoz",
            "curiosity_dialog",
        ],
        load_from_cache=True,
    )
    dm.prepare_data()

    dl = DataLoader(dm, batch_size=4, collate_fn=dm.collate_fn, shuffle=True)
    for batch in tqdm.tqdm(dl):
        pass
