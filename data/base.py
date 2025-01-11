from collections.abc import Generator
import logging
import math
import bisect
import torch

from abc import ABC, abstractmethod
from torch.utils.data import Dataset, ConcatDataset
from dataclasses import dataclass, field
from enum import IntEnum

logger = logging.getLogger(__name__)


def _get_ds_conv_id(conv_id: str) -> tuple[int, int]:
    conv_id = conv_id.lower()
    if "sw" in conv_id:
        ds = Datasets.SWITCHBOARD.value
        # sw4333A-ms98-a-0001 -> 4333
        conv_id_int = int(conv_id[2:6])
    elif "fe" in conv_id:
        ds = Datasets.FISHER.value
        # fe_03_00001 -> 00001
        conv_id_int = int(conv_id[-5:])
    elif "ed" in conv_id:
        ds = Datasets.EDACC.value
        # EDACC-C01 -> 01
        if "p" in conv_id:
            # EDACC-C01-P2 -> 01
            conv_id_int = int(conv_id[-5:-3])
            add = int(conv_id[-1])
        else:
            conv_id_int = int(conv_id[-2:])
            add = 0

        conv_id_int = conv_id_int * 10 + add
    else:
        raise ValueError(f"Unknown dataset for conversation ID {conv_id}")

    return ds, conv_id_int


class Datasets(IntEnum):
    # spoken_dm.py datasets
    SWITCHBOARD = 0
    FISHER = 1
    EDACC = 2

    # conversational_dm.py datasets
    CURIOSITY_DIALOG = 3
    DAILY_DIALOG = 4
    MULTIWOZ = 5
    METAWOZ = 6
    TASKMASTER1 = 7
    TASKMASTER2 = 8
    TASKMASTER3 = 9


class Split(IntEnum):
    TRAIN = 0
    VALID = 1
    TEST = 2


@dataclass
class Word:
    conv_id: str
    start: float
    end: float
    tokens: int
    word: str

    def __setitem__(self, key, value):
        self.__setattr__(key, value)

    def __getitem__(self, key):
        return self.__getattribute__(key)

    @staticmethod
    def default():
        return Word("", 0, math.inf, 0, "")


Dialog = list[list[Word]]


class TurnType(IntEnum):
    """
    (2) is reserved for the YIELD turn end type
    """

    NONE = 0
    NORMAL = 1
    BACKCHANNEL = 3
    INTERRUPT = 4
    OVERLAP = 5

    NON_OVERLAP = 6


class TurnEndType(IntEnum):
    NONE = 0
    NORMAL = 1
    YIELD = 2


@dataclass
class TurnID:
    """
    Represents a unique ID for a turn in a dialog with the following attributes:
        - dataset: the dataset ID
        - conv_id: the conversation ID. This is unique within a dataset
        - turn_type: the type of turn (normal, backchannel, interrupt, overlap)
        - speaker: the speaker (A/B)
        - turn_id: the turn ID. This is unique within a conversation irrespective of speaker and turn type.
            We can define a global ordering of turns in a conversation by sorting by this ID.
            turn_id is based on the order of start times of the turns in a conversation.
    """

    id: int

    def __setitem__(self, key, value):
        self.__setattr__(key, value)

    def __getitem__(self, key):
        return self.__getattribute__(key)

    def __hash__(self):
        return hash(self.id)

    @property
    def value(self):
        return self.id

    def update_id(
        self,
        dataset: int | Datasets | None = None,
        conv_id: int | str = 0,
        turn_type: int | TurnType = TurnType.NORMAL,
        speaker: int | str = "A",
        turn_id: int = 0,
    ):
        """
        Builds a unique ID for a turn in a dialog based on the given attributes:
            - dataset: the dataset ID. If None, conv_id must be a string which is parsed to get the dataset ID
            - conv_id: the conversation ID. This is unique within a dataset. If dataset is None, this must be a string
            - turn_type: the type of turn (normal, backchannel, interrupt, overlap)
            - speaker: the speaker (A/B)
            - turn_id: the turn ID. This is unique within a conversation irrespective of speaker.
                        Two turns of different turn types (normal/interrupt, backchannel/overlap) can have the same turn_id.

        """
        if isinstance(dataset, Datasets):
            dataset = dataset.value

        if isinstance(turn_type, TurnType):
            turn_type = turn_type.value

        if isinstance(speaker, str):
            if speaker not in {"A", "B"}:
                raise ValueError(f"Speaker must be 'A' or 'B', not {speaker}")

            speaker = 0 if speaker == "A" else 1

        if dataset is None:
            if not isinstance(conv_id, str):
                raise ValueError("Conversation ID must be a string if dataset is None")

            dataset, conv_id = _get_ds_conv_id(conv_id)
        elif isinstance(conv_id, str):
            raise ValueError(
                "Conversation ID must be an integer if dataset is not None"
            )

        self.id = (
            (dataset << 60)
            | (conv_id << 34)
            | (turn_type << 30)
            | (speaker << 29)
            | turn_id
        )

        return self

    @staticmethod
    def default():
        return TurnID(0)

    @staticmethod
    def build_id(
        dataset: int | Datasets | None = None,
        conv_id: int | str = 0,
        turn_type: int | TurnType = TurnType.NORMAL,
        speaker: int | str = "A",
        turn_id: int = 0,
    ):
        return TurnID.default().update_id(dataset, conv_id, turn_type, speaker, turn_id)


@dataclass
class Turn:
    """
    Represents a turn in a dialog with the following attributes:
        - word: the text of the entire turn
        - turn_type: the type of turn (normal, backchannel, interrupt, overlap)
        - turn_end_type: the type of turn ending (normal, yield)
        - start: the start time of the turn
        - end: the end time of the turn
        - id: the unique ID of the turn
        - curr_prev_id: the unique ID of the previous turn from the same speaker (includes bc/overlap)
        - curr_next_id: the unique ID of the next turn from the same speaker (includes bc/overlap)
        - other_prev_id: the unique ID of the previous turn from the other speaker.
                If the current turn is a backchannel or overlap turn and completely overlapped
                this will be the ID of the turn that the current turn is
                responding to
                If the current turn is a normal turn then this will be the ID of the prior normal turn from the other speaker
        - other_next_id: the unique ID of the next turn from the other speaker
                If the current turn is a backchannel turn (not overlap) and completely overlapped
                this will be the ID of the turn that the current turn is
                responding to.
                If the current turn is a normal turn then this will be the ID of the next normal turn from the other speaker
        - overlaps: a list of IDs of turns that are completely (not partially) overlapped by this turn
        - overlapped_by: the ID of the turn (if it exists) that this turn is completely (not partially) overlapped by
    """

    word: str = field(repr=False)
    turn_type: TurnType = field(repr=True)
    turn_end_type: TurnEndType = field(repr=True)
    start: float
    end: float
    speaker: str
    conv_id: str

    turn_index: int = 0

    id: TurnID = field(default_factory=TurnID.default)
    curr_prev_id: TurnID | None = field(default_factory=TurnID.default)
    curr_next_id: TurnID | None = field(default_factory=TurnID.default)
    other_prev_id: TurnID | None = field(default_factory=TurnID.default)
    other_next_id: TurnID | None = field(default_factory=TurnID.default)

    overlaps: list[TurnID] = field(default_factory=list)
    overlapped_by: TurnID | None = None

    def __post_init__(self):
        self.id.update_id(
            conv_id=self.conv_id,
            turn_id=self.turn_index,
            turn_type=self.turn_type,
            speaker=self.speaker,
        )

    def __setitem__(self, key, value):
        self.__setattr__(key, value)

    def __getitem__(self, key):
        return self.__getattribute__(key)

    def to_dict(self):
        return {
            "word": self.word,
            "turn_type": self.turn_type,
            "turn_end_type": self.turn_end_type,
            "start": self.start,
            "end": self.end,
            "speaker": self.speaker,
            "conv_id": self.conv_id,
            "turn_index": self.turn_index,
            "id": self.id.value,
            "curr_prev_id": (
                self.curr_prev_id.value if self.curr_prev_id is not None else None
            ),
            "curr_next_id": (
                self.curr_next_id.value if self.curr_next_id is not None else None
            ),
            "other_prev_id": (
                self.other_prev_id.value if self.other_prev_id is not None else None
            ),
            "other_next_id": (
                self.other_next_id.value if self.other_next_id is not None else None
            ),
            "overlaps": [x.value for x in self.overlaps],
            "overlapped_by": (
                self.overlapped_by.value if self.overlapped_by is not None else None
            ),
        }

    @staticmethod
    def from_dict(data):
        for key, value in data.items():
            if isinstance(value, torch.Tensor):
                data[key] = value.tolist()

        return Turn(
            data["word"],
            data["turn_type"],
            data["turn_end_type"],
            data["start"],
            data["end"],
            data["speaker"],
            data["conv_id"],
            data["turn_index"],
            TurnID(data["id"]),
            TurnID(data["curr_prev_id"]),
            TurnID(data["curr_next_id"]),
            TurnID(data["other_prev_id"]),
            TurnID(data["other_next_id"]),
            [TurnID(x) for x in data["overlaps"]],
            TurnID(data["overlapped_by"]),
        )

    @staticmethod
    def default(speaker: str = "A", conv_id: str = ""):
        return Turn(
            "", TurnType.NORMAL, TurnEndType.NORMAL, 0.0, math.inf, speaker, conv_id
        )

    def update(self, **kwargs):
        for key, value in kwargs.items():
            self.__setattr__(key, value)

        self.id.update_id(
            conv_id=self.conv_id,
            turn_id=self.turn_index,
            turn_type=self.turn_type,
            speaker=self.speaker,
        )

    def get_turn_duration(self):
        return self.end - self.start

    def get_turn_length(self):
        return len(self.word.split())


class DialogDMInterface(Dataset, ABC):
    @abstractmethod
    def prepare_data(self):
        pass

    @abstractmethod
    def setup(self) -> bool:
        pass

    @abstractmethod
    def reset(self):
        pass


class EmptyDM(DialogDMInterface):
    def __init__(self):
        self.data = []

    def prepare_data(self):
        pass

    def setup(self) -> bool:
        return True

    def reset(self):
        pass

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if len(self.data) == 0:
            raise ValueError("No data found in dataset")

        return self.data[idx]


class Process(ABC):
    def __init__(self, tokenizer, store_raw=False):
        self.tokenizer = tokenizer
        self.store_raw = store_raw

        self.tokens_dict = {
            "emp": self.tokenizer.convert_tokens_to_ids("<emp>"),
            "eot": self.tokenizer.convert_tokens_to_ids("<eot>"),
            "ebc": self.tokenizer.convert_tokens_to_ids("<ebc>"),
            "eint": self.tokenizer.convert_tokens_to_ids("<eint>"),
            "yield": self.tokenizer.convert_tokens_to_ids("<yield>"),
            "speakerA": self.tokenizer.convert_tokens_to_ids("<speakerA>"),
            "speakerB": self.tokenizer.convert_tokens_to_ids("<speakerB>"),
            "bc": self.tokenizer.convert_tokens_to_ids("<bc>"),
        }

        if not self.tokenizer.is_special_token("<yield>"):
            self.tokens_dict["yield"] = self.tokenizer.convert_tokens_to_ids("<eot>")

        if not self.tokenizer.is_special_token("<eint>"):
            self.tokens_dict["eint"] = self.tokenizer.convert_tokens_to_ids("<eot>")

        if not self.tokenizer.is_special_token("<ebc>"):
            self.tokens_dict["ebc"] = self.tokenizer.convert_tokens_to_ids("<eot>")

        if not self.tokenizer.is_special_token("<emp>"):
            self.tokens_dict["emp"] = self.tokenizer.convert_tokens_to_ids("<eot>")

        if not self.tokenizer.is_special_token("<speakerA>"):
            self.tokens_dict["speakerA"] = 1

        if not self.tokenizer.is_special_token("<speakerB>"):
            self.tokens_dict["speakerB"] = 2

        if not self.tokenizer.is_special_token("<bc>"):
            self.tokens_dict["bc"] = self.tokenizer.convert_tokens_to_ids("<eot>")

    @abstractmethod
    def config_to_dict(self):
        return {
            "process": self.__class__.__name__,
        }

    @abstractmethod
    def process(self, datasets, in_parallel=False):
        pass

    def _tokenize_sentence(self, dialog):
        tokens = self.tokenizer(
            dialog,
            padding="do_not_pad",
            truncation=True,
            max_length=12400,
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        return tokens

    def _match_tokens_words(self, tokens, dialog, sentence) -> Dialog:
        """
        Builds a list of dict (start_idx, end_idx, id, tokens) from tokens:
            output from tokenizer; dialog: features from dataset that contain
            word timings; sentence: string of the input

        This is done to split words within sentence into their respective and assign
        timings to each of these subtokens
        """
        dialog_word_idx = 0
        dialog_idx = 0

        decoded = ""
        decoded_split = []
        new_tokens = []
        token_timings = []

        utt_outputs = []
        outputs = []

        if len(dialog) == 0:
            return outputs

        conv_id = dialog[0]["conv_id"]

        for token_id, offset in zip(
            tokens["input_ids"][0], tokens["offset_mapping"][0]
        ):
            word = sentence[offset[0] : offset[1]].strip()
            decoded += word
            decoded_split.append(word)
            new_tokens.append(token_id.item())

            if word == "the":
                pass

            if dialog_word_idx >= len(dialog[dialog_idx]["wfeats"]):
                dialog_idx += 1
                dialog_word_idx = 0
                outputs.append(utt_outputs)
                utt_outputs = []

            curr_wfeat = dialog[dialog_idx]["wfeats"][dialog_word_idx]
            if decoded == curr_wfeat["word"] or len(decoded) > len(curr_wfeat["word"]):
                if len(decoded) > len(curr_wfeat["word"]):
                    logger.warning(
                        f"Decoded word '{decoded}' does not match '{curr_wfeat['word']}'"
                    )
                curr_wfeat["tokens"] = new_tokens

                # Add timings for start and end of each subtoken
                token_timings = []
                step_length = 1 / len(new_tokens)
                duration = curr_wfeat["end"] - curr_wfeat["start"]
                for idx, token in enumerate(new_tokens):
                    offset_start = step_length * idx * duration
                    offset_end = step_length * (idx + 1) * duration
                    start_time = round(curr_wfeat["start"] + offset_start, 5)
                    end_time = round(curr_wfeat["start"] + offset_end, 5)
                    token_timings.append((start_time, end_time))

                curr_wfeat["token_timings"] = token_timings
                curr_wfeat["start"] = round(curr_wfeat["start"], 3)
                curr_wfeat["end"] = round(curr_wfeat["end"], 3)

                for idx, token in enumerate(new_tokens):
                    output = Word(
                        conv_id,
                        token_timings[idx][0],
                        token_timings[idx][1],
                        token,
                        decoded_split[idx],
                    )
                    utt_outputs.append(output)

                new_tokens = []
                decoded = ""
                decoded_split = []
                dialog_word_idx += 1

        outputs.append(utt_outputs)

        assert all(
            all("tokens" in word for word in key["wfeats"]) for key in dialog
        ), f"{conv_id} has missing tokens"
        assert sum([len(x) for x in outputs]) == len(tokens["input_ids"][0])

        return outputs

    """
    Returns for each speaker a list of words for each turn within a dialog.

    SpeakerA: List[Turn[Word]]
    SpeakerB: List[Turn[Word]]
    """

    def _get_tokens(self, speakerA, speakerB) -> tuple[Dialog, Dialog]:
        sentenceA = " ".join(feature["text"] for feature in speakerA)
        sentenceB = " ".join(feature["text"] for feature in speakerB)

        tokensA = self._tokenize_sentence(sentenceA)
        tokensB = self._tokenize_sentence(sentenceB)

        dialogA = self._match_tokens_words(tokensA, speakerA, sentenceA)
        dialogB = self._match_tokens_words(tokensB, speakerB, sentenceB)

        return dialogA, dialogB


class ConcatDatasetLoader(ConcatDataset):
    def __init__(self, datasets):
        super().__init__(datasets)

    def __getitem__(self, idx):
        return super().__getitem__(idx)

    def get_batch(self, start=0, batch_size=4):
        if batch_size <= 0:
            raise ValueError("batch_size should be a positive integer")

        if start < 0:
            if -start > len(self):
                raise ValueError(
                    "absolute value of index should not exceed dataset length"
                )
            start = len(self) + start

        remaining = batch_size
        current_idx = start
        return_val = []
        while remaining > 0:
            dataset_idx = bisect.bisect_right(self.cumulative_sizes, current_idx)
            if dataset_idx >= len(self.datasets):
                return return_val

            if dataset_idx == 0:
                local_start = current_idx
            else:
                local_start = current_idx - self.cumulative_sizes[dataset_idx - 1]

            if local_start + remaining > len(self.datasets[dataset_idx]):
                local_end = len(self.datasets[dataset_idx])
            else:
                local_end = local_start + remaining

            local_batch_size = local_end - local_start
            batch = self.datasets[dataset_idx].get_batch(local_start, local_batch_size)

            if isinstance(batch, list):
                for b in batch:
                    yield b
            elif isinstance(batch, dict):
                first_value = next(iter(batch.values()))  # {"dialog": [N * []]}
                for i in range(len(first_value)):
                    item = {key: batch[key][i] for key in batch.keys()}
                    yield item
            elif isinstance(batch, Generator):
                for b in batch:
                    yield b

            remaining -= local_batch_size
            current_idx += local_batch_size
