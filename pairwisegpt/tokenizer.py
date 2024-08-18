# Adapted from TurnGPT

from tokenizers import Regex
from tokenizers.normalizers import (
    Lowercase,
    NFD,
    StripAccents,
    Replace,
    Strip,
    Sequence,
)
from transformers import AutoTokenizer
from transformers.tokenization_utils_base import BatchEncoding
from typing import List, Union
import torch
import re

import logging

logger = logging.getLogger(__name__)

TS_TOKENS = {
    "eos_token": "<eot>",
    "pad_token": "<|endoftext|>",
    "additional_special_tokens": ["<emp>",  "<sot>", "<sbc>", "<ebc>", "<sint>", "<eint>"],
    # "additional_special_tokens": ["<emp>", "<eot>", "<ebc>", "<eint>"],
}


class SpokenNormalizer:
    """
    Normalizer (as in the `tokenizers` framework) which removes punctuation, force lowercase, etc
    """

    def __init__(self):
        self.normalizer = SpokenNormalizer.build_normalizer()

    def normalize_string(self, s):
        s = self.add_whitespace_after_punctuation(s)
        return self.normalizer.normalize_str(s)

    def add_whitespace_after_punctuation(self, s):
        """
        Don't know how to do this with the `tokenizers` library.
        So simple regexp for now...

        Without this function:

            "hello,,,there;everybody.whats     how are you?"
            -> "hellothereeverybodywhats how are you" (once decoded)

        With:

            "hello,,,there;everybody.whats     how are you?"
            -> "hello there everybody whats how are you"

        """
        s = re.sub(r"[\,\.\:\;]+(\w+)", r" \1", s)
        return s

    @staticmethod
    def build_normalizer():
        normalizer = Sequence(
            [
                NFD(),
                Lowercase(),
                StripAccents(),
                # punctuation
                Replace(Regex(r'[\.\,\!\?\:\;\)\(\[\]"\-]'), ""),
                Replace(Regex(r"\s\s+"), " "),  # double spaces
                Strip(),
            ]
        )
        return normalizer


class SpokenDialogTokenizer(SpokenNormalizer):
    @property
    def unk_token(self):
        return self._tokenizer.unk_token

    @property
    def unk_token_id(self):
        return self._tokenizer.unk_token_id

    @property
    def eos_token(self):
        return self._tokenizer.eos_token

    @property
    def eos_token_id(self):
        return self._tokenizer.eos_token_id

    @property
    def pad_token(self):
        return self._tokenizer.pad_token_id

    @property
    def pad_token_id(self):
        return self._tokenizer.pad_token_id

    def __init__(
            self,
            pretrained_model_name_or_path: str = "gpt2",
            normalization=True,
            tokens=None,
    ):
        super().__init__()
        self.name_or_path = pretrained_model_name_or_path
        self._tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path, max_model_input_sizes=None
        )
        self.normalization = normalization

        # Set to large number to avoid warnings
        # Manually keep track of your models maximum input length
        self._tokenizer.model_max_length = 1e30

        if tokens is not None:
            TS_TOKENS['additional_special_tokens'] = tokens

        # This goes in logging
        num_added_toks = self._tokenizer.add_special_tokens(TS_TOKENS)

        s = "Tokenizer initialization:\n"
        s += f"\tWe added {num_added_toks} tokens -> Special token map\n"
        for k, v in self._tokenizer.special_tokens_map.items():
            s += f"\t{k}: {v}\n"
        logger.info(s)
        print(s)

    def __repr__(self):
        return self._tokenizer.__repr__()

    def __len__(self):
        return len(self._tokenizer)

    def normalize(self, string: str) -> str:
        if self.normalization:
            return self.normalize_string(string)
        return string

    def __call__(
            self,
            text: Union[str, List[str], List[List[str]]],
            return_token_type_ids: bool = True,
            include_pre_space: bool = False,
            include_end_ts: bool = True,
            **kwargs,
    ) -> BatchEncoding:
        """
        SpokenDialogTokenizer tokenization.

        `text` can be either a String, a List of Strings, or a List of Lists of Strings. The behaviour of
        this function depends on the `single_dialog` flag.

        `text` is String:           representation of entire dialog (including eos_token)
        `text` is List[str]:        representation of turns in a dialog (no eos_tokens)
        `text` is List[List[str]]:  multiple dialogs (lists of strings) (no eos_tokens)

        """

        # List of lists
        if isinstance(text, list) and isinstance(text[0], list):
            ret = {}
            for text_list in text:
                o = self(
                    text_list,
                    include_pre_space=include_pre_space,
                    include_end_ts=include_end_ts,
                )

                for k, v in o.items():
                    if not k in ret:
                        ret[k] = []
                    ret[k].append(v)
            return ret

        # List of strings, a dialog: ['hello', 'hello to you']
        elif isinstance(text, List):
            dialog_string = ""
            if include_pre_space:
                dialog_string = " "
            dialog_string += self.normalize(text[0])
            if len(text) > 1:
                dialog_string += self.eos_token
                for text_string in text[1:-1]:
                    dialog_string += " " + \
                        self.normalize(text_string) + self.eos_token
                dialog_string += " " + self.normalize(text[-1])
            if include_end_ts:
                dialog_string += self.eos_token
            text = dialog_string
        else:
            text = self.normalize(text)

        encoding = self._tokenizer(
            text=text,
            **kwargs,
        )

        return encoding

    def idx_to_tokens(self, ids):
        def list_ids_to_string(ids):
            return [
                self.convert_tokens_to_string(t)
                for t in self.convert_ids_to_tokens(ids)
            ]

        # tokenize keep tokens
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()

        if isinstance(ids, list):
            if isinstance(ids[0], list):
                ret = [list_ids_to_string(ids_list) for ids_list in ids]
            else:
                ret = list_ids_to_string(ids)
        else:
            ret = self.convert_tokens_to_string(
                self.convert_ids_to_tokens(ids))
        return ret

    def pad(self, *args, **kwargs):
        return self._tokenizer.pad(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self._tokenizer.decode(*args, **kwargs)

    def convert_ids_to_tokens(self, *args, **kwargs):
        return self._tokenizer.convert_ids_to_tokens(*args, **kwargs)

    def convert_tokens_to_ids(self, *args, **kwargs):
        return self._tokenizer.convert_tokens_to_ids(*args, **kwargs)

    def convert_tokens_to_string(self, *args, **kwargs):
        return self._tokenizer.convert_tokens_to_string(*args, **kwargs).strip()

    def batch_decode(self, *args, **kwargs):
        return self._tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self._tokenizer.decode(*args, **kwargs)

    def set_padding_side(self, padding_side='left'):
        if padding_side not in {'left', 'right'}:
            return

        self._tokenizer.padding_side = padding_side

    @property
    def special_tokens(self):
        return self._tokenizer.additional_special_tokens
