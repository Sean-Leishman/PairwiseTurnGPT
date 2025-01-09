import logging

import torch
import copy

from data.aligned_process import AlignedProcess, TurnType
from data.base import Process
from data.utils import get_logger
from enum import IntEnum

logger = get_logger(__name__)


class SerialisedProcessType(IntEnum):
    # Push all speaker utterances to end of turn (even when start is prior to the end)
    TurnEnd = 0
    TurnStart = 1  # Ignore all speaker utterances after the other turn starts
    OverlapStart = 2  # End turns on overlaps
    BackchannelStart = 3  # End turns on backchannels
    OverlapBackchannelStart = 4  # End turns on overlaps and backchannels


def to_serialised_process_type(value):
    if value == "turn-end":
        return SerialisedProcessType.TurnEnd
    elif value == "turn-start":
        return SerialisedProcessType.TurnStart
    elif isinstance(value, SerialisedProcessType):
        return value

    raise ValueError(f"SerialisedProcessType {value} not implemented")


logger = logging.getLogger(__name__)


def speaker_start_points(
    token_type_ids, emp_pairs, turn_type=TurnType.BACKCHANNEL, valid_sum=None
):
    token_of_turn_type = (token_type_ids == turn_type).int()
    valid_tokens = torch.logical_and(
        token_of_turn_type, torch.logical_not(emp_pairs)
    ).int()  # get all tokens that are not an eliminated emp pair
    speaker_start_points = (valid_tokens - valid_tokens.roll(1) > 0).int()

    if valid_sum is not None:
        assert (
            torch.sum(speaker_start_points) == valid_sum
        ), f"Sum of speaker_start_points={torch.sum(speaker_start_points)} not equal to valid_sum={valid_sum}"

    local_token_type_ids = token_type_ids.clone()
    local_token_type_ids[speaker_start_points] = turn_type
    local_token_type_ids[torch.logical_not(speaker_start_points)] = TurnType.NONE
    return local_token_type_ids


class SerialisedProcess(Process):
    def __init__(
        self,
        tokenizer,
        end_on=SerialisedProcessType.TurnEnd,
        combine_speaker=False,
        remove_emp_tokens=False,
        remove_special_tokens=False,
        no_overlap=False,
        split_utt=True,
        max_length=256,
        keep_length=64,
        overlap_length=10,
        end_of_utterance_tokens=["<yield>", "<eint>", "<ebc>"],
        yield_int_thresh=0.2,
        yield_overlap_thresh=2.0,
        *args,
        **kwargs,
    ):
        """
        SerialisedProcess is a process that takes a conversation and serialises it into a single stream

        Args:
            tokenizer (Tokenizer): Tokenizer to use for processing the data
            end_on (SerialisedProcessType, optional): Type of serialisation to use. Defaults to SerialisedProcessType.TurnEnd.
            combine_speaker (bool, optional): Whether to serialise the conversation into a single stream. Defaults to False.
            remove_emp_tokens (bool, optional): Whether to remove <emp> tokens from the conversation. Defaults to False.
            no_overlap (bool, optional): Whether to not allow lexical and turn-end token to stack. Defaults to False.
        """
        super().__init__(tokenizer)

        self.tokenizer = tokenizer

        self.end_on = end_on if end_on is not None else SerialisedProcessType.TurnEnd
        self.combine_speaker = combine_speaker
        self.remove_emp_tokens = remove_emp_tokens
        self.remove_special_tokens = remove_special_tokens
        self.no_overlap = no_overlap
        self.split_utt = split_utt
        assert not (
            self.combine_speaker and self.remove_emp_tokens
        ), "Combine speaker and remove emp tokens must not be used together"

        self.max_length = max_length
        self.keep_length = keep_length
        self.overlap_length = overlap_length

        self.end_of_utterance_tokens = end_of_utterance_tokens

        self.emp_token_id = self.tokenizer.convert_tokens_to_ids("<emp>")
        self.end_tokens = torch.tensor(
            [
                self.tokens_dict["eot"],
                self.tokens_dict["yield"],
            ]
        )
        self.turn_type_tensor = torch.tensor([TurnType.NORMAL, TurnType.INTERRUPT])

        # Include all turn-taking behaviour to capture the interruption point
        # for all of these
        self.aligned_processor = AlignedProcess(
            tokenizer,
            include_partial_overlaps=True,
            include_backchannels=True,
            include_overlaps=True,
            split_utt=False,
            end_of_utterance_tokens=self.end_of_utterance_tokens,
            yield_int_thresh=yield_int_thresh,
            yield_overlap_thresh=yield_overlap_thresh,
        )

        logger.info(f"SerialisedProcess initialised with end_on={self.end_on}")

    def __hash__(self):
        return hash(
            (
                self.end_on,
                self.combine_speaker,
                self.remove_emp_tokens,
                self.max_length,
                self.keep_length,
                self.overlap_length,
            )
        )

    def __str__(self):
        end_on = f"end_on={self.end_on}"
        combine_speaker = f"combine_speaker={int(self.combine_speaker)}"
        remove_emp_tokens = f"remove_emp_tokens={int(self.remove_emp_tokens)}"

        max_length = f"max_length={self.max_length}"
        keep_length = f"keep_length={self.keep_length}"
        overlap_length = f"overlap_length={self.overlap_length}"
        return f"SerialisedProcess({end_on},{combine_speaker},{remove_emp_tokens},{max_length},{keep_length},{overlap_length})"

    def config_to_dict(self):
        return {
            "process": "SerialisedProcess",
            "end_on": self.end_on,
            "combine_speaker": self.combine_speaker,
            "remove_emp_tokens": self.remove_emp_tokens,
            "max_length": self.max_length,
            "keep_length": self.keep_length,
            "overlap_length": self.overlap_length,
        }

    def __repr__(self):
        return self.__str__()

    def process(self, datasets=[], in_parallel=True, _misc=0):
        # Process the data
        for parent_ds in datasets:
            dss, start, batch_size = parent_ds
            for ds in dss.get_batch(start, batch_size):
                for conv in self.process_conversation(ds):
                    yield conv

    def process_conversation(self, ds):
        for conv in self.aligned_processor.process_conversation(ds):
            for serialised_turn in self._process(conv):
                if self.remove_special_tokens:
                    serialised_turn["speakerA"] = self._remove_special_tokens(
                        serialised_turn["speakerA"]
                    )
                    if "speakerB" in serialised_turn:
                        serialised_turn["speakerB"] = self._remove_special_tokens(
                            serialised_turn["speakerB"]
                        )

                for split_utt in self._split_utterances([serialised_turn]):
                    yield split_utt

    def _remove_special_tokens(self, data):
        special_tokens = torch.tensor(
            self.tokenizer.convert_tokens_to_ids(self.tokenizer.special_tokens)
            + [self.tokenizer.eos_token_id]
        )
        index = torch.logical_not(torch.isin(data["input_ids"], special_tokens))

        for key in data.keys():
            if isinstance(data[key], torch.Tensor):
                data[key] = data[key][index]
            elif isinstance(data[key], list):
                data[key] = [data[key][i] for i in range(len(data[key])) if index[i]]

        return data

    def _process(self, data):
        turns = data["turns"]

        output = {
            "speakerA": {
                "dialog": "",
                "tokens": [],
                "turn_type_ids": [],
                "turn_end_type_ids": [],
                "timings": [],
                "loss_mask": [],
                "conv_id": [],
            },
            "speakerB": {
                "dialog": "",
                "tokens": [],
                "turn_type_ids": [],
                "turn_end_type_ids": [],
                "timings": [],
                "loss_mask": [],
                "conv_id": [],
            },
        }
        output["speakerA"] = self._remove_turns_of_type(
            data["speakerA"], TurnType.BACKCHANNEL
        )
        output["speakerB"] = self._remove_turns_of_type(
            data["speakerB"], TurnType.BACKCHANNEL
        )
        output["speakerA"] = self._remove_turns_of_type(
            data["speakerA"], TurnType.OVERLAP
        )
        output["speakerB"] = self._remove_turns_of_type(
            data["speakerB"], TurnType.OVERLAP
        )

        output["speakerA"]["interrupt_points"] = self._add_speaker_start_points(
            output["speakerB"], output["speakerA"]
        )
        output["speakerB"]["interrupt_points"] = self._add_speaker_start_points(
            output["speakerA"], output["speakerB"]
        )

        output["speakerA"], output["speakerB"] = self._stack_turns(
            output["speakerA"], output["speakerB"], turns
        )

        output["speakerA"], output["speakerB"] = self._remove_emp_pairs(
            output["speakerA"], output["speakerB"]
        )

        if self.no_overlap:
            output["speakerA"], output["speakerB"] = self._push_overlap_of_turns(
                output["speakerA"], output["speakerB"]
            )

        if self.combine_speaker:
            output["speakerA"] = self._serialise_conversation(
                output["speakerA"], output["speakerB"]
            )
            output.pop("speakerB")

        if self.remove_emp_tokens:
            output["speakerA"] = self._remove_all_emp_tokens(output["speakerA"])
            if "speakerB" in output:
                output["speakerB"] = self._remove_all_emp_tokens(output["speakerB"])

        output["speakerA"]["loss_mask"] = torch.ones_like(
            output["speakerA"]["input_ids"]
        )
        if "speakerB" in output:
            output["speakerB"]["loss_mask"] = torch.ones_like(
                output["speakerB"]["input_ids"]
            )

        output["speakerA"]["conv_id"] = data["speakerA"]["conv_id"]
        output["turns"] = self._parse_turns(turns)

        yield output

    def _parse_turns(
        self, turns, remove_turn_types={TurnType.OVERLAP, TurnType.BACKCHANNEL}
    ):
        """
        Parse the turns to remove OVERLAP and BACKCHANNEL turns
        """
        return [turn for turn in turns if turn["turn_type"] not in remove_turn_types]

    def _add_speaker_start_points(
        self, current_speaker, other_speaker, shift_for_emp=True
    ):
        """
        Add interruption points to the conversation.
        This is done by finding the start of any turn and labelling it as
        its turn type (NORMAL, INTERRUPTION, BACKCHANNEL, OVERLAP).
        All other parts of the turn have no label.

        Shifts turn type to the right when there is an emp pair;
            which would be removed later.
        This is done so that we do not remove the start of the turn point
        and we move it to the right as that is the point of interruption where
        the next utterance begins
        """
        emp_pair = torch.zeros_like(current_speaker["input_ids"])
        if shift_for_emp:
            emp_pair = (current_speaker["input_ids"] == self.emp_token_id) & (
                other_speaker["input_ids"] == self.emp_token_id
            )

        token_type_ids = current_speaker["token_type_ids"]

        backchannel_interruption = speaker_start_points(
            token_type_ids, emp_pair, turn_type=TurnType.BACKCHANNEL
        )
        overlap_interruption = speaker_start_points(
            token_type_ids, emp_pair, turn_type=TurnType.OVERLAP
        )
        interruption = speaker_start_points(
            token_type_ids, emp_pair, turn_type=TurnType.INTERRUPT
        )
        normal_interruption = speaker_start_points(
            token_type_ids, emp_pair, turn_type=TurnType.NORMAL
        )

        return (
            backchannel_interruption
            + overlap_interruption
            + interruption
            + normal_interruption
        )

    def _remove_turns_of_type(self, speaker, turn_type):
        """
        Remove all turns of a given type from the conversation by setting tokens
        to <emp>.
        """
        remove_idxs = speaker["token_type_ids"] == turn_type
        speaker["input_ids"][remove_idxs] = self.emp_token_id
        return speaker

    def _remove_emp_pairs(self, speakerA, speakerB):
        """
        Remove pairs of turns where both speakers have been removed
        """
        remove_idxs = (speakerA["input_ids"] == self.emp_token_id) & (
            speakerB["input_ids"] == self.emp_token_id
        )
        for key in speakerA.keys():
            if isinstance(speakerA[key], torch.Tensor):
                assert (
                    speakerA[key].shape[0] == remove_idxs.shape[0]
                ), f"Shape mismatch for key={key} with shapes={speakerA[key].shape} and {remove_idxs.shape}"
                assert (
                    speakerB[key].shape[0] == remove_idxs.shape[0]
                ), f"Shape mismatch for key={key} with shapes={speakerB[key].shape} and {remove_idxs.shape}"

                if speakerA[key].dim() == 2:
                    speakerA[key] = speakerA[key][~remove_idxs, :]
                    speakerB[key] = speakerB[key][~remove_idxs, :]
                elif speakerA[key].dim() == 1:
                    speakerA[key] = speakerA[key][~remove_idxs]
                    speakerB[key] = speakerB[key][~remove_idxs]
                else:
                    raise ValueError(f"Speaker key {key} has dim={speakerA[key].dim()}")

            elif isinstance(speakerA[key], list):
                speakerA[key] = [
                    speakerA[key][i]
                    for i in range(len(speakerA[key]))
                    if not remove_idxs[i]
                ]
                speakerB[key] = [
                    speakerB[key][i]
                    for i in range(len(speakerB[key]))
                    if not remove_idxs[i]
                ]

        return speakerA, speakerB

    def _get_turn_indexes(
        self,
        speakerA,
        speakerB,
        turns,
        turn_types={TurnType.NORMAL, TurnType.INTERRUPT},
    ):
        turn_boundariesA = [
            ("A", turn["start"], turn["end"], turn)
            for turn in turns
            if turn["turn_type"] in turn_types and turn["speaker"] == "A"
        ]
        turn_boundariesB = [
            ("B", turn["start"], turn["end"], turn)
            for turn in turns
            if turn["turn_type"] in turn_types and turn["speaker"] == "B"
        ]
        turns_merged = sorted(turn_boundariesA + turn_boundariesB, key=lambda x: x[1])

        turn_idxs = []
        turn_idx = 0
        start = -1

        idx = 0
        list_timing = list(zip(speakerA["timings"], speakerB["timings"]))
        while idx < len(list_timing):
            timingA, timingB = list_timing[idx]
            if (
                timingA[0] == turns_merged[turn_idx][1]
                and turns_merged[turn_idx][0] == "A"
            ):
                start = idx
            elif (
                timingB[0] == turns_merged[turn_idx][1]
                and turns_merged[turn_idx][0] == "B"
            ):
                start = idx

            if (
                timingA[1] == turns_merged[turn_idx][2]
                and turns_merged[turn_idx][0] == "A"
            ):
                turn_idxs.append(("A", start, idx, turns_merged[turn_idx][3]))
                turn_idx += 1
                idx = max(start, 0)
                start = -1
            elif (
                timingB[1] == turns_merged[turn_idx][2]
                and turns_merged[turn_idx][0] == "B"
            ):
                turn_idxs.append(("B", start, idx, turns_merged[turn_idx][3]))
                turn_idx += 1
                idx = max(start, 0)
                start = -1
            else:
                idx += 1

            if turn_idx >= len(turns_merged):
                break

        assert all(
            [-1 < turn[1] and turn[1] <= turn[2] for turn in turn_idxs]
        ), f"Turns have negative length {turn_idxs}) conv_id={speakerA['conv_id']} with timings={[(x[1], x[2], list_timing[x[1]], list_timing[x[2]]) for x in turn_idxs if x[1] < 0 or x[2] < 0]}"
        return turn_idxs

    def _add_emp_tokens(self, result, index, num_tokens, key, value):
        if key not in result:
            raise ValueError(f"Key {key} not in result with keys={result.keys()}")
        if isinstance(result[key], torch.Tensor):
            result[key] = torch.cat(
                (
                    result[key][:index],
                    torch.tensor([value for _ in range(num_tokens)]),
                    result[key][index:],
                )
            )
            return result

        result[key] = (
            result[key][:index]
            + [value for _ in range(num_tokens)]
            + result[key][index:]
        )
        return result

    def _replace_with_emp_tokens(self, result, index, num_tokens, key, value):
        if key not in result:
            raise ValueError(f"Key {key} not in result with keys={result.keys()}")
        if isinstance(result[key], torch.Tensor):
            result[key][index : index + num_tokens] = value
            return result
        else:
            result[key][index : index + num_tokens] = [value for _ in range(num_tokens)]
            return result

    def _swap_tokens(self, result, src, target, key):
        if key not in result:
            raise ValueError(f"Key {key} not in result with keys={result.keys()}")
        if isinstance(result[key], torch.Tensor):
            result[key][src], result[key][target] = (
                result[key][target].clone(),
                result[key][src].clone(),
            )
        else:
            result[key][src], result[key][target] = (
                result[key][target],
                result[key][src],
            )

        return result

    def _stack_turns(self, speakerA, speakerB, turns):
        """
        Stack the turns of the conversation so that all turns of a speaker occur
        one after another.

        Note: We require a separate function for handling the case of starting turns
        at the interruption point
        """
        if self.end_on == SerialisedProcessType.TurnEnd:
            return self._stack_turns_on_end(speakerA, speakerB, turns)
        elif self.end_on == SerialisedProcessType.TurnStart:
            return self._stack_turns_on_start(speakerA, speakerB, turns)

        raise NotImplementedError(
            f"SerialisedProcessType {self.end_on} not implemented for _stack_turns"
        )

    def _stack_turns_on_start(self, speakerA, speakerB, turns):
        """
        Stack the turns of the conversation so that any interruption that results
        in a turn shift is shown at the correct token.
        Replaces lexical tokens with <emp> tokens but leaves the final special token
        which we assume exists
        This token is then shifted to the new turn-end.
        """
        turn_idxs = self._get_turn_indexes(speakerA, speakerB, turns)

        resultA = copy.deepcopy(speakerA)
        resultB = copy.deepcopy(speakerB)

        for idx in range(len(turn_idxs) - 1):
            curr_turn = turn_idxs[idx]
            next_turn = turn_idxs[idx + 1]

            if curr_turn[2] - next_turn[1] >= 0:
                if next_turn[0] == "A":
                    temp_resultA = resultA
                    temp_resultB = resultB
                elif next_turn[0] == "B":
                    temp_resultA = resultB
                    temp_resultB = resultA
                else:
                    raise ValueError(f"Speaker {next_turn[0]} not recognised")

                replace_emp_tokens = 1 + curr_turn[2] - next_turn[1]
                temp_resultB = self._replace_with_emp_tokens(
                    temp_resultB,
                    next_turn[1],
                    replace_emp_tokens,
                    "input_ids",
                    self.emp_token_id,
                )
                temp_resultB = self._replace_with_emp_tokens(
                    temp_resultB,
                    next_turn[1],
                    replace_emp_tokens,
                    "token_type_ids",
                    TurnType.NONE,
                )
                temp_resultB = self._replace_with_emp_tokens(
                    temp_resultB,
                    next_turn[1],
                    replace_emp_tokens,
                    "other_token_type_ids",
                    TurnType.NONE,
                )
                temp_resultB = self._replace_with_emp_tokens(
                    temp_resultB, next_turn[1], replace_emp_tokens, "speaker_ids", 0
                )
                temp_resultB = self._replace_with_emp_tokens(
                    temp_resultB,
                    next_turn[1],
                    replace_emp_tokens,
                    "interrupt_points",
                    TurnType.NONE,
                )
                temp_resultB = self._replace_with_emp_tokens(
                    temp_resultB, next_turn[1], replace_emp_tokens, "attention_mask", 1
                )
                temp_resultB = self._replace_with_emp_tokens(
                    temp_resultB, next_turn[1], replace_emp_tokens, "turn_ids", 0
                )
                temp_resultB = self._replace_with_emp_tokens(
                    temp_resultB, next_turn[1], replace_emp_tokens, "loss_mask", 1
                )

                # Swap all aside from the timings which may mean that we have
                # random (-1,-1) tokens but these should be non-lexical and therefore irrlevant
                # Can fix easily by just shifting everything down
                temp_resultB = self._swap_tokens(
                    temp_resultB, curr_turn[2] + 1, next_turn[1], "input_ids"
                )
                temp_resultB = self._swap_tokens(
                    temp_resultB, curr_turn[2] + 1, next_turn[1], "token_type_ids"
                )
                temp_resultB = self._swap_tokens(
                    temp_resultB, curr_turn[2] + 1, next_turn[1], "other_token_type_ids"
                )
                temp_resultB = self._swap_tokens(
                    temp_resultB, curr_turn[2] + 1, next_turn[1], "speaker_ids"
                )
                temp_resultB = self._swap_tokens(
                    temp_resultB, curr_turn[2] + 1, next_turn[1], "interrupt_points"
                )
                temp_resultB = self._swap_tokens(
                    temp_resultB, curr_turn[2] + 1, next_turn[1], "attention_mask"
                )
                temp_resultB = self._swap_tokens(
                    temp_resultB, curr_turn[2] + 1, next_turn[1], "turn_ids"
                )
                temp_resultB = self._swap_tokens(
                    temp_resultB, curr_turn[2] + 1, next_turn[1], "loss_mask"
                )

        return resultA, resultB

    def _stack_turns_on_end(self, speakerA, speakerB, turns):
        turn_idxs = self._get_turn_indexes(speakerA, speakerB, turns)
        if not all([len(x) == 4 for x in turn_idxs]):
            pass

        resultA = copy.deepcopy(speakerA)
        resultB = copy.deepcopy(speakerB)

        for idx in range(len(turn_idxs) - 1):
            curr_turn = turn_idxs[idx]
            next_turn = turn_idxs[idx + 1]

            if len(next_turn) > 4 or len(curr_turn) > 4:
                pass

            if curr_turn[2] - next_turn[1] >= 0:
                add_emp_tokens = curr_turn[2] + 1 - next_turn[1]
                if next_turn[0] == "A":
                    temp_resultA = resultA
                    temp_resultB = resultB
                elif next_turn[0] == "B":
                    temp_resultA = resultB
                    temp_resultB = resultA
                else:
                    raise ValueError(f"Speaker {next_turn[0]} not recognised")

                temp_resultA = self._add_emp_tokens(
                    temp_resultA,
                    next_turn[1],
                    add_emp_tokens,
                    "input_ids",
                    self.emp_token_id,
                )
                temp_resultA = self._add_emp_tokens(
                    temp_resultA, next_turn[1], add_emp_tokens, "timings", (-1, -1)
                )

                temp_resultA = self._add_emp_tokens(
                    temp_resultA,
                    next_turn[1],
                    add_emp_tokens,
                    "token_type_ids",
                    next_turn[3]["turn_type"],
                )
                temp_resultA = self._add_emp_tokens(
                    temp_resultA,
                    next_turn[1],
                    add_emp_tokens,
                    "other_token_type_ids",
                    next_turn[3]["turn_end_type"],
                )
                temp_resultA = self._add_emp_tokens(
                    temp_resultA,
                    next_turn[1],
                    add_emp_tokens,
                    "speaker_ids",
                    0 if next_turn[0] == "A" else 1,
                )
                temp_resultA = self._add_emp_tokens(
                    temp_resultA,
                    next_turn[1],
                    add_emp_tokens,
                    "interrupt_points",
                    TurnType.NONE,
                )
                temp_resultA = self._add_emp_tokens(
                    temp_resultA, next_turn[1], add_emp_tokens, "attention_mask", 1
                )
                temp_resultA = self._add_emp_tokens(
                    temp_resultA,
                    next_turn[1],
                    add_emp_tokens,
                    "turn_ids",
                    next_turn[3]["id"],
                )
                temp_resultA = self._add_emp_tokens(
                    temp_resultA, next_turn[1], add_emp_tokens, "loss_mask", 1
                )

                temp_resultB = self._add_emp_tokens(
                    temp_resultB,
                    curr_turn[2] + 2,
                    add_emp_tokens,
                    "input_ids",
                    self.emp_token_id,
                )
                temp_resultB = self._add_emp_tokens(
                    temp_resultB, curr_turn[2] + 2, add_emp_tokens, "timings", (-1, -1)
                )
                temp_resultB = self._add_emp_tokens(
                    temp_resultB,
                    curr_turn[2] + 2,
                    add_emp_tokens,
                    "token_type_ids",
                    curr_turn[3]["turn_type"],
                )
                temp_resultB = self._add_emp_tokens(
                    temp_resultB,
                    curr_turn[2] + 2,
                    add_emp_tokens,
                    "other_token_type_ids",
                    curr_turn[3]["turn_end_type"],
                )
                temp_resultB = self._add_emp_tokens(
                    temp_resultB,
                    curr_turn[2] + 2,
                    add_emp_tokens,
                    "speaker_ids",
                    0 if curr_turn[0] == "A" else 1,
                )
                temp_resultB = self._add_emp_tokens(
                    temp_resultB,
                    curr_turn[2] + 2,
                    add_emp_tokens,
                    "interrupt_points",
                    TurnType.NONE,
                )
                temp_resultB = self._add_emp_tokens(
                    temp_resultB, curr_turn[2] + 2, add_emp_tokens, "attention_mask", 1
                )
                temp_resultB = self._add_emp_tokens(
                    temp_resultB,
                    curr_turn[2] + 2,
                    add_emp_tokens,
                    "turn_ids",
                    curr_turn[3]["id"],
                )
                temp_resultB = self._add_emp_tokens(
                    temp_resultB, curr_turn[2] + 2, add_emp_tokens, "loss_mask", 1
                )

                turn_idxs = [
                    (
                        turn[0],
                        turn[1] + add_emp_tokens,
                        turn[2] + add_emp_tokens,
                        turn[3],
                    )
                    for turn in turn_idxs
                ]

        return resultA, resultB

    def _push_overlap_of_turns(self, speakerA, speakerB):
        """
        Prevent overlap of lexical and turn-end tokens by pushing the
        lexcial token past the end of the turn

        Called by self._serialise_conversation so that serialisation
        can easily be done by replacing <emp> tokens with the populated
        speaker channel
        """
        for idx in range(len(speakerA["input_ids"]) - 1, 0, -1):
            speaker_id = 0
            if (
                speakerA["input_ids"][idx] in self.end_tokens
                and speakerB["input_ids"][idx] != self.emp_token_id
            ):
                temp_speakerA = speakerA
                temp_speakerB = speakerB
            elif (
                speakerB["input_ids"][idx] in self.end_tokens
                and speakerA["input_ids"][idx] != self.emp_token_id
            ):
                temp_speakerA = speakerB
                temp_speakerB = speakerA

                speaker_id = 1
            else:
                continue

            for key in temp_speakerA.keys():
                if key == "input_ids":
                    value = torch.tensor([self.emp_token_id])
                elif key in {
                    "token_type_ids",
                    "other_token_type_ids",
                    "interrupt_points",
                }:
                    value = torch.tensor([TurnType.NONE])
                elif key == "speaker_ids":
                    value = torch.tensor([speaker_id])
                elif key in "attention_mask":
                    value = torch.tensor([1])
                elif key == "timings":
                    value = torch.tensor([(-1, -1)])
                elif key == "turn_ids":
                    value = torch.tensor([0])
                elif key == "loss_mask":
                    value = torch.tensor([1])
                else:
                    continue

                temp_speakerA[key] = torch.cat(
                    (
                        temp_speakerA[key][: idx + 1],
                        value,
                        temp_speakerA[key][idx + 1 :],
                    )
                )

            for key in temp_speakerB.keys():
                if key == "input_ids":
                    value = torch.tensor([self.emp_token_id])
                elif key in {
                    "token_type_ids",
                    "other_token_type_ids",
                    "interrupt_points",
                }:
                    value = torch.tensor([TurnType.NONE])
                elif key == "speaker_ids":
                    value = torch.tensor([speaker_id])
                elif key == "attention_mask":
                    value = torch.tensor([1])
                elif key == "timings":
                    value = torch.tensor([(-1, -1)])
                elif key == "turn_ids":
                    value = torch.tensor([0])
                elif key == "loss_mask":
                    value = torch.tensor([1])
                else:
                    continue

                temp_speakerB[key] = torch.cat(
                    (
                        temp_speakerB[key][:idx],
                        value,
                        temp_speakerB[key][idx:],
                    )
                )

        assert (
            len(speakerA["input_ids"]) == len(speakerB["input_ids"])
        ), f"Length of speakerA={len(speakerA['input_ids'])} not equal to speakerB={len(speakerB['input_ids'])}"
        assert (
            len(speakerA["input_ids"]) == len(speakerA["token_type_ids"])
        ), f"Length of speakerA={len(speakerA['input_ids'])} not equal to speakerA token_type_ids={len(speakerA['token_type_ids'])}"

        return speakerA, speakerB

    def _serialise_conversation(self, speakerA, speakerB):
        """
        Serialise the conversation into a single string
        """
        speakerA, speakerB = self._push_overlap_of_turns(speakerA, speakerB)
        for idx in range(len(speakerA["input_ids"])):
            if (
                speakerA["input_ids"][idx] == self.emp_token_id
                and speakerB["input_ids"][idx] != self.emp_token_id
            ):
                for key in speakerA.keys():
                    if isinstance(speakerA[key], str):
                        continue
                    speakerA[key][idx] = speakerB[key][idx]
                speakerA["input_ids"][idx] = speakerB["input_ids"][idx]
            elif (
                speakerA["input_ids"][idx] == self.emp_token_id
                and speakerB["input_ids"][idx] == self.emp_token_id
            ):
                raise ValueError(
                    f"(<emp>,{self.emp_token_id})=({self.tokenizer.convert_ids_to_tokens(self.emp_token_id)}) token present in both data streams for conv_id={speakerA['conv_id']}"
                )

        return speakerA

    def _remove_all_emp_tokens(self, speaker):
        """
        Remove all <emp> tokens from the conversation
        """
        remove_idxs = speaker["input_ids"] == self.emp_token_id
        for key in speaker.keys():
            if isinstance(speaker[key], torch.Tensor):
                speaker[key] = speaker[key][~remove_idxs]
            elif isinstance(speaker[key], list):
                speaker[key] = [
                    speaker[key][i]
                    for i in range(len(speaker[key]))
                    if not remove_idxs[i]
                ]

        return speaker

    def _split_utterances(self, data):
        """
        Split the utterances into individual turns
        """
        result = []
        if not self.split_utt:
            return data

        for conv in data:
            start_idx = 0
            end_idx = min(
                self.max_length + start_idx, len(conv["speakerA"]["input_ids"])
            )

            while end_idx <= len(conv["speakerA"]["input_ids"]):
                if end_idx - start_idx < self.keep_length:
                    end_idx = self.overlap_length
                    break

                temp = {}
                for speaker, value in conv.items():
                    if "speaker" not in speaker:
                        continue

                    temp[speaker] = {}
                    for key, val in value.items():
                        if isinstance(val, torch.Tensor):
                            temp[speaker][key] = val[start_idx:end_idx].clone()
                        elif isinstance(val, list):
                            temp[speaker][key] = copy.deepcopy(val[start_idx:end_idx])
                        else:
                            temp[speaker][key] = val

                temp["turns"] = conv["turns"]
                result.append(copy.deepcopy(temp))

                if end_idx == len(conv["speakerA"]["input_ids"]):
                    break

                start_idx = end_idx - self.overlap_length
                end_idx = min(
                    len(conv["speakerA"]["input_ids"]), self.max_length + start_idx
                )

        return result
