import torch
import pickle
import logging
import pprint

from base import Process, Dialog, Word, Turn, TurnType, TurnEndType
from data.utils import get_logger

from itertools import zip_longest

logger = get_logger(__name__, level=logging.DEBUG)


def handle_channel(
    turnA: Word | None,
    curr_turnA: Turn,
    turnB: Word | None,
    curr_turnB: Turn,
    yield_int_thresh=0.1,
) -> tuple[Turn, Turn]:
    if turnA is None:
        raise ValueError("turnA is None")
    if turnB is None:
        raise ValueError("turnB is None")

    if turnB[0].start + yield_int_thresh < turnA[-1].end:
        # turnB overlapped by turnA
        if turnB[-1].end < turnA[-1].end:
            pass

        curr_turnA.update(
            word=curr_turnA.word + " ".join([turn.word.lower() for turn in turnA]),
            turn_end_type=TurnEndType.YIELD,
            start=turnA[0].start,
            end=turnA[-1].end,
            other_next_id=curr_turnB.id,
        )

        curr_turnB.update(turn_type=TurnType.INTERRUPT, other_prev_id=curr_turnA.id)
    else:
        new_turn_type = (
            TurnType.NORMAL
            if curr_turnA.turn_type == TurnType.NONE
            else curr_turnA.turn_type
        )
        curr_turnA.update(
            word=curr_turnA.word + " ".join([turn.word.lower() for turn in turnA]),
            turn_type=new_turn_type,
            start=turnA[0].start,
            end=turnA[-1].end,
            other_next_id=curr_turnB.id,
        )
        curr_turnB.update(other_prev_id=curr_turnA.id)

    return curr_turnA, curr_turnB


def insert_bc_channel(
    dialog: Dialog,
    turns: list[Turn],
    bcs: Dialog,
    bc_token: int | None = None,
    speaker="A",
) -> tuple[Dialog, list[Turn]]:
    turn_iterator = enumerate(turns + [None])
    turn_idx, turn = next(turn_iterator)

    bc_iterator = enumerate(bcs + [None])
    bc_idx, bc = next(bc_iterator)

    new_turns = []
    new_dialog = []
    while turn is not None:
        if bc is not None and bc[0]["start"] < turn.start:
            assert bc[-1]["end"] < turn["end"]

            bc_word = (
                "<bc>"
                if bc_token is not None
                else " ".join(x["word"].lower() for x in bc)
            )

            new_turn = Turn(
                word=bc_word,
                turn_type=TurnType.BACKCHANNEL,
                turn_end_type=TurnEndType.NONE,
                start=bc[0]["start"],
                end=bc[-1]["end"],
                speaker=speaker,
                conv_id=bc[0]["conv_id"],
                turn_index=len(new_turns),
            )
            new_turns.append(new_turn)

            if bc_token is not None:
                bcs[bc_idx][0]["word"] = bc_word
                bcs[bc_idx][0]["tokens"] = bc_token
                bcs[bc_idx][0]["end"] = bc[-1]["end"]
                if len(bcs[bc_idx]) > 1:
                    bcs[bc_idx] = bcs[bc_idx][:1]

            new_dialog.append(bcs[bc_idx])
            bc_idx, bc = next(bc_iterator)

            continue

        new_dialog.append(dialog[turn_idx])
        new_turns.append(turn)
        turn_idx, turn = next(turn_iterator)

    while bc is not None:
        bc_word = (
            "<bc>" if bc_token is not None else " ".join(x["word"].lower() for x in bc)
        )

        new_turn = Turn(
            word=bc_word,
            turn_type=TurnType.BACKCHANNEL,
            turn_end_type=TurnEndType.NONE,
            start=bc[0]["start"],
            end=bc[-1]["end"],
            speaker=speaker,
            conv_id=bc[0]["conv_id"],
            turn_index=len(new_turns),
        )
        new_turns.append(new_turn)

        if bc_token is not None:
            bcs[bc_idx][0]["word"] = bc_word
            bcs[bc_idx][0]["tokens"] = bc_token
            bcs[bc_idx][0]["end"] = bc[-1]["end"]
            if len(bcs[bc_idx]) > 1:
                bcs[bc_idx] = bcs[bc_idx][:1]

        new_dialog.append(bcs[bc_idx])
        bc_idx, bc = next(bc_iterator)

    return new_dialog, new_turns


def insert_overlap_channel(
    dialog: Dialog, turns: list[Turn], overlaps: Dialog, speaker="A"
) -> tuple[Dialog, list[Turn]]:
    turn_iterator = enumerate(turns + [None])
    overlap_iterator = enumerate(overlaps + [None])

    turn_idx, turn = next(turn_iterator)
    _, overlap = next(overlap_iterator)

    new_dialog, new_turns = [], []

    while turn is not None:
        if overlap is not None and turn["start"] > overlap[0]["start"]:
            assert overlap[-1]["end"] < turn["end"]

            new_turns.append(
                Turn(
                    word=" ".join(x["word"].lower() for x in overlap),
                    turn_type=TurnType.OVERLAP,
                    turn_end_type=TurnEndType.NONE,
                    start=overlap[0]["start"],
                    end=overlap[-1]["end"],
                    speaker=speaker,
                    conv_id=overlap[0]["conv_id"],
                    turn_index=len(new_turns),
                )
            )
            new_dialog.append(overlap)

            _, overlap = next(overlap_iterator)
            continue

        new_dialog.append(dialog[turn_idx])
        new_turns.append(turn)
        turn_idx, turn = next(turn_iterator)

    while overlap is not None:
        new_turns.append(
            Turn(
                word=" ".join(x["word"].lower() for x in overlap),
                turn_type=TurnType.OVERLAP,
                turn_end_type=TurnEndType.NONE,
                start=overlap[0]["start"],
                end=overlap[-1]["end"],
                speaker=speaker,
                conv_id=overlap[0]["conv_id"],
                turn_index=len(new_turns),
            )
        )
        new_dialog.append(overlap)
        _, overlap = next(overlap_iterator)

    return new_dialog, new_turns


def update_turn_to_timing(turn_idx: int, turns: list[Turn], start: float, end: float):
    for i in range(turn_idx, len(turns)):
        if turns[i].start <= start and turns[i].end >= end:
            return i

    assert False, (turns, start, end)


def update_turn_endings(
    turnsA: list[Turn], turnsB: list[Turn], yield_overlap_thresh=2.0
):
    turnsA_idx = 0

    for turnB in turnsB:
        if turnB.turn_type != TurnType.OVERLAP:
            continue

        turnsA_idx = update_turn_to_timing(turnsA_idx, turnsA, turnB.start, turnB.end)
        if turnsA[turnsA_idx].end - turnB.end < yield_overlap_thresh:
            turnsA[turnsA_idx].turn_end_type = TurnEndType.YIELD

    return turnsA


class AlignedProcess(Process):
    def __init__(
        self,
        tokenizer,
        include_partial_overlaps=False,
        include_backchannels=False,
        include_overlaps=False,
        include_bc_token=False,
        filter_special_tokens=[],
        yield_int_thresh=0.1,
        yield_overlap_thresh=0.5,
        max_length=256,
        keep_length=64,
        overlap_length=10,
        split_utt=False,
        end_of_utterance_tokens=["<yield>", "<eint>", "<ebc>"],
        *args,
        **kwargs,
    ):
        super().__init__(tokenizer)
        self.include_partial_overlaps = include_partial_overlaps

        self.include_backchannels = include_backchannels
        self.include_overlaps = include_overlaps
        self.include_bc_token = include_bc_token

        self.filter_special_tokens = filter_special_tokens

        self.include_yield_token = "<yield>" in end_of_utterance_tokens
        self.yield_int_thresh = yield_int_thresh
        self.yield_overlap_thresh = yield_overlap_thresh

        self.max_length = max_length
        self.keep_length = keep_length
        self.overlap_length = overlap_length

        self.split_utt = split_utt

        logger.info("Initialized processor")

    def __hash__(self):
        return hash(
            (
                self.include_partial_overlaps,
                self.include_backchannels,
                self.include_overlaps,
                self.include_bc_token,
                self.yield_int_thresh,
                self.yield_overlap_thresh,
                self.split_utt,
                self.max_length,
                self.keep_length,
                self.overlap_length,
            )
        )

    def __str__(self):
        include_partial_overlaps = "include_partial_overlap=" + str(
            int(self.include_partial_overlaps)
        )
        include_backchannels = "include_backchannels=" + str(
            int(self.include_backchannels)
        )
        include_overlaps = "include_overlaps=" + str(int(self.include_overlaps))
        include_bc_token = "include_bc_token=" + str(int(self.include_bc_token))

        yield_int_thresh = f"yield_int_thresh={self.yield_int_thresh}"
        yield_overlap_thresh = f"yield_overlap_thresh={self.yield_overlap_thresh}"

        split_utt = f"split_utt={int(self.split_utt)}"
        max_length = f"max_length={self.max_length}"
        keep_length = f"keep_length={self.keep_length}"
        overlap_length = f"overlap_length={self.overlap_length}"

        return f"AlignedProcess({include_partial_overlaps},{include_backchannels},{include_overlaps},{include_bc_token},{yield_int_thresh},{yield_overlap_thresh},{split_utt},{max_length},{keep_length},{overlap_length})"

    def config_to_dict(self):
        return {
            "process": "AlignedProcess",
            "include_partial_overlaps": self.include_partial_overlaps,
            "include_backchannels": self.include_backchannels,
            "include_overlaps": self.include_overlaps,
            "include_bc_token": self.include_bc_token,
            "yield_int_thresh": self.yield_int_thresh,
            "yield_overlap_thresh": self.yield_overlap_thresh,
            "split_utt": self.split_utt,
            "max_length": self.max_length,
            "keep_length": self.keep_length,
            "overlap_length": self.overlap_length,
        }

    def __repr__(self):
        return self.__str__()

    def process(self, datasets=[], _misc=0):
        for parent_ds in datasets:
            dss, start, batch_size = parent_ds

            for ds in dss.get_batch(start, batch_size):
                generator = self.process_conversation(ds)
                for item in generator:
                    if item is not None:
                        yield item

    def process_conversation(self, ds):
        dataset = ds
        map_output_keys = {
            "tokens": "input_ids",
            "turn_ids": "turn_ids",
            "turn_type_ids": "token_type_ids",
            "turn_end_type_ids": "other_token_type_ids",
            "timings": "timings",
            "conv_id": "conv_id",
        }
        if (
            len(dataset["dialog"]["speakerA"]) == 0
            or len(dataset["dialog"]["speakerB"]) == 0
        ):
            logger.warning(f"Empty dialog {dataset['dialog']}")
            return

        output = {
            "speakerA": {
                "dialog": "",
                "tokens": [],
                "turn_ids": [],
                "turn_type_ids": [],
                "turn_end_type_ids": [],
                "timings": [],
                "loss_mask": [],
                "conv_id": dataset["dialog"]["speakerA"][0]["conv_id"],
            },
            "speakerB": {
                "dialog": "",
                "tokens": [],
                "turn_ids": [],
                "turn_type_ids": [],
                "turn_end_type_ids": [],
                "timings": [],
                "loss_mask": [],
                "conv_id": dataset["dialog"]["speakerB"][0]["conv_id"],
            },
        }

        assert len(dataset["dialog"]) == 2, f"{dataset['dialog']}"

        if (
            len(dataset["dialog"]["speakerA"]) == 0
            or len(dataset["dialog"]["speakerB"]) == 0
        ):
            logger.warning(f"Empty dialog {dataset['dialog']}")
            return

        dialogA, dialogB, turnsA, turnsB = self._init_turns(dataset)
        output = self._align_tokens(dialogA, dialogB, output)

        output["speakerA"], output["speakerB"] = self._add_ts_token_dialog(
            output["speakerA"], output["speakerB"], turnsA, turnsB
        )

        output["speakerA"] = self._add_ts_token_state(output["speakerA"], turnsA)
        output["speakerB"] = self._add_ts_token_state(output["speakerB"], turnsB)
        new_output = {}
        for speaker in ["speakerA", "speakerB"]:
            new_output[speaker] = {}
            for key in output[speaker].keys():
                if key == "conv_id":
                    new_output[speaker]["conv_id"] = output[speaker][key]
                elif key == "dialog":
                    continue
                else:
                    new_key = map_output_keys.get(key, None)
                    if new_key is not None:
                        new_output[speaker][new_key] = torch.tensor(
                            output[speaker][key]
                        )

            new_output[speaker]["attention_mask"] = torch.ones_like(
                new_output[speaker]["input_ids"]
            )
            new_output[speaker]["speaker_ids"] = torch.where(
                torch.ne(new_output[speaker]["token_type_ids"], TurnType.NONE),
                torch.tensor(
                    self.tokens_dict[speaker],
                    device=new_output[speaker]["input_ids"].device,
                ),
                torch.tensor(0, device=new_output[speaker]["input_ids"].device),
            )
            new_output[speaker]["loss_mask"] = torch.ones_like(
                new_output[speaker]["input_ids"]
            )

        output = new_output
        assert (
            output["speakerA"]["input_ids"].shape
            == output["speakerB"]["input_ids"].shape
        ), f"not matching shape {output['speakerA']['input_ids'].shape} == {output['speakerB']['input_ids'].shape}"
        assert (
            output["speakerA"]["input_ids"].shape
            == output["speakerA"]["token_type_ids"].shape
        ), f"{output['speakerA']['input_ids'].shape} {output['speakerA']['token_type_ids'].shape}"

        output_keys = {
            "input_ids",
            "turn_ids",
            "token_type_ids",
            "other_token_type_ids",
            "attention_mask",
            "speaker_ids",
            "timings",
            "loss_mask",
            "conv_id",
        }
        diff_keysA = set(output["speakerA"].keys()) - output_keys
        diff_keysB = set(output["speakerB"].keys()) - output_keys

        for key in diff_keysA:
            del output["speakerA"][key]
        for key in diff_keysB:
            del output["speakerB"][key]

        if len(self.filter_special_tokens) > 0:
            output["speakerA"] = self._remove_special_tokens(
                output["speakerA"], self.filter_special_tokens
            )
            output["speakerB"] = self._remove_special_tokens(
                output["speakerB"], self.filter_special_tokens
            )

        for split_utt in self._split_utterances([output]):
            turns_dict = []
            for turn in turnsA + turnsB:
                if pickle.dumps(turn) is None:
                    raise ValueError(f"None value for turn {turn}")

                turns_dict.append(turn.to_dict())

            split_utt["turns"] = turns_dict
            yield split_utt

    def _get_turns(
        self, dialogA: Dialog, dialogB: Dialog
    ) -> tuple[list[Turn], list[Turn]]:
        """
        Helper function to extract tokens from each speaker's dialog
        """

        turnsA = []
        turnsB = []

        curr_turnA = Turn.default("A", dialogA[0][0].conv_id)
        curr_turnB = Turn.default("B", dialogB[0][0].conv_id)
        assert curr_turnA.conv_id == curr_turnB.conv_id

        dialogs = list(zip_longest(dialogA + [None], dialogB + [None], fillvalue=None))

        idxA, idxB = 0, 0
        while idxA < len(dialogs) and idxB < len(dialogs):
            turnA, turnB = dialogs[idxA][0], dialogs[idxB][1]
            if turnA is None and turnB is not None:
                curr_turnB.update(
                    start=turnB[0].start,
                    end=turnB[-1].end,
                    word=curr_turnB.word
                    + " ".join([turn.word.lower() for turn in turnB]),
                    speaker="B",
                    turn_index=idxA + idxB,
                )
                turnsB.append(curr_turnB)
                idxB += 1

                curr_turnB = Turn.default("B", dialogB[0][0].conv_id)
                continue
            if turnB is None and turnA is not None:
                curr_turnA.update(
                    start=turnA[0].start,
                    end=turnA[-1].end,
                    word=curr_turnA.word
                    + " ".join([turn.word.lower() for turn in turnA]),
                    speaker="A",
                    turn_index=idxA + idxB,
                )
                turnsA.append(curr_turnA)
                idxA += 1

                curr_turnA = Turn.default("A", dialogA[0][0].conv_id)
                continue

            if turnA is None and turnB is None:
                idxA += 1
                idxB += 1
                continue

            if turnA[0].start < turnB[0].start:
                curr_turnA, curr_turnB = handle_channel(
                    turnA,
                    curr_turnA,
                    turnB,
                    curr_turnB,
                    yield_int_thresh=self.yield_int_thresh,
                )
                curr_turnA.update(turn_index=idxA + idxB)
                turnsA.append(curr_turnA)
                curr_turnA = Turn.default("A", dialogA[0][0].conv_id)
                idxA += 1
            else:
                curr_turnB, curr_turnA = handle_channel(
                    turnB,
                    curr_turnB,
                    turnA,
                    curr_turnA,
                    yield_int_thresh=self.yield_int_thresh,
                )
                curr_turnB.update(turn_index=idxA + idxB)
                turnsB.append(curr_turnB)
                curr_turnB = Turn.default("B", dialogB[0][0].conv_id)
                idxB += 1

        return turnsA, turnsB

    def _insert_bc(
        self,
        dialogA: Dialog,
        turnsA: list[Turn],
        bcA: Dialog,
        dialogB: Dialog,
        turnsB: list[Turn],
        bcB: Dialog,
    ):
        bc_token: int | None = self.tokens_dict["bc"] if self.include_bc_token else None
        dialogA, turnsA = insert_bc_channel(
            dialogA, turnsA, bcA, bc_token=bc_token, speaker="A"
        )
        dialogB, turnsB = insert_bc_channel(
            dialogB, turnsB, bcB, bc_token=bc_token, speaker="B"
        )
        return dialogA, turnsA, dialogB, turnsB

    def _insert_overlap(
        self,
        dialogA: Dialog,
        turnsA: list[Turn],
        overlapA: Dialog,
        dialogB: Dialog,
        turnsB: list[Turn],
        overlapB: Dialog,
    ):
        new_dialogA, turnsA = insert_overlap_channel(
            dialogA, turnsA, overlapA, speaker="A"
        )
        new_dialogB, turnsB = insert_overlap_channel(
            dialogB, turnsB, overlapB, speaker="B"
        )

        turnsA = update_turn_endings(
            turnsA, turnsB, yield_overlap_thresh=self.yield_overlap_thresh
        )
        turnsB = update_turn_endings(
            turnsB, turnsA, yield_overlap_thresh=self.yield_overlap_thresh
        )

        return new_dialogA, turnsA, new_dialogB, turnsB

    def _pseudo_insert_overlap(
        self,
        dialogA: Dialog,
        turnsA: list[Turn],
        overlapA: Dialog,
        dialogB: Dialog,
        turnsB: list[Turn],
        overlapB: Dialog,
    ):
        _, turnsA, _, turnsB = self._insert_overlap(
            dialogA, turnsA, overlapA, dialogB, turnsB, overlapB
        )

        if not self.include_overlaps:
            turnsA = [x for x in turnsA if x.turn_type != TurnType.OVERLAP]
            turnsB = [x for x in turnsB if x.turn_type != TurnType.OVERLAP]

        return turnsA, turnsB

    def _add_turn_ids(
        self, dialogA: Dialog, dialogB: Dialog, turnsA: list[Turn], turnsB: list[Turn]
    ) -> tuple[Dialog, Dialog, list[Turn], list[Turn]]:
        def _turn_channel_add_turn_ids(
            turnsA: list[Turn], turnsB: list[Turn], idxA: int, idxB: int
        ):
            if idxA > 0:
                turnsA[idxA].update(curr_prev_id=turnsA[idxA - 1].id)
            if idxA < len(turnsA) - 1:
                turnsA[idxA].update(curr_next_id=turnsA[idxA + 1].id)

            if turnsA[idxA].turn_type in [TurnType.BACKCHANNEL, TurnType.OVERLAP]:
                turnsA[idxA].update(other_prev_id=turnsB[idxB - 1].id)
                if turnsA[idxA]["start"] > turnsB[idxB]["start"]:
                    # A is overlapped by B entrirely; we already know that A.end < B.end
                    turnsA[idxA].update(
                        other_prev_id=turnsB[idxB].id,
                        other_next_id=turnsB[idxB].id,
                        overlapped_by=turnsB[idxB].id,
                    )
                    turnsB[idxB].overlaps.append(turnsA[idxA].id)

                else:
                    # A is before B@idxB ends but starts prior to the start of B@idxB
                    turnsA[idxA].update(other_next_id=turnsB[idxB].id)
                    if idxB > 0:
                        turnsA[idxA].update(other_prev_id=turnsB[idxB - 1].id)

                        if turnsB[idxB - 1].end > turnsA[idxA].start:
                            # A and B have partial overlap. A starts before B ends
                            turnsA[idxA].update(overlapped_by=turnsB[idxB - 1].id)
                            turnsB[idxB - 1].overlaps.append(turnsA[idxA].id)

            return turnsA, turnsB

        def _dialog_channel_add_turn_ids(dialog: Dialog, turns: list[Turn]):
            assert len(dialog) == len(turns), f"{len(dialog)} != {len(turns)}"
            for dialog_turn, turn in zip(dialog, turns):
                assert dialog_turn[0]["start"] == turn.start
                assert dialog_turn[-1]["end"] == turn.end

                for word in dialog_turn:
                    word["turn_id"] = turn.id

        idxA, idxB = 0, 0

        while idxA < len(turnsA) or idxB < len(turnsB):
            if idxA < len(turnsA) and idxB >= len(turnsB):
                turnsA, turnsB = _turn_channel_add_turn_ids(turnsA, turnsB, idxA, -1)
                idxA += 1
                continue
            if idxB < len(turnsB) and idxA >= len(turnsA):
                turnsB, turnsA = _turn_channel_add_turn_ids(turnsB, turnsA, idxB, -1)
                idxB += 1
                continue

            if turnsA[idxA].end < turnsB[idxB].end:
                turnsA, turnsB = _turn_channel_add_turn_ids(turnsA, turnsB, idxA, idxB)
                idxA += 1
            else:
                turnsB, turnsA = _turn_channel_add_turn_ids(turnsB, turnsA, idxB, idxA)
                idxB += 1

        _dialog_channel_add_turn_ids(dialogA, turnsA)
        _dialog_channel_add_turn_ids(dialogB, turnsB)

        return dialogA, dialogB, turnsA, turnsB

    """
    Initialize turns for each speaker and insert backchannels and overlaps if necessary
    """

    def _init_turns(self, dataset):
        speakerA = dataset["dialog"]["speakerA"]
        speakerB = dataset["dialog"]["speakerB"]
        dialogA, dialogB = self._get_tokens(speakerA, speakerB)

        speakerbcA = dataset["backchannel"]["speakerA"]
        speakerbcB = dataset["backchannel"]["speakerB"]
        bcA, bcB = self._get_tokens(speakerbcA, speakerbcB)

        speaker_overlapA = dataset["overlap"]["speakerA"]
        speaker_overlapB = dataset["overlap"]["speakerB"]
        overlapA, overlapB = self._get_tokens(speaker_overlapA, speaker_overlapB)

        turnsA, turnsB = self._get_turns(dialogA, dialogB)
        if len(turnsA) != len(dialogA):
            pass

        assert (
            len(turnsA) == len(dialogA)
        ), f"{len(turnsA)} != {len(dialogA)} {pprint.pformat(turnsA)} {pprint.pformat(dialogA)}"
        assert len(turnsB) == len(dialogB), f"{len(turnsB)} != {len(dialogB)}"

        if self.include_backchannels:
            dialogA, turnsA, dialogB, turnsB = self._insert_bc(
                dialogA, turnsA, bcA, dialogB, turnsB, bcB
            )

        if "4338" in turnsA[0].conv_id:
            pass
        if self.include_overlaps:
            dialogA, turnsA, dialogB, turnsB = self._insert_overlap(
                dialogA, turnsA, overlapA, dialogB, turnsB, overlapB
            )
        else:
            turnsA, turnsB = self._pseudo_insert_overlap(
                dialogA, turnsA, overlapA, dialogB, turnsB, overlapB
            )

        dialogA, dialogB, turnsA, turnsB = self._add_turn_ids(
            dialogA, dialogB, turnsA, turnsB
        )

        if isinstance(dialogA[0], list):
            dialogA = [word for sentence in dialogA for word in sentence]
            dialogB = [word for sentence in dialogB for word in sentence]

        return dialogA, dialogB, turnsA, turnsB

    def _align_channel(
        self, idxA, dialogA, idxB, dialogB, outputA, outputB, half_duration=0
    ):
        outputA["tokens"].append(dialogA[idxA]["tokens"])
        outputA["timings"].append([dialogA[idxA]["start"], dialogA[idxA]["end"]])

        overlap = dialogA[idxA]["end"] - dialogB[idxB]["start"]
        # B is closer to current A then the next A word
        if overlap > half_duration:
            outputB["tokens"].append(dialogB[idxB]["tokens"])
            outputB["timings"].append([dialogB[idxB]["start"], dialogB[idxB]["end"]])

            idxB += 1
        else:
            outputB["tokens"].append(self.tokens_dict["emp"])
            outputB["timings"].append([dialogA[idxA]["start"], dialogA[idxA]["end"]])

        idxA += 1

        return idxA, idxB, outputA, outputB

    def _align_tokens(self, dialogA, dialogB, output):
        idxA, idxB = 0, 0
        while idxA < len(dialogA) and idxB < len(dialogB):
            durationA = dialogA[idxA]["end"] - dialogA[idxA]["start"]
            durationB = dialogB[idxB]["end"] - dialogB[idxB]["start"]
            half_duration = durationA / 2
            if durationA > durationB:
                half_duration = durationB / 2

            # Add A dialog first
            if dialogA[idxA]["start"] <= dialogB[idxB]["start"]:
                idxA, idxB, output["speakerA"], output["speakerB"] = (
                    self._align_channel(
                        idxA,
                        dialogA,
                        idxB,
                        dialogB,
                        output["speakerA"],
                        output["speakerB"],
                        half_duration,
                    )
                )

            # Add B dialog first
            elif dialogB[idxB]["start"] < dialogA[idxA]["start"]:
                idxB, idxA, output["speakerB"], output["speakerA"] = (
                    self._align_channel(
                        idxB,
                        dialogB,
                        idxA,
                        dialogA,
                        output["speakerB"],
                        output["speakerA"],
                        half_duration,
                    )
                )

            else:
                assert False, "Should not reach here"

        # Add leftover tokens
        if idxA < len(dialogA):
            output["speakerA"]["tokens"].extend([x["tokens"] for x in dialogA[idxA:]])
            output["speakerA"]["timings"].extend(
                [[x["start"], x["end"]] for x in dialogA[idxA:]]
            )

            output["speakerB"]["tokens"].extend(
                [self.tokens_dict["emp"] for _ in dialogA[idxA:]]
            )
            output["speakerB"]["timings"].extend(
                [[x["start"], x["end"]] for x in dialogA[idxA:]]
            )

        if idxB < len(dialogB):
            output["speakerB"]["tokens"].extend([x["tokens"] for x in dialogB[idxB:]])
            output["speakerB"]["timings"].extend(
                [[x["start"], x["end"]] for x in dialogB[idxB:]]
            )

            output["speakerA"]["tokens"].extend(
                [self.tokens_dict["emp"] for _ in dialogB[idxB:]]
            )
            output["speakerA"]["timings"].extend(
                [[x["start"], x["end"]] for x in dialogB[idxB:]]
            )

        return output

    def _add_ts_token_dialog(self, dialogsA, dialogsB, turnsA, turnsB):
        """
        Add special token to indicate turn endings.
        Add automatically if the next token is an empty (<emp>) token.

        Otherwise, we insert a token which means that an empty token is added in the
        other speaker's utterance
        """
        turn_idxA = 0
        turn_idxB = 0
        idx = 0

        for dialog in zip(
            dialogsA["tokens"],
            dialogsA["timings"],
            dialogsB["tokens"],
            dialogsB["timings"],
        ):
            _, timingA, _, timingB = dialog
            if turn_idxA < len(turnsA) and timingA[1] == turnsA[turn_idxA]["end"]:
                eot_token_id = self.tokens_dict["eot"]
                if turnsA[turn_idxA]["turn_end_type"] == TurnEndType.YIELD:
                    eot_token_id = self.tokens_dict["yield"]
                elif turnsA[turn_idxA]["turn_type"] == TurnType.OVERLAP:
                    eot_token_id = self.tokens_dict["eint"]
                elif turnsA[turn_idxA]["turn_type"] == TurnType.BACKCHANNEL:
                    eot_token_id = self.tokens_dict["ebc"]

                if (
                    idx < len(dialogsA["tokens"]) - 1
                    and dialogsA["tokens"][idx + 1] == self.tokens_dict["emp"]
                ):
                    dialogsA["tokens"][idx + 1] = eot_token_id
                    dialogsA["timings"][idx + 1] = [-1, -1]
                else:
                    dialogsA["tokens"].insert(idx + 1, eot_token_id)
                    dialogsA["timings"].insert(idx + 1, [-1, -1])

                    dialogsB["tokens"].insert(idx + 1, self.tokens_dict["emp"])
                    dialogsB["timings"].insert(idx + 1, [-1, -1])
                turn_idxA += 1

            if turn_idxB < len(turnsB) and timingB[1] == turnsB[turn_idxB]["end"]:
                eot_token_id = self.tokens_dict["eot"]
                if turnsB[turn_idxB]["turn_end_type"] == TurnEndType.YIELD:
                    eot_token_id = self.tokens_dict["yield"]
                elif turnsB[turn_idxB]["turn_type"] == TurnType.OVERLAP:
                    eot_token_id = self.tokens_dict["eint"]
                elif turnsB[turn_idxB]["turn_type"] == TurnType.BACKCHANNEL:
                    eot_token_id = self.tokens_dict["ebc"]

                if (
                    idx < len(dialogsB["tokens"]) - 1
                    and dialogsB["tokens"][idx + 1] == self.tokens_dict["emp"]
                ):
                    dialogsB["tokens"][idx + 1] = eot_token_id
                    dialogsB["timings"][idx + 1] = [-1, -1]
                else:
                    dialogsB["tokens"].insert(idx + 1, eot_token_id)
                    dialogsB["timings"].insert(idx + 1, [-1, -1])

                    dialogsA["tokens"].insert(idx + 1, self.tokens_dict["emp"])
                    dialogsA["timings"].insert(idx + 1, [-1, -1])
                turn_idxB += 1

            idx += 1

        return dialogsA, dialogsB

    """
    Build the turn state for each token in the dialog.
    For this we have two states:
        1. TurnType: None, Normal, Backchannel, Overlap
        2. TurnEndType: None, Normal, Yield

    We include these states for all tokens in each turn including the end-of-turn token
    """

    def _add_ts_token_state(self, dialogs, turns):
        turn_idx = 0
        for dialog in zip(dialogs["tokens"], dialogs["timings"]):
            token, timing = dialog
            in_turn = (
                turn_idx < len(turns)
                and timing[0] >= turns[turn_idx]["start"]
                and timing[1] <= turns[turn_idx]["end"]
            )
            is_special = token in [
                self.tokens_dict["eot"],
                self.tokens_dict["yield"],
                self.tokens_dict["eint"],
                self.tokens_dict["ebc"],
            ]
            if in_turn or is_special:
                if turn_idx >= len(turns):
                    dialogs["turn_ids"].append(0)
                    dialogs["turn_type_ids"].append(TurnType.NONE)
                    dialogs["turn_end_type_ids"].append(TurnEndType.NONE)
                    continue

                dialogs["turn_ids"].append(turns[turn_idx].id.value)
                dialogs["turn_type_ids"].append(turns[turn_idx]["turn_type"])
                dialogs["turn_end_type_ids"].append(turns[turn_idx]["turn_end_type"])

                if is_special:
                    turn_idx += 1

            elif turn_idx < len(turns) and timing[1] > turns[turn_idx]["end"]:
                # when updating the turn we should whether the next token is occuring directly after the special token
                # TODO Fixup here where `in_turn` is set after turn-ending if the other speaker's tokens end prior to the current speaker's turn ending
                turn_idx += 1
                if (
                    turn_idx < len(turns)
                    and timing[0] >= turns[turn_idx]["start"]
                    and timing[1] <= turns[turn_idx]["end"]
                ):
                    dialogs["turn_ids"].append(turns[turn_idx].id.value)
                    dialogs["turn_type_ids"].append(turns[turn_idx]["turn_type"])
                    dialogs["turn_end_type_ids"].append(
                        turns[turn_idx]["turn_end_type"]
                    )
                else:
                    dialogs["turn_ids"].append(0)
                    dialogs["turn_type_ids"].append(TurnType.NONE)
                    dialogs["turn_end_type_ids"].append(TurnEndType.NONE)
            else:
                dialogs["turn_ids"].append(0)
                dialogs["turn_type_ids"].append(TurnType.NONE)
                dialogs["turn_end_type_ids"].append(TurnEndType.NONE)

        return dialogs

    def _remove_special_tokens(self, data, special_tokens=[]):
        special_tokens = torch.tensor(special_tokens)
        index = torch.logical_not(torch.isin(data["input_ids"], special_tokens))

        for key in data.keys():
            if isinstance(data[key], torch.Tensor):
                data[key] = data[key][index]
            elif isinstance(data[key], list):
                data[key] = [data[key][i] for i in range(len(data[key])) if index[i]]

        return data

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
                    temp[speaker] = {}
                    for key, val in value.items():
                        if isinstance(val, torch.Tensor):
                            temp[speaker][key] = val[start_idx:end_idx].clone()
                        elif isinstance(val, list):
                            temp[speaker][key] = val[start_idx:end_idx]
                        else:
                            temp[speaker][key] = val

                result.append(temp)

                if end_idx == len(conv["speakerA"]["input_ids"]):
                    break

                start_idx = end_idx - self.overlap_length
                end_idx = min(
                    len(conv["speakerA"]["input_ids"]), self.max_length + start_idx
                )

        return result
