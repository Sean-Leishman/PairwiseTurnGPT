import re
import logging
import os

from data.utils import read_txt, remove_multiple_whitespace, get_logger
from difflib import SequenceMatcher
from enum import IntEnum

logger = get_logger(__name__, level=logging.DEBUG)

BACKCHANNELS = [
    "yeah",
    "umhum",
    "uhhuh",
    "right",
    "oh",
    "oh yeah",
    "yeah yeah",
    "right right",
    "oh really",
    "umhum umhum",
    "uhhuh uhhuh",
    "oh uhhuh" "uh" "uhhuh uhhuh",
]


class AssignType(IntEnum):
    UNASSIGNED = -1
    NORMAL = (0,)
    FROM_PREVIOUS = (1,)
    FROM_NEXT = (2,)
    FROM_PREVIOUS_AND_NEXT = 3


def save_words_in_file(filename, save_dir="words_split"):
    with open(filename, "r") as f:
        lines = f.readlines()

    while len(lines) > 0:
        conv_id, *_ = lines[0].split(" ")

        save_filename = os.path.join(save_dir, f"{conv_id}.txt")
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        with open(save_filename, "w") as f:
            idx = 0
            while len(lines) > 0 and idx < len(lines):
                curr_conv_id, *_ = lines[0].split(" ")
                if curr_conv_id == conv_id:
                    f.write(lines.pop(0))
                else:
                    idx += 1


def approx_equals(str1, str2):
    """
    Check if two strings are approximately equal.
    Where one string is "EDACC-CXX" and the other is "EDACC-CXX_P00"
    """
    if str1 == str2:
        return True

    if str1.startswith(str2):
        return True

    if str2.startswith(str1):
        return True

    return False


def sub_regex(s):
    """
    Switchboard annotation specific regexp.

    See:
        - `datasets_turntaking/features/dataset/switchboard.md`
        - https://www.isip.piconepress.com/projects/switchboard/doc/transcription_guidelines/transcription_guidelines.pdf

    """
    # Noise
    s = re.sub(r"\[noise\]", "", s)
    s = re.sub(r"\[vocalized-noise\]", "", s)

    s = re.sub(r"\[silence\]", "", s)

    # <b_aside>, <e_aside>
    s = re.sub(r"<b_aside>", "", s)
    s = re.sub(r"<e_aside>", "", s)

    # laughter
    s = re.sub(r"\[laughter\]", "", s)
    # laughing and speech e.g. [laughter-yeah] -> yeah
    s = re.sub(r"\[laughter-(\w*)\]", r"\1", s)
    s = re.sub(r"\[laughter-(\w*\'*\w*)\]", r"\1", s)

    # [laughter-don'[t]]
    s = re.sub(r"\[laughter-(\w*)\'*\[(\w*)\]\]", r"\1'\2", s)

    # [laughter-don'[t]-]
    s = re.sub(r"\[laughter-(\w*)\'*\[(\w*)\]*.?\]", r"\1'\2", s)

    # Partial words: w[ent] -> went
    s = re.sub(r"(\w+)\[(\w*\'*\w*)\]", r"\1\2", s)
    # Partial words: -[th]at -> that
    s = re.sub(r"-\[(\w*\'*\w*)\](\w+)", r"\1\2", s)

    s = re.sub(r"\[(\w+)]'s", r"\1's", s)
    s = re.sub(r"(.*?)\[(.*?)\]", r"\1\2", s)

    # restarts
    s = re.sub(r"(\w+)-\s", r"\1 ", s)
    s = re.sub(r"(\w+)-$", r"\1", s)

    # Pronounciation variants
    s = re.sub(r"(\w+)\_\d", r"\1", s)

    # Mispronounciation [splace/space] -> space
    s = re.sub(r"\[\w+\/(\w+)\]", r"\1", s)

    # Coinage. remove curly brackets... keep word
    s = re.sub(r"\{(\w*)\}", r"\1", s)

    # remove double spacing on last
    s = re.sub(r"\s\s+", " ", s)

    s = s.replace("-", "")
    return s.strip()  # remove whitespace start/end


def preproc(word):
    word = remove_multiple_whitespace(word)
    word = word.lower()
    word = sub_regex(word)
    return word


def _fill_null(wfeats, start, end):
    def flush_window():
        # Split time evenly between all words in the Windows
        start = wfeats[curr_wfeat_window[0]]["start"]
        end = wfeats[curr_wfeat_window[-1]]["end"]
        for word_idx in curr_wfeat_window:
            wfeats[word_idx]["start"] = start + word_idx * (end - start) / len(
                curr_wfeat_window
            )
            wfeats[word_idx]["end"] = start + (word_idx + 1) * (end - start) / len(
                curr_wfeat_window
            )

    if len(wfeats) == 0:
        print("No words found in turn")
        return wfeats

    if wfeats[0]["start"] is None:
        wfeats[0]["start"] = start
    if wfeats[-1]["end"] is None:
        wfeats[-1]["end"] = end

    for idx, wfeat in enumerate(wfeats):
        if wfeat["start"] is None:
            if idx > 0 and wfeats[idx - 1]["end"] is not None:
                wfeats[idx]["start"] = wfeats[idx - 1]["end"]
                wfeats[idx]["assign_type"] = AssignType.FROM_PREVIOUS
        if wfeat["end"] is None:
            if idx < len(wfeats) - 1 and wfeats[idx + 1]["start"] is not None:
                wfeats[idx]["end"] = wfeats[idx + 1]["start"]
                wfeats[idx]["assign_type"] = AssignType.FROM_NEXT

                if wfeats[idx]["assign_type"] == AssignType.FROM_PREVIOUS:
                    wfeats[idx]["assign_type"] = AssignType.FROM_PREVIOUS_AND_NEXT

    # Find all NONE assignments in a row and split time evenly between them
    curr_wfeat_window = []
    for idx, wfeat in enumerate(wfeats):
        if wfeat["start"] is not None and wfeat["end"] is not None:
            if len(curr_wfeat_window) > 0:
                flush_window()

            curr_wfeat_window = []
            continue

        if len(curr_wfeat_window) == 0:
            if wfeat["start"] is None:
                logger.warning(
                    f"Start should always be set as first word in window {wfeat}"
                )

            curr_wfeat_window.append(idx)
            continue

        if wfeat["end"] is not None:
            # last word in sequence
            curr_wfeat_window.append(idx)
            flush_window()
            continue

        curr_wfeat_window.append(idx)

    return wfeats


def _match_wfeats_target(wfeats, wfeats_target):
    assert len(wfeats) == len(
        wfeats_target
    ), f"len(wfeats) = {len(wfeats)} != len(wfeats_target) = {len(wfeats_target)}"
    for wfeat, target_word in zip(wfeats, wfeats_target):
        if wfeat["word"] != target_word:
            wfeat["word"] = target_word

    return wfeats


def _limit_words(words, start, end, buffer=1):
    return [
        word for word in words if word[0] >= start - buffer and word[1] <= end + buffer
    ]


def match_ratio(s1, s2):
    return SequenceMatcher(None, s1, s2).ratio()


def _find_word_in_turn(
    words, target_start, target_end, target_word, last_timing=None, words_since_last=0
):
    target_word = preproc(target_word)
    best_match = None
    best_match_ratio = 0.0
    for idx, word in enumerate(words):
        start, end, word = word
        if start < target_start - 1:
            continue
        if end > target_end + 1:
            break

        if last_timing is not None:
            if last_timing > start:
                # Ensure current word is after the last word
                continue
            if start - last_timing > (words_since_last + 1):
                # Ensure current word is within 1 second of the last word
                continue
        elif start - target_start > 0.75 and words_since_last == 0:
            # Where last_timing is not set, so it is the first word in the turn
            # Ensure the word is within 0.75 seconds of the target start
            # `start` can be set to be
            continue

        word_match = match_ratio(preproc(word), target_word)
        if word_match > 0.8 and word_match > best_match_ratio:
            best_match_ratio = word_match
            best_match = (start, end, word)
            words.pop(idx)

    return best_match


def _extract_word_features(words, start=0.0, end=1.0, target_words=[]):
    if len(target_words) == 0 or len(words) == 0:
        return []

    wfeats = []
    last_word_timing = None
    words_since_last = 0
    for target_word in target_words:
        word_in_turn = _find_word_in_turn(
            words,
            start,
            end,
            target_word,
            last_timing=last_word_timing,
            words_since_last=words_since_last,
        )
        if word_in_turn is None:
            # require first word in turn to be set
            if len(wfeats) == 0:
                wfeats.append(
                    {
                        "start": start,
                        "end": None,
                        "word": target_word,
                        "assign_type": AssignType.UNASSIGNED,
                    }
                )
                last_word_timing = start
                words_since_last += 1
                continue

            wfeats.append(
                {
                    "start": None,
                    "end": None,
                    "word": target_word,
                    "assign_type": AssignType.UNASSIGNED,
                }
            )
            words_since_last += 1
            continue

        start_word, end_word, word = word_in_turn
        last_word_timing = end_word
        words_since_last = 0
        word = preproc(word)

        wfeats.append(
            {
                "start": start_word,
                "end": end_word,
                "word": word,
                "assign_type": AssignType.NORMAL,
            }
        )

    return _match_wfeats_target(_fill_null(wfeats, start, end), target_words)


def extract_dialog(conv_id, words_filename="", turns_filename=""):
    if not os.path.exists(words_filename) and not os.path.exists(turns_filename):
        logger.warning(f"Files {words_filename} and {turns_filename} not found")
        return []

    words = read_txt(words_filename)
    turns = read_txt(turns_filename)

    word_dict = {}
    for word in words:
        curr_conv_id, _, start, duration, word, _ = word.split(" ")
        if not approx_equals(curr_conv_id, conv_id):
            continue

        start = float(start)
        end = start + float(duration)
        if curr_conv_id not in word_dict:
            word_dict[conv_id] = [(start, end, word)]
        else:
            word_dict[conv_id].append((start, end, word))

    if len(word_dict) == 0:
        raise ValueError(f"Conv_id {conv_id} not found in words with format")

    utterances = []
    for turn in turns:
        curr_conv_id, _, speaker_id, start, end, _, *turn_words = turn.split(" ")
        start, end = float(start), float(end)

        if not approx_equals(curr_conv_id, conv_id):
            continue

        words = word_dict.get(curr_conv_id, None)
        if words is None:
            raise ValueError(f"Conv_id {curr_conv_id} not found in words with format")

        turn_words = [preproc(word) for word in turn_words]
        wfeats = _extract_word_features(
            words, start=start, end=end, target_words=turn_words
        )

        curr_utterance = {
            "text": " ".join(turn_words),
            "wfeats": wfeats,
            "start": wfeats[0]["start"],
            "end": wfeats[-1]["end"],
            "conv_id": curr_conv_id,
            "speaker": "A" if "A" == speaker_id[-1] else "B",
        }
        utterances.append(curr_utterance)
    return utterances


def combine_dialogue_without_timings(dialogue, separated_by=0.5):
    dialogue.sort(key=lambda key: (key["start"], -key["end"]))
    return join_utterance_separated_by(dialogue, separated_by=separated_by)


def join_utterance_separated_by(dialogs, separated_by=0.5):
    drefined = []

    lasts = [None for _ in range(2)]
    dic = {"A": 0, "B": 1}
    for _, curr in enumerate(dialogs):
        # If current text is entriely contained within the last utterance
        last_current = lasts[dic[curr["speaker"]]]
        if last_current is None:
            lasts[dic[curr["speaker"]]] = curr
            continue

        # Join utterances from current speaker < separated_by
        if (
            last_current is not None
            and curr["start"] - last_current["end"] < separated_by
        ):
            last_current["text"] += f" {curr['text']}"
            last_current["end"] = curr["end"]
            last_current["wfeats"].extend(curr["wfeats"])

        else:
            drefined.append(last_current)
            lasts[dic[curr["speaker"]]] = curr

    drefined.append(lasts[0])
    drefined.append(lasts[1])

    drefined.sort(key=lambda x: (x["start"], -x["end"]))
    return drefined


def remove_backchannels(dialogs, pre_silence=1, post_silence=1, bc_duration=1):
    new_dialog, _ = pairwise_remove_backchannels(
        dialogs, pre_silence, post_silence, bc_duration
    )
    return new_dialog


def pairwise_remove_backchannels(dialogs, pre_silence=1, post_silence=1, bc_duration=1):
    dialogsA = [x for x in dialogs if x["speaker"] == "A"]
    dialogsB = [x for x in dialogs if x["speaker"] == "B"]

    assert (
        len(dialogsA) + len(dialogsB) == len(dialogs)
    ), f"dialogs not separated by speaker: {len(dialogsA)} + {len(dialogsB)} != {len(dialogs)}"

    def remove_bc_from_channel(dialogs, end_of_utterance_time=0):
        last_end = 0
        new_dialog = []
        new_bc = []
        for idx, dialog in enumerate(dialogs):
            bc_in = dialog["text"] in BACKCHANNELS

            # pre silence is 1s, post silence is 1s and utterance length is less than 1
            duration = dialog["end"] - dialog["start"]
            pre_sil = dialog["start"] - last_end

            last_end = dialog["end"]

            post_sil = end_of_utterance_time - dialog["end"]
            if idx != len(dialogs) - 1:
                post_sil = dialogs[idx + 1]["start"] - dialog["end"]

            if (
                bc_in
                and duration <= bc_duration
                and pre_sil >= pre_silence
                and post_sil >= post_silence
            ):
                new_bc.append(dialog)
                continue

            new_dialog.append(dialog)
        return new_dialog, new_bc

    end_of_utterance_time = max(dialogsA[-1]["end"], dialogsB[-1]["end"])
    new_dialogsA, new_bcA = remove_bc_from_channel(dialogsA, end_of_utterance_time)
    new_dialogsB, new_bcB = remove_bc_from_channel(dialogsB, end_of_utterance_time)

    new_dialogs = new_dialogsA + new_dialogsB
    new_bc = new_bcA + new_bcB

    new_dialogs.sort(key=lambda key: (key["start"], -key["end"]))
    new_bc.sort(key=lambda key: (key["start"], -key["end"]))

    return new_dialogs, new_bc


def remove_overlaps(dialogs):
    drefined = [dialogs[0]]
    overlaps = []
    for idx, curr in enumerate(dialogs[1:]):
        # If current text is entriely contained within the last utterance
        if drefined[-1]["start"] <= curr["start"] <= drefined[-1]["end"]:
            if drefined[-1]["start"] <= curr["end"] <= drefined[-1]["end"]:
                overlaps.append(curr)
                continue

        drefined.append(curr)
    return drefined, overlaps


def insert_overlapped_bc(dialogs, backchannels):
    """
    Reinserts backchannels into the dialog if it was removed but the
    utterance is completely contained within own speaker's utterance.
    Occurs as the backchannel performs no overlap checking
    """

    def insert_bc_into_channel(dialogs, backchannels):
        new_bc = []

        for bc in backchannels:
            new_bc.append(bc)
            for dialog in dialogs:
                if dialog["start"] <= bc["start"] <= dialog["end"]:
                    new_bc.pop(-1)
                    for idx, wfeat in enumerate(dialog["wfeats"][1:], start=1):
                        prev_wfeat = dialog["wfeats"][idx - 1]
                        if prev_wfeat["end"] <= bc["start"] <= wfeat["start"]:
                            for bc_wfeat in bc["wfeats"][::-1]:
                                dialog["wfeats"].insert(idx, bc_wfeat)
                            break
                    dialog["text"] = " ".join([x["word"] for x in dialog["wfeats"]])
                    break
                if dialog["start"] > bc["start"]:
                    break

        return dialogs, new_bc

    old_length_dialogs = len(dialogs)
    old_length_bc = len(backchannels)

    dialogsA = [x for x in dialogs if x["speaker"] == "A"]
    dialogsB = [x for x in dialogs if x["speaker"] == "B"]

    backchannelsA = [x for x in backchannels if x["speaker"] == "A"]
    backchannelsB = [x for x in backchannels if x["speaker"] == "B"]

    new_dialogsA, new_bcA = insert_bc_into_channel(dialogsA, backchannelsA)
    new_dialogsB, new_bcB = insert_bc_into_channel(dialogsB, backchannelsB)

    new_dialogs = new_dialogsA + new_dialogsB
    new_dialogs.sort(key=lambda key: (key["start"], -key["end"]))

    new_bc = new_bcA + new_bcB
    new_bc.sort(key=lambda key: (key["start"], -key["end"]))

    return new_dialogs, new_bc


def combine_consecutive_trps(dialogs, bc=[], overlap=[]):
    temp_dialogs = [x | {"dialog_type": "dialog"} for x in dialogs]
    temp_bc = [x | {"dialog_type": "bc"} for x in bc]
    temp_overlaps = [x | {"dialog_type": "overlap"} for x in overlap]
    temp_dialogs = temp_dialogs + temp_bc + temp_overlaps
    temp_dialogs.sort(key=lambda key: (key["start"], -key["end"]))

    combined_dialogs = [temp_dialogs[0]]
    for idx in range(1, len(temp_dialogs)):
        if (
            combined_dialogs[-1]["speaker"] == temp_dialogs[idx]["speaker"]
            and temp_dialogs[idx]["dialog_type"] == "dialog"
            and combined_dialogs[-1]["dialog_type"] == "dialog"
        ):
            combined_dialogs[-1]["text"] += f" {temp_dialogs[idx]['text']}"
            combined_dialogs[-1]["end"] = temp_dialogs[idx]["end"]
            combined_dialogs[-1]["wfeats"].extend(temp_dialogs[idx]["wfeats"])
            combined_dialogs[-1]["dialog_type"] = "dialog"
        else:
            combined_dialogs.append(temp_dialogs[idx])
    return combined_dialogs


def separate_by_speaker(dialog, conv_id=0):
    speakerA = []
    speakerB = []

    for idx, utterance in enumerate(dialog):
        dialog[idx]["conv_id"] = conv_id
        if utterance["speaker"] == "A":
            speakerA.append(utterance)

        elif utterance["speaker"] == "B":
            speakerB.append(utterance)
        else:
            raise Exception(f"No label for: {utterance['speaker']}")

    return {"speakerA": speakerA, "speakerB": speakerB}
