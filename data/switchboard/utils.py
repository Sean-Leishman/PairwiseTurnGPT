import re

from data.utils import remove_multiple_whitespace, read_txt

OmitText = [
    "[silence]",
    "[noise]",
    "[vocalized-noise]",
]

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


def _clean_dialogs():
    pass


def _read_transcript_line(line):
    sepLine = line.split(" ")

    text = " ".join(sepLine[3:]).strip()
    start = float(sepLine[1])
    end = float(sepLine[2])

    return text, start, end


def _return_overlap(textA, textB, startA, startB, endA, endB):
    if startA > startB and endA < endB:
        return textA, "A"
    elif startB > startA and endB < endA:
        return textB, "B"

    return None, None


def _check_overlap_silence(past_line, next_line, thresh=1):
    past_text, past_start, past_end = _read_transcript_line(past_line)
    next_text, next_start, next_end = _read_transcript_line(next_line)

    if past_text != "[silence]" or next_text != "[silence]":
        return False

    if past_end - past_start < 1:
        return False

    if next_end - next_start < 1:
        return False

    return True


# Preprocessing handled by TurnGPT (https://github.com/ErikEkstedt/datasets_turntaking/blob/main/datasets_turntaking/dataset/switchboard/utils.py)


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


def extract_speaker_timings(transcript, min_word_diff=0.05):
    out = [[], []]
    for speaker in [0, 1]:
        for utterance in transcript[speaker]:
            start, end = utterance["wfeats"][0]["start"], utterance["wfeats"][0]["end"]

            for word in utterance["wfeats"][1:]:
                if word["start"] - end < min_word_diff:
                    end = word["end"]
                else:
                    out[speaker].append((start, end))
                    start = word["start"]
                    end = word["end"]

            out[speaker].append((start, end))
    return out


def print_transcript_timing(dialog, timings):
    dialog = dialog[0]
    timing = timings[0]

    for idx in range(len(dialog)):
        print(dialog[idx])
        print(timing[idx])


def extract_dialog(filenames):
    trans_filenameA, words_filenameA, trans_filenameB, words_filenameB = filenames

    utterancesA = _extract_utterance_word_feats(
        trans_filenameA, words_filenameA, speaker="A"
    )
    utterancesB = _extract_utterance_word_feats(
        trans_filenameB, words_filenameB, speaker="B"
    )

    return [utterancesA, utterancesB]


def _extract_word_features(filename, speaker):
    words = read_txt(filename)

    word_feats = {}
    for word_row in words:
        word_row = remove_multiple_whitespace(word_row).strip()

        key, start, end, word = word_row.split(" ")

        # Apply regex?
        word = sub_regex(word)

        # Check if word should be omitted
        if not (word in OmitText or word == ""):
            if key in word_feats:
                word_feats[key].append(
                    {
                        "word": word,
                        "start": float(start),
                        "end": float(end),
                    }
                )
            else:
                word_feats[key] = [
                    {
                        "word": word,
                        "start": float(start),
                        "end": float(end),
                    }
                ]
    return word_feats


def _extract_utterance_word_feats(trans_filename, words_filename, speaker):
    word_feats = _extract_word_features(words_filename, speaker)

    transcript = read_txt(trans_filename)

    utterances = []
    for row in transcript:
        key, start, end, *words = row.split(" ")

        if not (words[0] in OmitText and len(words) == 1):
            word_feat = word_feats.get(key, None)

            if word_feat is None:
                continue

            for x in word_feat:
                if isinstance(x, list):
                    pass

            words = " ".join(words)

            # Apply regex?
            words = sub_regex(words)

            utterances.append(
                {
                    "text": words,
                    "wfeats": word_feat,
                    "start": word_feat[0]["start"],
                    "end": word_feat[-1]["end"],
                    "speaker": speaker,
                }
            )
    return utterances


def remove_words_from_dialog(dialog):
    new_dialog = [[], []]
    for speaker in [0, 1]:
        for utterance in dialog[speaker]:
            new_dialog[speaker].append(
                {
                    "text": utterance["text"],
                    "start": utterance["start"],
                    "end": utterance["end"],
                }
            )

    return new_dialog


"""
Only combines based on turns identified within the structure of the conversation and so based
on the start of an utterance without consideration of the word level
"""


def combine_dialogue_without_timings(dialogue, separated_by=0.5):
    combined = dialogue[0]
    combined.extend(dialogue[1])
    combined.sort(key=lambda key: (key["start"], -key["end"]))

    combined = join_utterance_separated_by(combined, separated_by=separated_by)
    return combined


def _pp_dialogue(dialogue):
    out = ""
    start = 0
    curr_speaker = None

    for idx in range(len(dialogue)):
        if curr_speaker is None or curr_speaker != dialogue[idx]["speaker"]:
            curr_speaker = dialogue[idx]["speaker"]
            out += f": {start} - {dialogue[idx-1]['end']}"
            start = dialogue[idx]["start"]
            out += f"\n{curr_speaker}"
        out += f" {dialogue[idx]['text']}"

    print(out)


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


def pairwise_remove_overlaps(dialogs, speakers=2):
    drefined = []
    overlaps = []

    lasts = [None for _ in range(speakers)]
    dic = {"A": 0, "B": 1}
    for idx, curr in enumerate(dialogs):
        # If current text is entriely contained within the last utterance
        last = lasts[not dic[curr["speaker"]]]
        last_current = lasts[dic[curr["speaker"]]]

        if last is not None and last["start"] <= curr["start"] <= last["end"]:
            if last["start"] <= curr["end"] <= last["end"]:
                overlaps.append(curr)
                continue

        lasts[dic[curr["speaker"]]] = curr
        drefined.append(curr)
    return drefined, overlaps


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
    new_dialogsA, new_bcA = remove_bc_from_channel(
        dialogsA, end_of_utterance_time)
    new_dialogsB, new_bcB = remove_bc_from_channel(
        dialogsB, end_of_utterance_time)

    new_dialogs = new_dialogsA + new_dialogsB
    new_bc = new_bcA + new_bcB

    new_dialogs.sort(key=lambda key: (key["start"], -key["end"]))
    new_bc.sort(key=lambda key: (key["start"], -key["end"]))

    return new_dialogs, new_bc


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
                    dialog["text"] = " ".join(
                        [x["word"] for x in dialog["wfeats"]])
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


"""
Actually just need to convert into format for parent dataset.
Where in __getitem__(idx) idx refers to the conversation and returns all turns within
a conversation
So this function just needs to return the turn list for a conversation
"""


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


def extract_word_features(dialog):
    utterancesA = []
    utterancesB = []

    for d in dialog[0]:
        utterancesA.extend(d["wfeats"])
    for d in dialog[1]:
        utterancesB.extend(d["wfeats"])

    return [utterancesA, utterancesB]


def pairwise_extract_dialog(filenames):
    trans_filenameA, words_filenameA, trans_filenameB, words_filenameB = filenames

    utterancesA = _pairwise_extract_utterance_word_feats(
        trans_filenameA, words_filenameA, speaker="A"
    )
    utterancesB = _pairwise_extract_utterance_word_feats(
        trans_filenameB, words_filenameB, speaker="B"
    )

    return [utterancesA, utterancesB]


def _pairwise_extract_utterance_word_feats(trans_file, word_file, speaker):
    # Only require parsing of the word features??
    # Due to pairwise nature of data allowing overlapping dialogs

    words = read_txt(word_file)

    utterance = []
    for word_row in words:
        word_row = remove_multiple_whitespace(word_row.strip())
        key, start, end, word = word_row.split(" ")

        word = sub_regex(word)
        if not (word in OmitText or len(word) == 0):
            utterance.append(
                {
                    "word": word,
                    "start": float(start),
                    "end": float(end),
                    "speaker": speaker,
                    "conv_id": key,
                }
            )

    return utterance


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
