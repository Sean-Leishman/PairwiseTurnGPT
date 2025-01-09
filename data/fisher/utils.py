import os
import re
from typing import List

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
    "oh uhhuh"
    "uh"
    "uhhuh uhhuh",
]


def regexp(s, remove_restarts=False):
    """
    See information about annotations at:
    * https://catalog.ldc.upenn.edu/docs/LDC2004T19/fe_03_readme.txt
    Regexp
    ------
    * Special annotations:  ["[laughter]", "[noise]", "[lipsmack]", "[sigh]"]
    * double paranthesis "((...))" was not heard by the annotator
      and can be empty but if not empty the annotator made their
      best attempt of transcribing what was said.
    * What's "[mn]" ? (can't find source?)
        * Inaudible
        * seems to be backchannel or laughter
        * oh/uh-huh/mhm/hehe
    * Names/accronyms (can't find source?)
        * t._v. = TV
        * m._t._v. = MTV
    """

    # Noise
    s = re.sub(r"\[noise\]", "", s)
    # laughter
    s = re.sub(r"\[laughter\]", "", s)
    s = re.sub(r"\[laugh\]", "", s)
    # lipsmack
    s = re.sub(r"\[lipsmack\]", "", s)
    # sigh
    s = re.sub(r"\[sigh\]", "", s)
    # [mn] inaubible?
    s = re.sub(r"\[mn\]", "", s)

    s = re.sub(r"\[cough\]", "", s)
    s = re.sub(r"\[breath\]", "", s)
    s = re.sub(r"\[pause\]", "", s)
    s = re.sub(r"\[sneeze\]", "", s)

    # clean restarts
    # if remove_restarts=False "h-" -> "h"
    # if remove_restarts=True  "h-" -> ""
    if remove_restarts:
        s = re.sub(r"(\w+)-\s", " ", s)
        s = re.sub(r"(\w+)-$", r"", s)
    else:
        s = re.sub(r"(\w+)-\s", r"\1 ", s)
        s = re.sub(r"(\w+)-$", r"\1", s)

    # doubble paranthesis (DP) with included words
    # sometimes there is DP inside another DP
    s = re.sub(r"\(\(((.*?)+)\)\)", r"\1", s)
    s = re.sub(r"\(\(((.*?)+)\)\)", r"\1", s)

    s = re.sub(r"\[\[((.*?)+)\]\]", r"\1", s)

    # empty doubble paranthesis
    s = re.sub(r"\(\(\s*\)\)", "", s)

    # Names/accronyms
    s = re.sub(r"\.\_", "", s)

    # remove punctuation
    # (not included in annotations but artifacts from above)
    s = re.sub(r"\.", "", s)
    s = re.sub(r"[,-?.]", "", s)

    # remove double spacing on last
    s = re.sub(r"\s\s+", " ", s)
    return s.strip()  # remove whitespace start/end


def read_txt(path: str):
    with open(path) as f:
        lines = f.readlines()
    return lines


def _extract_word_features(path):
    word_feats = {}
    if not os.path.isfile(path):
        print(f"path: {path} not found")
        return word_feats

    with open(path) as f:
        line = f.readline()
        while line:
            speaker, turn, start, end, word = line.split(" ")
            key = f"{speaker}{turn}"

            word = regexp(word)
            if word == "":
                line = f.readline()
                continue

            if key not in word_feats:
                word_feats[key] = []

            word_feats[key].append(
                {
                    "word": word,
                    "start": float(start),
                    "end": float(end),
                    "speaker": speaker
                }
            )
            line = f.readline()

    return word_feats


"""
TODO
Do the word feature within a utility function and create transcript files that contain the utterance ID so each line has a separate IDs and we assign words
these IDs inside of their file. In the same format as switchboard. As right now we have to run this code for 10 minutes everytime we would like to 
recrate the dataset which will not do 
"""


def extract_dialog(paths: List[str], with_word_feats=True):
    anno = [[], []]

    path, word_path = paths

    key = path.split("/")[-1][:-4]
    # returns map with words features as value and word as key
    word_feats = _extract_word_features(word_path)
    for key in word_feats.keys():
        channel = 0 if key[0] == 'A' else 1

        anno[channel].append({
            "start": word_feats[key][0]["start"],
            "wfeats": word_feats[key],
            "end": word_feats[key][-1]["end"],
            "text": " ".join([x["word"] for x in word_feats[key]]),
            "speaker": key[0],
        })

    return anno


def read_indexes():
    indexes = {}
    with open("/home/seanleishman/BespokeTart/data/fisher/fisher/data/word_timings/indexes.txt") as f:
        line = f.readline()
        while line:
            key, value = line.split(" ")
            indexes[key] = int(value)

            line = f.readline()
    return indexes


def join_utterance_separated_by(dialogs, separated_by=0.5):
    drefined = []

    lasts = [None for _ in range(2)]
    dic = {'A': 0, 'B': 1}
    for idx, curr in enumerate(dialogs):
        # If current text is entriely contained within the last utterance
        last_current = lasts[dic[curr['speaker']]]
        if last_current is None:
            lasts[dic[curr['speaker']]] = curr
            continue

        # Join utterances from current speaker < separated_by
        if last_current is not None and curr['start'] - last_current['end'] < separated_by:
            last_current['text'] += f" {curr['text']}"
            last_current['end'] = curr['end']
            last_current['wfeats'].extend(curr['wfeats'])

        else:
            drefined.append(last_current)
            lasts[dic[curr['speaker']]] = curr

    drefined.append(lasts[0])
    drefined.append(lasts[1])

    if all([x is None for x in drefined]):
        return []

    drefined.sort(key=lambda x: (x['start'], -x['end']))

    return drefined


def combine_dialogue_without_timings(dialog, separated_by=2):
    combined = dialog[0]
    combined.extend(dialog[1])
    combined.sort(key=lambda key: key['start'])

    combined = join_utterance_separated_by(
        combined, separated_by=separated_by)
    return combined


def remove_overlaps(dialogs):
    if len(dialogs) == 0:
        return [], []

    drefined = [dialogs[0]]
    overlaps = []
    for idx, curr in enumerate(dialogs[1:]):
        if drefined[-1]["start"] <= curr["start"] <= drefined[-1]["end"]:
            if drefined[-1]["start"] <= curr["end"] <= drefined[-1]["end"]:
                overlaps.append(curr)
                continue

        drefined.append(curr)

    return drefined, overlaps


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
    temp_dialogs.sort(key=lambda key: (key['start'], -key['end']))

    if len(temp_dialogs) == 0:
        return [], []

    combined_dialogs = [temp_dialogs[0]]
    combined_backchannels = []
    for idx in range(1, len(temp_dialogs)):
        if temp_dialogs[idx]['dialog_type'] == "dialog" and combined_dialogs[-1]['dialog_type'] == "dialog":
            if temp_dialogs[idx]['dialog_type'] in {"dialog", "backchannel"} and combined_dialogs[-1]['speaker'] == temp_dialogs[idx]['speaker']:
                combined_dialogs[-1]['text'] += f" {temp_dialogs[idx]['text']}"
                combined_dialogs[-1]['end'] = temp_dialogs[idx]['end']
                combined_dialogs[-1]['dialog_type'] = temp_dialogs[idx]['dialog_type']
                combined_dialogs[-1]['wfeats'].extend(
                    temp_dialogs[idx]['wfeats'])
            elif temp_dialogs[idx]['dialog_type'] == "backchannel":
                # backchannel is to be combined with the same speaker's utterance
                combined_backchannels.append(temp_dialogs[idx])
            else:
                combined_dialogs.append(temp_dialogs[idx])

    return combined_dialogs, combined_backchannels


def pairwise_remove_backchannels(dialogs, pre_silence=1, post_silence=1, bc_duration=1):
    dialogsA = [x for x in dialogs if x['speaker'] == 'A']
    dialogsB = [x for x in dialogs if x['speaker'] == 'B']

    if len(dialogsA) == 0 or len(dialogsB) == 0:
        return dialogs, []

    assert len(dialogsA) + len(dialogsB) == len(
        dialogs), f"Dialogs not separated by speaker: {len(dialogsA)} + {len(dialogsB)} != {len(dialogs)}; type(dialogs[0])={type(dialogs[0])}"

    def remove_bc_from_channel(dialogs, end_of_utterance_time=0):
        last_end = 0
        new_dialog = []
        new_bc = []
        for idx, dialog in enumerate(dialogs):
            bc_in = dialog['text'] in BACKCHANNELS

            # Pre silence is 1s, Post silence is 1s and Utterance Length is less than 1
            duration = dialog['end'] - dialog['start']
            pre_sil = dialog['start'] - last_end

            last_end = dialog['end']

            post_sil = end_of_utterance_time - dialog["end"]
            if idx != len(dialogs) - 1:
                post_sil = dialogs[idx+1]['start'] - dialog['end']

            if bc_in and duration <= bc_duration and pre_sil >= pre_silence and post_sil >= post_silence:
                new_bc.append(dialog)
                continue

            new_dialog.append(dialog)
        return new_dialog, new_bc

    end_of_utterance_time = max(dialogsA[-1]['end'], dialogsB[-1]['end'])
    new_dialogsA, new_bcA = remove_bc_from_channel(
        dialogsA, end_of_utterance_time)
    new_dialogsB, new_bcB = remove_bc_from_channel(
        dialogsB, end_of_utterance_time)

    new_dialogs = new_dialogsA + new_dialogsB
    new_bc = new_bcA + new_bcB

    new_dialogs.sort(key=lambda key: (key['start'], -key['end']))
    new_bc.sort(key=lambda key: (key['start'], -key['end']))

    return new_dialogs, new_bc


def remove_backchannels(dialogs, pre_silence=1, post_silence=1, bc_duration=1):
    new_dialogs, _ = pairwise_remove_backchannels(
        dialogs, pre_silence, post_silence, bc_duration)
    return new_dialogs


def pairwise_extract_dialog(filenames):
    trans_filename, words_filename = filenames

    words = read_txt(words_filename)
    utterance = []

    for word_row in words:
        word_row = word_row.strip()
        if word_row == "" or word_row[0] == "#":
            continue

        key, speaker, start, duration, word = word_row.split(" ")
        start = float(start)
        end = start + float(duration)
        speaker = speaker.replace(":", "")
        word = regexp(word)
        if len(word) > 0:
            utterance.append({
                "start": start,
                "end": end,
                "text": word,
                "speaker": speaker
            })


def separate_by_speaker(dialog_ord, conv_id=-1):
    new_dialogA = []
    new_dialogB = []
    for idx, turn in enumerate(dialog_ord):
        turn['conv_id'] = conv_id
        if turn['speaker'] == 'A':
            new_dialogA.append(turn)
        else:
            new_dialogB.append(turn)
    return {'speakerA': new_dialogA, 'speakerB': new_dialogB}


def read_word_timings_into_files():
    prev_key = -1
    with open("/home/seanleishman/BespokeTart/data/fisher/fisher/data/word_timings/word_timings.txt") as word_file:
        word_line = word_file.readline()
        while word_line:
            key, *values = word_line.split(" ")
            if prev_key != key:
                prev_key = key
                mid = key[6:9]
                write_dir = f"/home/seanleishman/BespokeTart/data/fisher/fisher/data/words/{mid}"
                if not os.path.isdir(write_dir):
                    os.mkdir(write_dir)

                wf = open(os.path.join(write_dir, f"{key}.txt"), "w")

            wf.write(f"{key} {' '.join(values)}")

            word_line = word_file.readline()


def write_indexes():
    filename = "/home/seanleishman/BespokeTart/data/fisher/fisher/data/word_timings/word_timings.txt"
    print("START READING")
    indexes = {}
    count = 0

    with open(filename) as f:
        line = f.readline()
        while line:
            key, *_ = line.split(" ")
            if key not in indexes:
                indexes[key] = count

            count += 1
            line = f.readline()
    print("END READING")

    print("WRITE INDEXES")
    with open("/home/seanleishman/BespokeTart/data/fisher/fisher/data/word_timings/indexes.txt", "w") as w:
        for k, v in indexes.items():
            w.write(f"{k} {v}\n")


def update_missing_turn_timings(buffer, prior_end=None):
    if buffer[-1]['end'] is None:
        print("Start or end is None")
        print(buffer)
        print()
        return False

    start = float(buffer[0]['start']
                  ) if buffer[0]['start'] is not None else prior_end
    end = float(buffer[-1]['end'])

    step = (end-start) / len(buffer)
    for idx, word in enumerate(buffer):
        word['start'] = round(start + idx * step, 2)
        word['end'] = round(word['start'] + step, 2)

    return True


def add_missing_timings(word_features, speaker="A"):
    buffer = []
    prior_end = 0
    for idx, turn in enumerate(word_features):
        for word in turn:
            if word['end'] is None:
                assert word['start'] is None or len(
                    buffer) == 0, "Start is None but end is not"
                buffer.append(word)
            elif word['start'] is None and word['end'] is not None:
                buffer.append(word)
                if not update_missing_turn_timings(buffer, prior_end=prior_end):
                    print("Buffer", buffer)
                    print("Turn", turn)
                buffer = []

            prior_end = word['end'] if word['end'] is not None else prior_end

        assert len(buffer) == 0, "Buffer not empty"

    return word_features


if __name__ == "__main__":
    read_word_timings_into_files()
