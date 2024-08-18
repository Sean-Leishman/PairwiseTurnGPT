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
    "oh uhhuh"
    "uh"
    "uhhuh uhhuh",
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
        return textA, 'A'
    elif startB > startA and endB < endA:
        return textB, 'B'

    return None, None


def _check_overlap_silence(past_line, next_line, thresh=1):
    past_text, past_start, past_end = _read_transcript_line(past_line)
    next_text, next_start, next_end = _read_transcript_line(next_line)

    if past_text != '[silence]' or next_text != '[silence]':
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
    # print_transcript_timing(transcript, out)
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
        trans_filenameA,
        words_filenameA,
        speaker='A'
    )
    utterancesB = _extract_utterance_word_feats(
        trans_filenameB,
        words_filenameB,
        speaker='B'
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
                word_feats[key] = [{
                    "word": word,
                    "start": float(start),
                    "end": float(end),
                }]
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
            new_dialog[speaker].append({
                "text": utterance["text"],
                "start": utterance["start"],
                "end": utterance["end"]
            })

    return new_dialog


"""
Only combines based on turns identified within the structure of the conversation and so based 
on the start of an utterance without consideration of the word level
"""


def combine_dialogue_without_timings(dialogue):
    combined = dialogue[0]
    combined.extend(dialogue[1])
    combined.sort(key=lambda key: (key['start'], -key['end']))

    combined = join_utterance_separated_by(combined)
    return combined


def _pp_dialogue(dialogue):
    out = ""
    start = 0
    curr_speaker = None

    for idx in range(len(dialogue)):
        if curr_speaker is None or curr_speaker != dialogue[idx]['speaker']:
            curr_speaker = dialogue[idx]['speaker']
            out += f": {start} - {dialogue[idx-1]['end']}"
            start = dialogue[idx]['start']
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
    dic = {'A': 0, 'B': 1}
    for idx, curr in enumerate(dialogs):
        # If current text is entriely contained within the last utterance
        # from the same speaker
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

    drefined.sort(key=lambda x: (x['start'], -x['end']))
    return drefined


def pairwise_remove_overlaps(dialogs, speakers=2):
    drefined = []
    overlaps = []

    lasts = [None for _ in range(speakers)]
    dic = {'A': 0, 'B': 1}
    for idx, curr in enumerate(dialogs):
        # If current text is entriely contained within the last utterance
        last = lasts[not dic[curr['speaker']]]
        last_current = lasts[dic[curr['speaker']]]

        if last is not None and last["start"] <= curr["start"] <= last["end"]:
            if last["start"] <= curr["end"] <= last["end"]:
                overlaps.append(curr)
                continue

        lasts[dic[curr['speaker']]] = curr
        drefined.append(curr)
    return drefined, overlaps


def remove_backchannels(dialogs, pre_silence=1, post_silence=1, bc_duration=1):
    new_dialog, _ = pairwise_remove_backchannels(
        dialogs, pre_silence, post_silence, bc_duration)
    return new_dialog


def pairwise_remove_backchannels(dialogs, pre_silence=1, post_silence=1, bc_duration=1):
    dialogsA = [x for x in dialogs if x['speaker'] == 'A']
    dialogsB = [x for x in dialogs if x['speaker'] == 'B']

    assert len(dialogsA) + len(dialogsB) == len(dialogs), "Missing speaker tag"

    def remove_bc_from_channel(dialogs):
        last_end = 0
        new_dialog = []
        new_bc = []
        for idx, dialog in enumerate(dialogs):
            bc_in = dialog['text'] in BACKCHANNELS

            # Pre silence is 1s, Post silence is 1s and Utterance Length is less than 1
            duration = dialog['end'] - dialog['start']
            pre_sil = dialog['start'] - last_end

            last_end = dialog['end']

            post_sil = 0.1
            if idx != len(dialogs) - 1:
                post_sil = dialogs[idx+1]['start'] - dialog['end']

            if bc_in and duration < bc_duration and pre_sil > pre_silence and post_sil > post_silence:
                new_bc.append(dialog)
                continue

            new_dialog.append(dialog)
        return new_dialog, new_bc

    new_dialogsA, new_bcA = remove_bc_from_channel(dialogsA)
    new_dialogsB, new_bcB = remove_bc_from_channel(dialogsB)

    new_dialogs = new_dialogsA + new_dialogsB
    new_bc = new_bcA + new_bcB

    new_dialogs.sort(key=lambda key: (key['start'], -key['end']))
    new_bc.sort(key=lambda key: (key['start'], -key['end']))

    return new_dialogs, new_bc


"""
Actually just need to convert into format for parent dataset.
Where in __getitem__(idx) idx refers to the conversation and returns all turns within
a conversation
So this function just needs to return the turn list for a conversation
"""


def combine_consecutive_trps(dialogs):
    combined_dialogs = [dialogs[0]]
    for idx in range(1, len(dialogs)):
        if combined_dialogs[-1]['speaker'] == dialogs[idx]['speaker']:
            combined_dialogs[-1]['text'] += f" {dialogs[idx]['text']}"
            combined_dialogs[-1]['end'] = dialogs[idx]['end']
            combined_dialogs[-1]['wfeats'].extend(dialogs[idx]['wfeats'])
        else:
            combined_dialogs.append(dialogs[idx])
    return combined_dialogs


def extract_word_features(dialog):
    utterancesA = []
    utterancesB = []

    for d in dialog[0]:
        utterancesA.extend(d['wfeats'])
    for d in dialog[1]:
        utterancesB.extend(d['wfeats'])

    return [utterancesA, utterancesB]


def pairwise_extract_dialog(filenames):
    trans_filenameA, words_filenameA, trans_filenameB, words_filenameB = filenames

    utterancesA = _pairwise_extract_utterance_word_feats(
        trans_filenameA,
        words_filenameA,
        speaker='A'
    )
    utterancesB = _pairwise_extract_utterance_word_feats(
        trans_filenameB,
        words_filenameB,
        speaker='B'
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


def separate_by_speaker(dialog, dual_dialog=None, conv_id=0):
    speakerA = []
    speakerB = []

    word_countA = 0
    word_countB = 0

    for idx, utterance in enumerate(dialog):
        dialog[idx]['conv_id'] = conv_id
        if utterance['speaker'] == 'A':
            speakerA.append(utterance)

            # Processing backchannels or overlaps
            if dual_dialog is None:
                continue

            # Ensure processing is equal for pairwise and ordinary words
            word_countA += len(utterance['wfeats'])
        elif utterance['speaker'] == 'B':
            speakerB.append(utterance)

            # Processing backchannels or overlaps
            if dual_dialog is None:
                continue

            word_countB += len(utterance['wfeats'])
        else:
            raise Exception(f"No label for: {utterance['speaker']}")

    return {'speakerA': speakerA, 'speakerB': speakerB}


def remove_backchannels2(dialogs):
    last_endA = 0
    last_endB = 0

    backchannelA = None
    backchannelB = None

    output = []

    for idx in range(len(dialogs)):
        if dialogs[idx]['speaker'] == 'A':
            if backchannelA is not None:
                if not _remove_backchannel(backchannelA, dialogs[idx]):
                    output.append(backchannelA)

            backchannelA = _potential_backchannel(last_endA, dialogs[idx])
            if backchannelA is None:
                output.append(dialogs[idx])
            last_endA = dialogs[idx]['end']
        else:
            if backchannelB is not None:
                if not _remove_backchannel(backchannelB, dialogs[idx]):
                    output.append(backchannelB)

            backchannelB = _potential_backchannel(last_endB, dialogs[idx])
            if backchannelB is None:
                output.append(dialogs[idx])
            last_endB = dialogs[idx]['end']

    # _pp_dialogue(dialogs)
    # _pp_dialogue(output)
    return output


def _remove_backchannel(backchannelA, dialog):
    return (dialog['start'] - backchannelA['end']) > 1


def _potential_backchannel(last_phrase, current_phrase):
    if (current_phrase['start'] - last_phrase) > 1 and current_phrase['text'] in BACKCHANNELS:
        return current_phrase

    return None
