import re
import torch
import os
import json
import math
import logging


def get_abs_path(filepath):
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), filepath)


def read_json(path):
    with open(path, "r") as f:
        data = json.load(f)
    return data


def write_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f)


def read_txt(filename):
    with open(filename, "r") as f:
        lines = f.readlines()
    return lines


def remove_multiple_whitespace(s):
    s = re.sub(r"\t", " ", s)
    return re.sub(r"\s\s+", " ", s)


# Assume batch_idx already used to select correctly
def pp_single_dialogs(tokenizer, input_ids, curr, timings, others=[], offset=5):
    tt1 = input_ids[curr[0] : curr[1]]

    s1 = tokenizer.convert_ids_to_tokens(tt1)

    if len(s1) == 0:
        curr = [curr[0] + offset, curr[1] + offset]
        return curr, False

    t1 = [
        f"({round(x[0].item(), 2)}, {round(x[1].item(), 2)})"
        for x in timings[curr[0] : curr[1]]
    ]
    max_len = max(max(len(str(el)) for el in s1 + t1), 18)
    fs3 = " ".join(f"{t:<{max_len}}" for t in t1)

    fs1 = " ".join(f"{it:<{max_len}}" for it in s1)

    new_others = []
    for x in others:
        x = x.tolist()

        if isinstance(x[0], float):
            x = [round(i, 3) for i in x]

        other = " ".join(f"{x:<{max_len}}" for x in x[curr[0] : curr[1]])
        new_others.append(other)
    others = new_others

    br = False
    for i in range(len(s1)):
        if s1[i] != "<|endoftext|>":
            br = True

    curr = [curr[0] + offset, curr[1] + offset]

    print(f"A: {fs1}")
    print(f"T: {fs3}")
    for other in others:
        print(f"S: {other}")

    return curr, br


def str_pair_dialogs(
    curr_str,
    tokenizer,
    input_ids,
    start=0,
    timings=None,
    speaker="A",
    token_types=None,
    others={},
    columns=-1,
    column_width=17,
    width=80,
):
    max_len = 20
    num_of_columns = 0
    if width != -1:
        if columns == -1:
            tokens = tokenizer.convert_ids_to_tokens(input_ids[start:])
            for idx in range(start, len(input_ids)):
                tok = tokens[idx - start]
                max_len = max(max_len, len(str(tok)))

                num_of_columns = idx - start + 1
                if column_width + num_of_columns * max_len > width:
                    num_of_columns -= 1
                    break

            max_len = (width - 10) // (num_of_columns + 1)
        else:
            num_of_columns = columns
            max_len = (width - 10) // (num_of_columns + 1)

        if num_of_columns == 0:
            return curr_str, start, False
    else:
        num_of_columns = len(input_ids) - start

    curr = [start, start + num_of_columns]
    start = curr[1]
    tt1 = [x for x in input_ids[curr[0] : curr[1]] if x >= 0]

    s1 = tokenizer.convert_ids_to_tokens(tt1)

    t1 = []
    fs3 = ""

    fs1 = " ".join(f"{it:<{max_len}}" for it in s1)

    token_types_s = ""
    if token_types is not None:
        token_types_s = " ".join(
            f"{x:<{max_len}}" for x in token_types[curr[0] : curr[1]]
        )

    if timings is not None:
        if torch.is_tensor(timings):
            t1 = [
                f"{round(x[0].item(), 3), round(x[1].item(), 3)}"
                for x in timings[curr[0] : curr[1]]
            ]
        else:
            t1 = [
                f"{round(x[0], 3), round(x[1], 3)}" for x in timings[curr[0] : curr[1]]
            ]
        fs3 = " ".join(f"{t:<{max_len}}" for t in t1)

    new_others = {}
    for key, x in others.items():
        if torch.is_tensor(x):
            x = x.tolist()

        if len(x) == 0:
            continue

        if isinstance(x[0], float):
            x = [round(i, 3) for i in x]

        other = " ".join(f"{x:<{max_len}}" for x in x[curr[0] : curr[1]])
        new_others[key] = other
    others = new_others

    br = False
    for i in range(len(s1)):
        if s1[i] != "<|endoftext|>":
            br = True

    curr_str += f"{speaker:<{column_width}}: {fs1}\n"
    if fs3 != "":
        key = "Timings"
        curr_str += f"{key:<{column_width}}: {fs3}\n"
    if token_types != "":
        key = "Token Types"
        curr_str += f"{key:<{column_width}}: {token_types_s}\n"
    for key, value in others.items():
        curr_str += f"{key:<{column_width}}: {value}\n"

    return curr_str, start, num_of_columns, br


def pp_pair_dialogs(tokenizer, input_ids, **kwargs):
    curr_str, start, columns, br = str_pair_dialogs("", tokenizer, input_ids, **kwargs)
    print(curr_str)
    return start, columns, br


def retokenize(source_tokenizer, target_tokenizer, source):
    output = {k: {} for k in source.keys()}

    for speaker in source.keys():
        word_out = source_tokenizer.decode(source[speaker]["input_ids"])
        tokenized_out = target_tokenizer(word_out, end_ts=source_tokenizer.eos_token)
        output[speaker]["input_ids"] = torch.tensor(tokenized_out["input_ids"])
        output[speaker]["attention_mask"] = torch.tensor(
            tokenized_out["attention_mask"]
        )

        if "speaker_ids" in tokenized_out:
            output[speaker]["speaker_ids"] = torch.tensor(tokenized_out["speaker_ids"])

        output[speaker]["conv_id"] = source[speaker]["conv_id"]

    return output


def get_matching_utterance(dataset, instance):
    """
    Given an instance, find the matching utterance in the dataset

    Args:
        instance (dict): The instance to match (should be the trimmed combined instance)
    """
    for idx in range(len(dataset)):
        if dataset[idx]["speakerA"]["conv_id"] == instance["speakerA"]["conv_id"]:
            return get_matching_subset(dataset[idx], instance)

    raise ValueError("No matching utterance found")


def trim_instance(instance, trim_target):
    for speaker, data in instance.items():
        for key in data.keys():
            instance[speaker][key] = data[key][trim_target[0] : trim_target[1]]

    return instance


def get_matching_subset(source, target, trim_target=None):
    if trim_target is None:
        if "timings" not in source["speakerA"]:
            raise ValueError(
                f"No timing information found in source instance {source['speakerA'].keys()}"
            )
        trim_target = (0, len(source["speakerA"]["timings"]))
    target = trim_instance(target, trim_target)

    if "timings" not in target["speakerA"]:
        raise ValueError(
            f"No timing information found in target instance {target['speakerA'].keys()}"
        )

    startA, endA = (
        target["speakerA"]["timings"][0][0],
        target["speakerA"]["timings"][-1][1],
    )
    startB, endB = math.inf, math.inf
    if "speakerB" in target:
        startB, endB = (
            target["speakerB"]["timings"][0][0],
            target["speakerB"]["timings"][-1][1],
        )

    start, end = (startA, endA) if startA < startB and startA != -1 else (startB, endB)
    if start == -1 or end == -1:
        raise ValueError("Invalid start/end times")

    output = {k: {} for k in source.keys()}
    for speaker, data in source.items():
        start_idx = -1
        end_idx = -1

        for idx, timing in enumerate(data["timings"]):
            if timing[0] >= start and start_idx == -1:
                start_idx = idx
            if timing[1] >= end and end_idx == -1:
                end_idx = idx
                break

        if start_idx == -1 or end_idx == -1:
            raise ValueError("Invalid start/end times")

        for key in data.keys():
            output[speaker][key] = data[key][start_idx:end_idx]

    return output, target


def extract_turns_from_single_stream(instance, remove_tokens=torch.tensor([])):
    """
    Extracts the turns from the instance

    Args:
        instance (dict): The instance to extract turns from

    Returns:
        list: A list of turns
    """
    speaker = "speakerA"
    speaker_id = None
    turns = []

    conv_id = instance[speaker]["conv_id"]
    speaker_ids = instance[speaker]["speaker_ids"]
    input_ids = instance[speaker]["input_ids"]
    if "attention_mask" not in instance[speaker]:
        attention_mask = torch.ones_like(input_ids)
    else:
        attention_mask = instance[speaker]["attention_mask"]

    if "timings" not in instance[speaker]:
        timings = torch.zeros_like(input_ids)
    else:
        timings = instance[speaker]["timings"]

    current_turn = {
        "conv_id": conv_id,
        "speaker_ids": [],
        "input_ids": [],
        "attention_mask": [],
        "timings": [],
    }

    for idx in range(len(input_ids)):
        if speaker_id != speaker_ids[idx]:
            # New turn
            if len(current_turn["input_ids"]) > 0:
                turns.append(current_turn)

            speaker_id = speaker_ids[idx]
            current_turn = {
                "conv_id": conv_id,
                "speaker_ids": [],
                "input_ids": [],
                "attention_mask": [],
                "timings": [],
            }

        current_turn["input_ids"].append(input_ids[idx])
        current_turn["speaker_ids"].append(speaker_ids[idx])
        current_turn["attention_mask"].append(attention_mask[idx])
        current_turn["timings"].append(timings[idx])

    turns.append(current_turn)
    for turn in turns:
        eot_idx = torch.isin(torch.tensor(turn["input_ids"]), remove_tokens)
        for key in turn.keys():
            if isinstance(turn[key], list):
                turn[key] = torch.tensor(turn[key])
                turn[key] = turn[key][~eot_idx]
            else:
                turn[key] = turn[key]
    return turns


def decode_turns(turns, tokenizer):
    """
    Adds `text` field to each dictionary in list of turns.
    Consists of the decoded text from the input_ids
    """

    for turn in turns:
        turn["text"] = tokenizer.decode(turn["input_ids"])

    return turns


def extract_speakerA_speakerB(dialog):
    speakerA = []
    speakerB = []
    for idx in range(len(dialog)):
        if dialog[idx]["speaker_ids"][0] == 0:
            speakerA.append(dialog[idx])
        elif dialog[idx]["speaker_ids"][0] == 1:
            speakerB.append(dialog[idx])
        else:
            raise ValueError("Invalid speaker ID")

    return speakerA, speakerB


def get_logger(namespace=__name__, level=logging.DEBUG):
    logger = logging.getLogger(namespace)
    format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    formatter = logging.Formatter(
        format,
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger.setLevel(level)

    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger
