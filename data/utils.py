import re


def read_txt(filename):
    with open(filename, "r") as f:
        lines = f.readlines()
    return lines


def remove_multiple_whitespace(s):
    s = re.sub(r"\t", " ", s)
    return re.sub(r"\s\s+", " ", s)


# Assume batch_idx already used to select correctly
def pp_single_dialogs(tokenizer, input_ids, curr, timings, others=[], offset=5):
    tt1 = input_ids[curr[0]:curr[1]]
    si1 = [el for el in tokenizer.decode(tt1).replace(
        "<emp>", " <emp> ").split(" ") if el != '']

    s1 = tokenizer.convert_ids_to_tokens(tt1)

    if len(s1) == 0:
        curr = [curr[0] + offset, curr[1] + offset]
        return curr, False

    t1 = [
        f"({round(x[0].item(),2)}, {round(x[1].item(), 2)})" for x in timings[curr[0]:curr[1]]]
    max_len = max(max(len(str(el)) for el in s1 + t1), 18)
    fs3 = " ".join(f"{t:<{max_len}}" for t in t1)

    fs1 = " ".join(f"{it:<{max_len}}" for it in s1)

    new_others = []
    for x in others:
        x = x.tolist()

        if isinstance(x[0], float):
            x = [round(i, 3) for i in x]

        other = " ".join(
            f"{x:<{max_len}}" for x in x[curr[0]:curr[1]])
        new_others.append(other)
    others = new_others

    br = False
    for i in range(len(s1)):
        if s1[i] != '<|endoftext|>':
            br = True

    curr = [curr[0] + offset, curr[1] + offset]

    print(f"A: {fs1}")
    print(f"T: {fs3}")
    for other in others:
        print(f"S: {other}")

    return curr, br


def pp_pair_dialogs(tokenizer, input_ids, curr=[0, 5], timings=None, speaker='A', token_types=None, others={}, offset=5):
    tt1 = [x for x in input_ids[curr[0]:curr[1]] if x >= 0]
    si1 = [el for el in tokenizer.decode(tt1).replace(
        "<emp>", " <emp> ").split(" ") if el != '']

    s1 = tokenizer.convert_ids_to_tokens(tt1)

    if len(s1) == 0:
        curr = [curr[0] + offset, curr[1] + offset]
        return curr, False

    t1 = []
    fs3 = ""
    if timings is not None:
        t1 = [
            f"{round(x[0].item(),2),round(x[1].item(), 2)}" for x in timings[curr[0]:curr[1]]]
        fs3 = " ".join(f"{t}" for t in t1)

    max_len = max(max(len(str(el)) for el in s1 + t1), 13)
    fs1 = " ".join(f"{it:<{max_len}}" for it in s1)

    token_types_s = ""
    if token_types is not None:
        token_types_s = " ".join(
            f"{x:<{max_len}}" for x in token_types[curr[0]:curr[1]])

    new_others = {}
    for key,x in others.items():
        x = x.tolist()

        if len(x) == 0:
            continue

        if isinstance(x[0], float):
            x = [round(i, 3) for i in x]

        other = " ".join(
            f"{x:<{max_len}}" for x in x[curr[0]:curr[1]])
        new_others[key] = other
    others = new_others

    br = False
    for i in range(len(s1)):
        if s1[i] != '<|endoftext|>':
            br = True

    curr = [curr[0] + offset, curr[1] + offset]

    print(f"{speaker:<12}: {fs1}")
    if fs3 != "":
        key = "Timings"
        print(f"{key:<12}: {fs3}")
    if token_types != "":
        key = "Token Types"
        print(f"{key:<12}: {token_types_s}")
    for key, value in others.items():
        print(f"{key:<12}: {value}")

    return curr, br
