from data.serialised_process import SerialisedProcessType
from data.generation_dm import GenerationDM
from data.pairwise_generation_dm import Datasets
from pairwisegpt.tokenizer import SpokenDialogTokenizer

import argparse
import torch

tokenizer = SpokenDialogTokenizer()


def load_generation_dm(split="test", single_stream=True, serialised=None, *args, **kwargs):
    dm = GenerationDM(split=split, tokenizer=tokenizer, split_utt=False, datasets=[
                      Datasets.SWITCHBOARD], combine_speaker=single_stream, serialised=serialised, *args, **kwargs)
    dm.prepare_data()

    return dm


def print_base(aligned, single, conv_id):
    aligned_it = iter(aligned.show_input_iterator(conv_id=conv_id))
    single_it = iter(single.show_input_iterator(conv_id=conv_id))

    aligned_item = next(aligned_it)
    single_item = next(single_it)

    while aligned_item is not None and single_item is not None:
        print(f"Aligned Turn:")
        aligned_item = next(aligned_it, None)

        print(f"Single Turn:")
        single_item = next(single_it, None)

        print("=====================================")

        go = input()
        if go == "q":
            break


def compare():
    aligned = load_generation_dm(
        single_stream=False, include_partial_overlaps=True, include_backchannels=True, include_overlaps=True)
    single = load_generation_dm(
        single_stream=True, serialised=SerialisedProcessType.TurnEnd)

    for idx in range(len(aligned)):
        print_base(aligned, single, aligned[idx]["speakerA"]["conv_id"])


def print_dialog(conv_id):
    aligned = load_generation_dm(
        split="val", single_stream=False, include_partial_overlaps=True, include_backchannels=True, include_overlaps=True)
    single = load_generation_dm(
        split="val", single_stream=True, serialised=SerialisedProcessType.TurnEnd)

    for idx in range(len(aligned)):
        if conv_id in aligned[idx]["speakerA"]["conv_id"]:
            print_base(aligned, single, aligned[idx]["speakerA"]["conv_id"])
            break


def find_err_ebc():
    ebc_token_id = tokenizer.convert_tokens_to_ids("<ebc>")
    serialised = load_generation_dm(
        single_stream=True, serialised=SerialisedProcessType.TurnEnd)
    aligned = load_generation_dm(
        single_stream=False, include_partial_overlaps=True, include_backchannels=True, include_overlaps=True)

    for idx in range(len(serialised)):
        num_ebc = torch.eq(
            serialised[idx]["speakerA"]["input_ids"], ebc_token_id).sum()

        if num_ebc == 0:
            continue

        print(
            f'Found <ebc> at conv_id={serialised[idx]["speakerA"]["conv_id"]} with {num_ebc} occurrences')
        serialised.show_input(serialised[idx])
        print("=================================")
        aligned.show_input(conv_id=serialised[idx]["speakerA"]["conv_id"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--find-err-ebc", action="store_true")
    parser.add_argument("--print-dialog", type=str, default=None)
    args = parser.parse_args()

    if args.compare:
        compare()
    elif args.find_err_ebc:
        find_err_ebc()
    elif args.print_dialog is not None:
        print_dialog(args.print_dialog)
    else:
        print("No action specified. Running compare()")
        compare()
