import os
import random

from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from data.switchboard.utils import (
    extract_dialog,
    remove_backchannels,
    combine_consecutive_trps,
    remove_overlaps,
    combine_dialogue_without_timings,
    pairwise_extract_dialog,
    separate_by_speaker,
    pairwise_remove_backchannels,
    insert_overlapped_bc,
)
from datasets import Dataset as HFDataset
from data.utils import get_logger
from huggingface_hub import snapshot_download


def get_abs_path(filepath):
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), filepath)


TRANSCRIPT_DIRECTORIES = [
    # get_abs_path("switchboard/cellular/transcripts/data"),
    get_abs_path("switchboard/switchboard1/transcriptions/swb_ms98_transcriptions")
]

FILENAMES_FILE = get_abs_path(
    "switchboard/switchboard1/transcriptions/swb_ms98_transcriptions/AAREADME.text"
)

SMALL_SET_SIZE = 10

logger = get_logger(__name__)


class SwitchboardDataset(Dataset):
    def __init__(
        self,
        split="train",
        pairwise=True,
        remove_backchannels=False,
        pre_silence=1,
        post_silence=1,
        bc_duration=1,
        trp_separated_by=0.5,
        set_file_splits=False,
        keep_in_memory=False,
        load_from_hub=False,
        save_to_disk=True,
        dev_mode=False,
        parse_dialogs=None,
    ):
        self.split = split
        self.pairwise = pairwise
        self.remove_backchannels = remove_backchannels
        self.dev_mode = dev_mode

        self.pre_silence = pre_silence
        self.post_silence = post_silence
        self.bc_duration = bc_duration
        self.trp_separated_by = trp_separated_by

        self.keep_in_memory = keep_in_memory
        self.data_dir = get_abs_path(os.path.join("switchboard", ".cache"))

        self.load_from_hub = load_from_hub
        self.save_to_disk = save_to_disk

        self.set_file_splits = set_file_splits

        self.parse_dialogs = parse_dialogs

    def __len__(self):
        if self.dialogs is None:
            return 0

        return len(self.dialogs)

    def __getitem__(self, idx):
        if self.dialogs is None:
            raise ValueError("No dialogs loaded")

        return self.dialogs[idx]

    def __str__(self):
        return "Switchboard"

    def __call__(self):
        if self.load_from_hub:
            self.dialogs = self._load_from_hub()
            logger.info("Loaded from HuggingFace Hub")
            return self

        self.filenames = self.read_files()
        self.dialogs = self.read_dialog()
        return self

    def _load_from_hub(self):
        repo_id = "seanleishman/Switchboard"
        filename = self.generate_save_path()

        if not os.path.exists(self.data_dir):
            os.mkdir(self.data_dir)

        # Download entire repo in .cache
        # If exists then uses local cache
        local_repo_path = snapshot_download(
            repo_id=repo_id, repo_type="dataset", token=os.getenv("HUGGINGFACE_TEST")
        )

        if not os.path.exists(local_repo_path):
            raise ValueError(f"Failed to download {repo_id}")

        requested_path = os.path.join(local_repo_path, filename + ".hf")
        ds = HFDataset.load_from_disk(requested_path)

        return ds

    def get_batch(self, start=0, batch_size=4):
        if self.dialogs is None:
            raise ValueError("No dialogs loaded")

        for idx in range(start, start + batch_size):
            yield self.dialogs[idx]

        # return self.dialogs[start : start + batch_size]

    def generate_save_path(self):
        pairwise = "pairwise" if self.pairwise else ""
        remove_bc = "remove_bc" if self.remove_backchannels else ""
        pre_silence = f"pre_silence_{self.pre_silence}" if self.pre_silence != 0 else ""
        post_silence = (
            f"post_silence_{self.post_silence}" if self.post_silence != 0 else ""
        )
        bc_duration = f"bc_duration_{self.bc_duration}" if self.bc_duration != 0 else ""
        trp_separated_by = (
            f"trp_separated_by_{self.trp_separated_by}"
            if self.trp_separated_by != 0
            else ""
        )
        dev_mode = "debug" if self.dev_mode else ""

        return f"Switchboard_{'_'.join(
            [
                self.split,
                pairwise,
                remove_bc,
                pre_silence,
                post_silence,
                bc_duration,
                trp_separated_by,
                dev_mode,
            ])}"

    def read_file_splits(self):
        logger.info(f"Reading Switchboard file splits for {self.split} split ...")

        filename = get_abs_path(os.path.join("splits", f"{self.split}.txt"))
        if not os.path.isfile(filename) or self.set_file_splits:
            logger.error(f"data: no splits {self.split} file found at {filename}")
            self.generate_file_splits()

        split_filenames = {}
        with open(filename) as f:
            line = f.readline()
            while line:
                values = line.split("\t")
                conv_id = values[0].strip()

                prefix_dict = os.path.join(
                    TRANSCRIPT_DIRECTORIES[0], conv_id[:2], conv_id
                )

                split_filenames[conv_id] = []
                filenames = os.listdir(prefix_dict)
                filenames.sort()
                if self.dev_mode and self.parse_dialogs is None:
                    filenames = filenames[:SMALL_SET_SIZE]

                for file in filenames:
                    split_filenames[conv_id].append(
                        os.path.join(prefix_dict, file.strip())
                    )

                line = f.readline()

                if self.dev_mode and self.parse_dialogs is None:
                    if len(split_filenames) == SMALL_SET_SIZE:
                        break

        return split_filenames

    def read_files(self):
        if self.split == "all" or self.parse_dialogs is not None:
            self.split = "train"
            train_files = self.read_file_splits()
            self.split = "test"
            test_files = self.read_file_splits()
            self.split = "val"
            val_files = self.read_file_splits()

            self.split = "all"

            if self.parse_dialogs is not None:
                print(f"Parse Dialogs from {self.parse_dialogs}")
                files = {}
                for file in self.parse_dialogs:
                    new_dict = {**train_files, **val_files, **test_files}
                    if file in new_dict:
                        files[file] = new_dict[file]
                    else:
                        print(
                            f"Parse Dialogs could not find {file} with example key {new_dict.keys()}"
                        )

                return files

            return {**train_files, **val_files, **test_files}

        return self.read_file_splits()

    def generate_file_splits(self):
        if not os.path.exists(get_abs_path("splits")):
            os.mkdir(get_abs_path("splits"))

        line_idx = 0
        files = {}
        lines = []
        with open(FILENAMES_FILE) as f:
            line = f.readline()
            while line:
                if line_idx >= 17:
                    if (line_idx - 17) % 5 == 0:
                        if len(lines) >= 5:
                            files[lines[0]] = [lines[1], lines[2], lines[3], lines[4]]
                            lines = []
                    lines.append(line.strip())
                line_idx += 1
                line = f.readline()

        train, test = train_test_split(list(files.keys()), test_size=0.2, shuffle=False)
        val, test = train_test_split(test, test_size=0.5, shuffle=False)

        train_filename = get_abs_path("splits/train.txt")
        val_filename = get_abs_path("splits/val.txt")
        test_filename = get_abs_path("splits/test.txt")

        with open(train_filename, "w") as f:
            for key in train:
                value = files[key]
                f.write(f"{key}\t{value[0]}\t{value[1]}\t{value[2]}\t{value[3]}\n")

        with open(test_filename, "w") as f:
            for key in test:
                value = files[key]
                f.write(f"{key}\t{value[0]}\t{value[1]}\t{value[2]}\t{value[3]}\n")

        with open(val_filename, "w") as f:
            for key in val:
                value = files[key]
                f.write(f"{key}\t{value[0]}\t{value[1]}\t{value[2]}\t{value[3]}\n")

    def read_dialog(self):
        logger.info(f"Loading Switchboard Dataset for {self.split} split ...")
        NUM_WORKERS = min(24, os.cpu_count() - 1) if not self.dev_mode else 1

        ds = HFDataset.from_generator(
            self._read_dialog,
            gen_kwargs={
                "filenames": self.filenames,
                "_misc": random.random(),  # [random.random () for _ in range(len(self.filenames))],
            },
            num_proc=NUM_WORKERS,
            keep_in_memory=True,
        )
        if self.save_to_disk:
            ds.save_to_disk(
                os.path.join(self.data_dir, f"{self.generate_save_path()}.hf")
            )

        return ds

    def _read_dialog(self, filenames=[], _misc=[]):
        logger.info(f"_read_dialog with {len(filenames)} files and _misc {_misc}")
        for key, filename in filenames.items():
            item = self._process_dialog(key, filename)
            if item is not None:
                yield item

    def _process_dialog(self, key, filename):
        if self.pairwise:
            dialog = pairwise_extract_dialog(filename)

            dialog_ord = extract_dialog(filename)
            dialog_ord = combine_dialogue_without_timings(
                dialog_ord, separated_by=self.trp_separated_by
            )
            dialog_ord, backchannels = pairwise_remove_backchannels(
                dialog_ord, self.pre_silence, self.post_silence, self.bc_duration
            )
            dialog_ord, overlaps = remove_overlaps(dialog_ord)
            dialog_ord = combine_consecutive_trps(dialog_ord)
            dialog_ord, backchannels = insert_overlapped_bc(dialog_ord, backchannels)

            new_dialog = {}
            conv_id = dialog[0][0]["conv_id"]
            new_dialog["dialog"] = separate_by_speaker(dialog_ord, conv_id=conv_id)
            new_dialog["backchannel"] = separate_by_speaker(
                backchannels, conv_id=conv_id
            )
            new_dialog["overlap"] = separate_by_speaker(overlaps, conv_id=conv_id)
            dialog = new_dialog
        else:
            dialog = extract_dialog(filename)
            dialog = combine_dialogue_without_timings(dialog)

            if self.remove_backchannels:
                dialog = remove_backchannels(
                    dialog, self.pre_silence, self.post_silence, self.bc_duration
                )

            dialog, _ = remove_overlaps(dialog)
            dialog = combine_consecutive_trps(dialog)
            for x in dialog:
                x["key"] = key

        return dialog

    def save_dialogs(self, prefix_dir):
        # Assume self.filenames correspond with self.dialogs
        for idx, key in enumerate(self.filenames):
            filename = key
            filename = os.path.join(prefix_dir, filename)

            if idx >= len(self.dialogs):
                return

            with open(filename, "w") as f:
                f.writelines(self.dialogs[idx])


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--load-from-hub", action="store_true")

    sd = SwitchboardDataset(load_from_hub=True, save_to_disk=False)
    sd()
