from torch.utils.data import Dataset

from data.edacc.utils import (
    extract_dialog,
    combine_dialogue_without_timings,
    pairwise_remove_backchannels,
    remove_overlaps,
    combine_consecutive_trps,
    insert_overlapped_bc,
    separate_by_speaker,
    save_words_in_file,
)
import tqdm
import os
import logging
import multiprocessing


def get_abs_path(filepath):
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), filepath)


def find_file(folder, filename):
    return os.path.join(TRANSCRIPT_DIRECTORY, folder, filename)


TRANSCRIPT_DIRECTORY = get_abs_path("edacc/edacc_v1.0/")


FILENAMES_FILE = get_abs_path(
    "switchboard/switchboard1/transcriptions/swb_ms98_transcriptions/AAREADME.text"
)

SMALL_SET_SIZE = 10


class EdAccDataset(Dataset):
    def __init__(
        self,
        split="train",
        remove_backchannels=False,
        pre_silence=1,
        post_silence=1,
        bc_duration=1,
        trp_separated_by=0.5,
        set_file_splits=False,
        set_split_word_text_files=True,
        dev_mode=False,
        parse_dialogs=None,
    ):
        self.logger = logging.getLogger(__name__)
        self.split = split
        self.remove_backchannels = remove_backchannels
        self.dev_mode = dev_mode

        self.pre_silence = pre_silence
        self.post_silence = post_silence
        self.bc_duration = bc_duration
        self.trp_separated_by = trp_separated_by

        self.set_file_splits = set_file_splits
        self.set_split_word_text_files = set_split_word_text_files

        self.parse_dialogs = parse_dialogs

        self.conversation_filenames = []
        self.word_filenames = []
        self.turn_filenames = []

        self.parsed_conv_ids = []

    def __len__(self):
        return len(self.dialogs)

    def __getitem__(self, idx):
        return self.dialogs[idx]

    def __str__(self):
        return "EdAcc"

    def __call__(self):
        if self.set_split_word_text_files:
            self.split_word_text_files()

        _ = self.load_dialog()
        self.dialogs = self.read_dialog()
        return self

    def get_batch(self, start=0, batch_size=4):
        return self.dialogs[start : start + batch_size]

    def _get_folders(self):
        folders = []
        if self.split == "val" or self.split == "train":
            folders = ["dev"]
        elif self.split == "test":
            folders = ["dev", "test"]
        elif self.split == "all":
            folders = ["dev", "test"]
        else:
            raise ValueError(f"Invalid split: {self.split}")

        self.logger.warning(f"Using {folders} for {self.split} split")
        return folders

    def split_word_text_files(self):
        folders = self._get_folders()

        conversation_filenames = [
            (folder, self.read_convs(find_file(folder, "conv.list")))
            for folder in folders
        ]
        self.conversation_filenames = [
            (x[0], y) for x in conversation_filenames for y in x[1]
        ]

        for folder in folders:
            save_words_in_file(
                find_file(folder, "company.ctm"), find_file(folder, "split_words")
            )
            save_words_in_file(
                find_file(folder, "stm"), find_file(folder, "split_text")
            )

        self.logger.info("File splits saved")

    def load_dialog(self):
        self.logger.info(
            f"Loading EdAcc Dataset for {self.split} split into memory ..."
        )

        folders = self._get_folders()

        conversation_filenames = [
            (folder, self.read_convs(find_file(folder, "conv.list")))
            for folder in folders
        ]
        self.conversation_filenames = [
            (x[0], y) for x in conversation_filenames for y in x[1]
        ]
        if self.dev_mode:
            self.conversation_filenames = self.conversation_filenames[:SMALL_SET_SIZE]

        self.word_filenames = [
            (conv_id, find_file(folder, os.path.join("split_words", f"{conv_id}.txt")))
            for folder, conv_id in self.conversation_filenames
        ]
        self.turn_filenames = [
            (conv_id, find_file(folder, os.path.join("split_text", f"{conv_id}.txt")))
            for folder, conv_id in self.conversation_filenames
        ]

    def read_convs(self, conv_filename):
        with open(conv_filename, "r") as f:
            lines = f.readlines()
            convs = []
            for line in lines:
                convs.append(line.strip())
            return convs

    def read_dialog(self, words_filename="", turns_filename="", timings_filename=""):
        self.logger.info(f"Loading EdAcc Dataset for {self.split} split ...")

        results = []

        with multiprocessing.Manager() as manager:
            shared_list = manager.list()
            with tqdm.tqdm(
                self.conversation_filenames, desc="EdAcc Processing"
            ) as progress_bar:

                def update(*a):
                    progress_bar.update(1)

                with multiprocessing.Pool(processes=1) as pool:
                    for idx in range(len(self.conversation_filenames)):
                        conv_id = self.conversation_filenames[idx][1]
                        word_conv_id, word_filename = self.word_filenames[idx]
                        turn_conv_id, turn_filename = self.turn_filenames[idx]

                        assert conv_id == word_conv_id == turn_conv_id

                        results.append(
                            pool.apply_async(
                                self._read_dialog,
                                args=(
                                    conv_id,
                                    word_filename,
                                    turn_filename,
                                    shared_list,
                                ),
                                callback=update,
                            )
                        )

                    for result in results:
                        result.get()

                    pool.close()
                    pool.join()

            self.parsed_conv_ids = [
                conv["dialog"]["speakerA"][0]["conv_id"] for conv in shared_list
            ]
            return list(shared_list)

    def _read_dialog(self, conv_id, word_filename, turn_filename, shared_list):
        dialog = extract_dialog(conv_id, word_filename, turn_filename)
        if len(dialog) == 0:
            return

        dialog = combine_dialogue_without_timings(
            dialog, separated_by=self.trp_separated_by
        )
        dialog, backchannels = pairwise_remove_backchannels(
            dialog, self.pre_silence, self.post_silence, self.bc_duration
        )
        dialog, overlaps = remove_overlaps(dialog)
        dialog = combine_consecutive_trps(dialog)
        dialog, backchannels = insert_overlapped_bc(dialog, backchannels)

        conv_id = dialog[0]["conv_id"]
        dialog_dict = {}
        dialog_dict["dialog"] = separate_by_speaker(dialog, conv_id=conv_id)
        dialog_dict["backchannel"] = separate_by_speaker(backchannels, conv_id=conv_id)
        dialog_dict["overlap"] = separate_by_speaker(overlaps, conv_id=conv_id)

        shared_list.append(dialog_dict)

    def save_dialogs(self, prefix_dir):
        # Assume self.filenames correspond with self.dialogs
        for idx, key in enumerate(self.parsed_conv_ids):
            filename = key
            filename = os.path.join(prefix_dir, filename)

            if idx >= len(self.dialogs):
                return

            with open(filename, "w") as f:
                f.writelines(self.dialogs[idx])


if __name__ == "__main__":
    ed = EdAccDataset(remove_backchannels=True, set_split_word_text_files=True)
    ed()
