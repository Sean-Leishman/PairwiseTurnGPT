from data.switchboard.utils import extract_dialog, extract_speaker_timings, \
    remove_backchannels, combine_consecutive_trps, remove_overlaps, \
    combine_dialogue_without_timings, extract_word_features, pairwise_extract_dialog, remove_backchannels2, \
    separate_by_speaker, pairwise_remove_backchannels, pairwise_remove_overlaps
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
from torch.utils.data import Dataset
import tqdm
import os
import logging


def get_abs_path(filepath):
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), filepath)


TRANSCRIPT_DIRECTORIES = [
    # get_abs_path("switchboard/cellular/transcripts/data"),
    get_abs_path(
        "switchboard/switchboard1/transcriptions/swb_ms98_transcriptions")
]

FILENAMES_FILE = get_abs_path(
    "switchboard/switchboard1/transcriptions/swb_ms98_transcriptions/AAREADME.text")

SMALL_SET_SIZE = 10


class SwitchboardDataset(Dataset):
    def __init__(self,
                 split="train",
                 pairwise=False,
                 remove_backchannels=False,
                 pre_silence=1,
                 post_silence=1,
                 bc_duration=1,
                 dev_mode=False,
                 parse_dialogs=None,
                 ):
        self.logger = logging.getLogger(__name__)
        self.split = split
        self.pairwise = pairwise
        self.remove_backchannels = remove_backchannels
        self.dev_mode = dev_mode

        self.pre_silence = pre_silence
        self.post_silence = post_silence
        self.bc_duration = bc_duration

        self.parse_dialogs = parse_dialogs

    def __len__(self):
        return len(self.dialogs)

    def __getitem__(self, idx):
        return self.dialogs[idx]

    def __str__(self):
        return "Switchboard"

    def __call__(self):
        self.logger.info(
            f"Switchboard Config: Pairwise {self.pairwise}, ReBC {self.remove_backchannels}")
        self.filenames = self.read_files()
        self.dialogs = self.read_dialog()
        # self.tokens = self.tokenize()

    def read_file_splits(self):
        self.logger.info(f"data: reading {self.split} data")

        filename = get_abs_path(os.path.join("splits", f"{self.split}.txt"))
        if not os.path.isfile(filename):
            self.logger.error(f"data: no splits {self.split} files")
            self.generate_file_splits()

        split_filenames = {}
        with open(filename) as f:
            line = f.readline()
            while line:
                values = line.split("\t")
                conv_id = values[0].strip()

                prefix_dict = os.path.join(
                    TRANSCRIPT_DIRECTORIES[0], conv_id[:2], conv_id)

                split_filenames[conv_id] = []
                filenames = os.listdir(prefix_dict)
                filenames.sort()
                if self.dev_mode and self.parse_dialogs is None:
                    filenames = filenames[:SMALL_SET_SIZE]

                for file in filenames:
                    split_filenames[conv_id].append(os.path.join(
                        prefix_dict, file.strip()))

                line = f.readline()

                if self.dev_mode and self.parse_dialogs is None:
                    if len(split_filenames) == SMALL_SET_SIZE:
                        break

        return split_filenames

    def read_files(self):
        if self.split == 'all' or self.parse_dialogs is not None:
            self.split = 'train'
            train_files = self.read_file_splits()
            self.split = 'test'
            test_files = self.read_file_splits()
            self.split = 'val'
            val_files = self.read_file_splits()

            self.split = 'all'

            if self.parse_dialogs is not None:
                print(f"Parse Dialogs from {self.parse_dialogs}")
                files = {}
                for file in self.parse_dialogs:
                    new_dict = {**train_files, **val_files, **test_files}
                    if file in new_dict:
                        files[file] = new_dict[file]
                    else:
                        print(
                            f"Parse Dialogs could not find {file} with example key {new_dict.keys()}")

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
                            files[lines[0]] = [
                                lines[1], lines[2], lines[3], lines[4]]
                            lines = []
                    lines.append(line.strip())
                line_idx += 1
                line = f.readline()

        train, test = train_test_split(
            list(files.keys()), test_size=.2, shuffle=False)
        val, test = train_test_split(test, test_size=.3, shuffle=False)

        train_filename = get_abs_path("splits/train.txt")
        val_filename = get_abs_path("splits/val.txt")
        test_filename = get_abs_path("splits/test.txt")

        with open(train_filename, "w") as f:
            for key in train:
                value = files[key]
                f.write(
                    f"{key}\t{value[0]}\t{value[1]}\t{value[2]}\t{value[3]}\n")

        with open(test_filename, "w") as f:
            for key in test:
                value = files[key]
                f.write(
                    f"{key}\t{value[0]}\t{value[1]}\t{value[2]}\t{value[3]}\n")

        with open(val_filename, "w") as f:
            for key in val:
                value = files[key]
                f.write(
                    f"{key}\t{value[0]}\t{value[1]}\t{value[2]}\t{value[3]}\n")

    """
    Main function for reading the dialogues from the switchboard dataset
    Built to handle the pairwise and non-pairwise case

    Pairwise:
        - Extracts the dialgues from the switchboard dataset as a tuple (speakerA, speakerB) where each speaker is a list of utterances 
        - Joins the utterances of the same speaker that are seperated by less than 0.5s
        - Iterate through turns and remove backchannels and overlaps producing new dictionary for backchannels and overlaps
        - Combine consecutive turns of the same speaker
        - Generate dictionary for the dialogue that has three keys (dialog, backchannel, overlap) each containing a dictionary of the speaker and their utterances
    Serialised:
        - Extracts the dialogues from the switchboard dataset as a list of utterances
        - Combines the utterances of the same speaker that are seperated by less than 0.5s
        - Iterate through turns and remove backchannels and overlaps
        - Combine consecutive turns of the same speaker
    """

    def read_dialog(self):
        self.logger.info(f"data ({self.split}): loading switchboard data")

        dialogs = []
        for key in tqdm.tqdm(self.filenames):
            # vad = extract_speaker_timings(dialog)
            if self.pairwise:
                dialog = pairwise_extract_dialog(self.filenames[key])

                dialog_ord = extract_dialog(self.filenames[key])
                dialog_ord = combine_dialogue_without_timings(dialog_ord)
                dialog_ord, backchannels = pairwise_remove_backchannels(
                    dialog_ord, self.pre_silence, self.post_silence, self.bc_duration)
                dialog_ord, overlaps = remove_overlaps(dialog_ord)
                dialog_ord = combine_consecutive_trps(dialog_ord)

                new_dialog = {}
                conv_id = dialog[0][0]['conv_id']
                new_dialog['dialog'] = separate_by_speaker(
                    dialog_ord, dialog, conv_id=conv_id)
                new_dialog['backchannel'] = separate_by_speaker(
                    backchannels, conv_id=conv_id)
                new_dialog['overlap'] = separate_by_speaker(
                    overlaps, conv_id=conv_id)
                dialog = new_dialog
            else:
                dialog = extract_dialog(self.filenames[key])
                dialog = combine_dialogue_without_timings(dialog)

                if self.remove_backchannels:
                    dialog = remove_backchannels(
                        dialog, self.pre_silence, self.post_silence, self.bc_duration)

                dialog, _ = remove_overlaps(dialog)
                dialog = combine_consecutive_trps(dialog)
                for x in dialog:
                    x['key'] = key

            dialogs.append(dialog)

        # print(dialogs[0])
        return dialogs

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
    sd = SwitchboardDataset(remove_backchannels=True, pairwise=True)
    sd()
    sd.save_dialogs(get_abs_path(os.path.join("splits", "data")))
