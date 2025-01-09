import os
import re
import gc
import random

from torch.utils.data import Dataset
from datasets import Dataset as HFDataset

from sklearn.model_selection import train_test_split
from data.fisher.utils import (
    extract_dialog,
    remove_overlaps,
    combine_dialogue_without_timings,
    combine_consecutive_trps,
    remove_backchannels,
    pairwise_remove_backchannels,
    separate_by_speaker,
    add_missing_timings,
    regexp,
)
from data.utils import get_logger


def get_abs_path(filepath):
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), filepath)


def match_regex(word, target):
    return regexp(word) == regexp(target)


TRANSCRIPT_DEST = get_abs_path("fisher/data/trans")
WORD_TRANSCRIPT_DEST = get_abs_path("fisher/data/words")
CLEANED_WORD_TRANSCRIPT_DEST = get_abs_path("fisher/data/cleaned_words")
CLEANED_WORD_INDEX = get_abs_path("fisher/data/cleaned_words/index")

WORD_TIMINGS_SOURCE = get_abs_path("fisher/ctm")

SMALL_SET_SIZE = 10

logger = get_logger(__name__)


def file_generator(
    parent_dir,
    out_alt_dir=[WORD_TRANSCRIPT_DEST, TRANSCRIPT_DEST, CLEANED_WORD_TRANSCRIPT_DEST],
):
    for dir in os.listdir(parent_dir):
        for file in os.listdir(os.path.join(parent_dir, dir)):
            yield [os.path.join(root, dir, file) for root in out_alt_dir]


def strip_punctuation(string):
    return list(filter(lambda x: len(x) > 0, [re.sub(r"[()]", "", s) for s in string]))


class FisherDataset(Dataset):
    def __init__(
        self,
        split="train",
        remove_backchannels=False,
        pairwise=True,
        pre_silence=1,
        post_silence=1,
        bc_duration=1,
        trp_separated_by=0.5,
        parse_dialogs=None,
        fix_word_timings=False,
        split_word_timings=False,
        set_file_splits=False,
        dev_mode=False,
        keep_in_memory=True,
        start_from=None,
    ):
        self.split = split

        self.remove_backchannels = remove_backchannels
        self.pre_silence = pre_silence
        self.post_silence = post_silence
        self.bc_duration = bc_duration

        self.pairwise = pairwise

        self.dev_mode = dev_mode
        self.parse_dialogs = parse_dialogs

        self.fix_word_timings = fix_word_timings
        self.split_word_timings = split_word_timings

        self.start_from = start_from
        self.set_file_splits = set_file_splits

        self.trp_separated_by = trp_separated_by

        self.dev_mode = dev_mode

        self.keep_in_memory = keep_in_memory
        self.data_dir = get_abs_path(os.path.join("fisher", ".cache"))

    def __len__(self):
        if self.dialogs is None:
            return 0

        return len(self.dialogs)

    def __getitem__(self, idx):
        if self.dialogs is None:
            raise ValueError("No dialogs found")

        if self.keep_in_memory:
            return self.dialogs[idx]

        return self.dialogs[idx]

    def __call__(self):
        self.filenames = self.read_files()
        self.dialogs = self.read_dialog()
        gc.collect()

        return self

    def __str__(self):
        return "Fisher"

    def generate_save_path(self):
        remove_backchannels = "remove_backchannels" if self.remove_backchannels else ""
        pairwise = "pairwise" if self.pairwise else ""
        pre_silence = f"pre_silence_{self.pre_silence}" if self.pre_silence != 1 else ""
        post_silence = (
            f"post_silence_{self.post_silence}" if self.post_silence != 1 else ""
        )
        bc_duration = f"bc_duration_{self.bc_duration}" if self.bc_duration != 1 else ""
        trp_separated_by = (
            f"trp_separated_by_{self.trp_separated_by}"
            if self.trp_separated_by != 0.5
            else ""
        )

        return f"{self.__str__()}_{self.split}_{remove_backchannels}_{pairwise}_{pre_silence}_{post_silence}_{bc_duration}_{trp_separated_by}"

    def get_batch(self, start=0, batch_size=4):
        end = min(start + batch_size, len(self))

        if self.dialogs is None:
            raise ValueError("No dialogs found")

        for i in range(start, end):
            yield self.dialogs[i]
        # return self.dialogs[start:end]

    def check_fix_word_timings(self):
        if not os.path.exists(CLEANED_WORD_TRANSCRIPT_DEST):
            logger.error(
                f"No cleaned word timings found at {CLEANED_WORD_TRANSCRIPT_DEST}"
            )
            return False

        if not os.path.exists(CLEANED_WORD_INDEX):
            logger.error(f"No cleaned word index found at {CLEANED_WORD_INDEX}")
            return False

        # Verify that all files in the index exist (some files in self.filenames may not exist)
        filenames = set()
        with open(CLEANED_WORD_INDEX) as f:
            line = f.readline()
            while line:
                filenames.add(line.strip())
                line = f.readline()

        for _, cleaned_word_filepath in self.filenames.values():
            if (
                not os.path.exists(cleaned_word_filepath)
                and cleaned_word_filepath in filenames
            ):
                logger.error(f"Missing cleaned word timings at {cleaned_word_filepath}")
                return False

        return True

    def check_split_word_timings(self):
        if not os.path.exists(WORD_TRANSCRIPT_DEST):
            return False

        for _, cleaned_word_filepath in self.filenames.values():
            if not os.path.exists(cleaned_word_filepath):
                return False

        return True

    def read_dialog(self) -> HFDataset | list[dict]:
        logger.info(f"({self.split}): loading data")
        NUM_WORKERS = min(24, os.cpu_count() - 1) if not self.dev_mode else 1

        if self.split_word_timings or not self.check_split_word_timings():
            self._split_word_timings()

        if self.fix_word_timings or not self.check_fix_word_timings():
            logger.error(
                f"No word timings found at {get_abs_path('fisher/data/cleaned_words')}"
            )
            self._fix_word_timings()

        if self.keep_in_memory:
            return list(self._read_dialog())

        ds = HFDataset.from_generator(
            self._read_dialog,
            num_proc=NUM_WORKERS,
            gen_kwargs={
                "filenames": self.filenames,
                "_misc": random.random(),  # [random.random() for _ in range(len(self.filenames))],
            },
            keep_in_memory=True
        )

        ds.save_to_disk(
            os.path.join(
                self.data_dir,
                f"{self.generate_save_path()}.hf",
            ),
            num_shards=NUM_WORKERS,
        )

        return ds

    def _read_dialog(self, filenames=[], _misc=[]):
        logger.info(f"Processing {len(filenames)} files with _misc={_misc}")
        filenames = [(x, y[0], y[1]) for x, y in filenames.items()]
        for _, (conv_id, filename, cleaned_word_filepath) in enumerate(filenames):
            item = self._process_dialog(conv_id, filename, cleaned_word_filepath)
            if item is not None:
                yield item

    def _process_dialog(
        self,
        conv_id: str,
        filename: str,
        cleaned_word_filepath: str,
    ):
        if self.pairwise:
            dialog = extract_dialog((filename, cleaned_word_filepath))
            dialog = combine_dialogue_without_timings(
                dialog, separated_by=self.trp_separated_by
            )
            dialog, backchannels = pairwise_remove_backchannels(
                dialog, self.pre_silence, self.post_silence, self.bc_duration
            )
            dialog, overlaps = remove_overlaps(dialog)
            dialog, backchannels = combine_consecutive_trps(
                dialog, backchannels, overlaps
            )

            new_dialog = {}
            new_dialog["dialog"] = separate_by_speaker(dialog, conv_id=conv_id)
            new_dialog["backchannel"] = separate_by_speaker(
                backchannels, conv_id=conv_id
            )
            new_dialog["overlap"] = separate_by_speaker(overlaps, conv_id=conv_id)
            dialog = new_dialog

        else:
            dialog = extract_dialog(filename, with_word_feats=self.pairwise)
            dialog = combine_dialogue_without_timings(dialog)

            if self.remove_backchannels:
                dialog = remove_backchannels(
                    dialog, self.pre_silence, self.post_silence, self.bc_duration
                )

            dialog = remove_overlaps(dialog)
            dialog = combine_consecutive_trps(dialog)
            dialog[0]["key"] = conv_id

        if (
            len(dialog) == 0
            or len(dialog["dialog"]["speakerA"]) == 0
            or len(dialog["dialog"]["speakerB"]) == 0
        ):
            logger.warning(f"Skipping {conv_id} due to empty")
            return None

        return dialog

    def read_file_splits(self):
        filename = get_abs_path(os.path.join("splits", f"{self.split}.txt"))

        logger.info(f"Reading {self.split} data from {filename}")
        if not os.path.exists(filename) or self.set_file_splits:
            logger.error(
                f"No split for {self.split} files. exists={os.path.exists(filename)} set_file_splits={self.set_file_splits}  Generating..."
            )
            self.generate_file_splits()

        split_filenames = {}
        with open(filename) as f:
            line = f.readline()
            while line:
                file = line.strip()

                # Get filename with respect to relative dir and stripped of ext
                conv_id = file.split("/")[-1][:-4]

                # Just in case we also require word timings
                split_filenames[conv_id] = [
                    file,
                    get_abs_path(
                        os.path.join(
                            "fisher",
                            "data",
                            "cleaned_words",
                            conv_id[6:9],
                            f"{conv_id}.txt",
                        )
                    ),
                ]
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
                logger.info(f"Parse dialogs from {self.parse_dialogs}")
                files = {}
                for file in self.parse_dialogs:
                    new_dict = {**train_files, **val_files, **test_files}
                    if file in new_dict:
                        files[file] = new_dict[file]
                    else:
                        logger.info(
                            f"Parse dialogs could not find {file} from {list(new_dict.items())[0]}..."
                        )

                return files

            return {**train_files, **val_files, **test_files}

        return self.read_file_splits()

    def _split_word_timings(self):
        """
        Splits the word timings into separate files for each conversation
        """
        logger.info("Splitting word timings")
        if not os.path.exists(WORD_TIMINGS_SOURCE):
            logger.error(f"No word timings found at {WORD_TIMINGS_SOURCE}")
            return

        curr_file_id = None
        curr_file = None

        with open(WORD_TIMINGS_SOURCE) as f:
            line = f.readline()
            while line:
                id, speaker, start, duration, word = line.strip().split()
                dir = os.path.join(WORD_TRANSCRIPT_DEST, id[-5:-2])
                if not os.path.exists(dir):
                    os.makedirs(dir)
                    logger.info(f"Creating directory {dir}")

                if curr_file_id != id:
                    if curr_file is not None:
                        curr_file.close()
                    curr_file = open(os.path.join(dir, f"{id}.txt"), "w")
                    curr_file_id = id

                curr_file.write(
                    f"{id} {speaker} {start} {round(float(start)+float(duration),2)} {word}\n"
                )

                line = f.readline()

        curr_file.close()

    def _fix_word_timings(self):
        logger.info("Fixing word timings")
        if not os.path.exists(WORD_TRANSCRIPT_DEST):
            logger.error(
                f"No word transcript directory found at {WORD_TRANSCRIPT_DEST}"
            )
            return
        if not os.path.exists(TRANSCRIPT_DEST):
            logger.error(f"No word timings found at {TRANSCRIPT_DEST}")
            return

        if not os.path.exists(CLEANED_WORD_TRANSCRIPT_DEST):
            os.mkdir(CLEANED_WORD_TRANSCRIPT_DEST)

        with open(CLEANED_WORD_INDEX, "w") as f:
            f.write("")

        for word_file, trans_file, out_file in file_generator(WORD_TRANSCRIPT_DEST):
            with open(word_file) as f:
                lines = f.readlines()

            # (id, speaker, start, end, word)
            word_feats = [line.strip().split() for line in lines if len(line) > 0]

            with open(trans_file) as f:
                lines = f.readlines()

            # (start, end, speaker, words)
            trans_feat = [
                strip_punctuation(line.strip().split())
                for line in lines
                if line[0] != "#" and len(line.strip()) > 0
            ]

            conv_id = lines[0].split()[-1].split(".")[0]
            if self.start_from is not None and self.start_from not in conv_id:
                continue
            elif self.start_from is not None:
                self.start_from = None

            word_featsA = [word for word in word_feats if word[1] == "A"]
            trans_featA = [tran for tran in trans_feat if tran[2] == "A:"]

            word_featsB = [word for word in word_feats if word[1] == "B"]
            trans_featB = [tran for tran in trans_feat if tran[2] == "B:"]

            word_featsA = self._generate_cleaned_transcript(
                word_featsA, trans_featA, conv_id=conv_id
            )
            word_featsB = self._generate_cleaned_transcript(
                word_featsB, trans_featB, speaker="B", conv_id=conv_id
            )

            if word_featsA is None or word_featsB is None:
                logger.warning("Skipping conv_id", conv_id)
                continue

            if not os.path.exists(os.path.dirname(out_file)):
                os.mkdir(os.path.dirname(out_file))

            with open(out_file, "w") as f:
                for idx, word_feats in enumerate(word_featsA):
                    for word in word_feats:
                        f.write(
                            f"{word['speaker']} {idx} {word['start']} {word['end']} {word['word']}\n"
                        )

                for idx, word_feats in enumerate(word_featsB):
                    for word in word_feats:
                        f.write(
                            f"{word['speaker']} {idx} {word['start']} {word['end']} {word['word']}\n"
                        )

            with open(CLEANED_WORD_INDEX, "a") as f:
                f.write(f"{out_file}\n")

    def _generate_cleaned_transcript(
        self, word_feats, trans_feat, speaker="A", conv_id=None
    ):
        """
        Takes as input the word timings and the transcript of a single speaker
        and generates a cleaned transcript

        Initialise the "theorised" word indexes that should be filled
        For each word in the transcript find the appropriate word

        TODO: Clean of punctuation in cases where there is, within the transcript
            "(( uh-huh ))" -> "uh-huh"
        """
        if len(word_feats) == 0:
            return None

        def match_word(word, target):
            return word == target or word == "<unk>" or match_regex(word, target)

        calc_word_feats = [
            [
                {
                    "start": None if idx > 0 else turn[0],
                    "end": None if idx < len(turn[3:]) - 1 else turn[1],
                    "speaker": speaker,
                    "word": word,
                }
                for idx, word in enumerate(turn[3:])
            ]
            for turn in trans_feat
        ]

        current_turn_index = 0
        current_word_index = 0
        total_word_idx = 0
        word = word_feats[total_word_idx]
        while total_word_idx < len(word_feats) and current_turn_index < len(trans_feat):
            current_start, current_end, _, *words = trans_feat[current_turn_index]
            current_start, current_end = float(current_start), float(current_end)
            word = word_feats[total_word_idx]

            _, _, word_start, word_end, word = word
            word_start, word_end = float(word_start), float(word_end)

            if word_start < current_start - 0.1:
                # Keep looking for the word that starts in the turn
                total_word_idx += 1
                continue

            if word_start >= current_end - 0.01 or current_word_index >= len(words):
                current_turn_index += 1
                current_word_index = 0

                while word_start > current_start and total_word_idx > 0:
                    total_word_idx -= 1
                    word_start = float(word_feats[total_word_idx][2])

                continue

            if word_start >= current_start - 0.01 and word_end <= current_end + 0.01:
                if match_word(word, words[current_word_index]):
                    calc_word_feats[current_turn_index][current_word_index] = {
                        "start": word_start,
                        "end": word_end,
                        "speaker": speaker,
                        "word": words[current_word_index],
                    }
                else:
                    total_word_idx += 1
                    continue

                current_word_index += 1
            elif (
                current_word_index == 0 or current_word_index == len(words) - 1
            ) and match_word(word, words[current_word_index]):
                calc_word_feats[current_turn_index][current_word_index] = {
                    "start": word_start,
                    "end": word_end,
                    "speaker": speaker,
                    "word": words[current_word_index],
                }
                current_word_index += 1

            total_word_idx += 1

        return add_missing_timings(calc_word_feats, speaker)

    def pprint(self, calc_word_feats):
        for turn in calc_word_feats:
            out = ""
            for word in turn:
                out += f"({word['word']} {word['start']} {word['end']}) "

            logger.info(f"Turn: {out}")

    def generate_file_splits(self):
        if not os.path.exists(get_abs_path("splits")):
            os.mkdir(get_abs_path("splits"))

        splits_dir = get_abs_path("splits")
        files = []
        for dir in os.listdir(TRANSCRIPT_DEST):
            for file in os.listdir(os.path.join(TRANSCRIPT_DEST, dir)):
                files.append(os.path.join(TRANSCRIPT_DEST, dir, file))

        train, test = train_test_split(files, test_size=0.2, shuffle=False)
        val, test = train_test_split(test, test_size=0.5, shuffle=False)

        train_filename = get_abs_path("splits/train.txt")
        test_filename = get_abs_path("splits/test.txt")
        val_filename = get_abs_path("splits/val.txt")

        with open(train_filename, "w") as f:
            for key in train:
                f.write(f"{key}\n")

        with open(test_filename, "w") as f:
            for key in test:
                f.write(f"{key}\n")

        with open(val_filename, "w") as f:
            for key in val:
                f.write(f"{key}\n")


if __name__ == "__main__":
    ds = FisherDataset(fix_word_timings=False, start_from=None, set_file_splits=True)
    ds()
