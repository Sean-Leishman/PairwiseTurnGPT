import torch
import logging
import os
import copy
import tqdm

from torch.utils.data import DataLoader, Dataset, ConcatDataset

from data.switchboard import SwitchboardDataset
from pairwisegpt.tokenizer import SpokenDialogTokenizer

from data.utils import pp_pair_dialogs, pp_single_dialogs

from enum import IntEnum
import numpy as np

import pickle

def get_abs_path(filepath):
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), filepath)


DATASETS = [SwitchboardDataset]
CACHE_PATH = get_abs_path(".cache")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TurnType(IntEnum):
    NONE = 0
    NORMAL = 1
    YIELD = 2  # TURN SHIFT CAUSED BY INTERRUPTION / OVERLAP (CLOSE ENOUGH)
    BACKCHANNEL = 3
    INTERRUPT = 4
    OVERLAP = 5

    NON_YIELD=6 # FOR METRIC GENERATION

    ALL = 7

    NON_OVERLAP = 8


class PairwiseGenerationDM(Dataset):
    """
    Implementation for use with Pairwise TurnGPT. Can also load with TurnGPT
    style turn stacking for both pairwise and serialised versions.

    Attributes
    -----------
    tokenizer: Transformers.TokenizerFast
        used to tokenize text when loading data
    device: str
        defines device which stores tensors
    savepath: str
        defines folder which should contain saved datasets
    overwrite: bool
        fetch and tokenize all datasets
    load_from_cache_file: bool
    max_length: int
        maximum number of tokens within a sequence
    keep_length: int
        only keep sequences with this number of tokens
    overlap_length: int
        number of tokens that overlap between consecutive sequences if the total sequence exceeds maximum length
    yield_overlap_thresh: int
        defines a yield turn end if overlap is within X seconds of the turn end
    dev_mode: bool
        if set to true then loads a subset of files to test functions
    split_utt: bool
        if set to false do not split utterances into batches of length max_length
    savedata: bool
        if set to true then data is saved to a cache file
    no_emp_tokens: bool
        if set to true then the input streams are not time aligned
    remove_start_tokens: bool
        if set to false then each turn is surrounded by start then end of turn tokens
    remove_overlaps bool
        if set to true remove overlaps and backchannels like in TurnGPT
    store_raw: bool
        if set to true store turn shift data structure
    parse_dialogs: List[str]
        list of keys to query and parse. used for debugging
    combine_speaker: bool
        if set to true then combine input streams for TurnGPT
    load_metrics_from_all: bool
        if set to true then do not use computed metrics, get from previous runs based on the entire dataset
    normalize_time: bool
        if set to true then normalize projection labels for regression task
    categorize_projection: bool
        if set to true then assign bins to each projection
    category_bins: int
        if categorize_projection is set, divide statistic into X bins
    include_bc_token: bool
        replace backchannels with a special <bc> token
    include_end_bc_token: bool
        use a <ebc> token rather than <eot> token to end a backchannel turn
    include_overlap_token: bool
        use a <eint> token rather than <eot> token to end a overlap turn
    include_yield_turn: bool
        use a <yield> token to end a yield turn rather than an <eot> token
    individual_speaker_tokens: bool
        if set then use a special token rather than 1 and 2
    filter_bc_overlap_token: bool
        if set to true then replace <ebc> and <eint> tokens with <emp> token
    evaluate_on_full: bool
        ensure that if serialising, data is consistent with the fully time aligned set
    use_speaker_token_in_embedding: bool
        if set then the speaker_ids are put into a special tensor
    basic_mode: bool
        if set then the two speaker streams are serialised
    basic_with_special: bool
        if set then the two speaker streams are serialised with backchannels and overlaps included (unused)
    """

    def __init__(self, split="train",
                 tokenizer=None,
                 device="cuda:0",
                 savepath=None,
                 overwrite=False,
                 load_from_cache_file=False,
                 max_length=256,
                 keep_length=64,
                 overlap_length=10,
                 yield_overlap_thresh=4,
                 basic_mode=False,
                 basic_with_special=False,
                 dev_mode=False,
                 split_utt=True,
                 savedata=True,
                 no_emp_tokens=False,
                 remove_start_tokens=False,
                 remove_overlaps=False,
                 remove_backchannels=False,
                 store_raw=False,
                 parse_dialogs=None,
                 combine_speaker=False,
                 load_metrics_from_all=False,
                 normalize_time=False,
                 categorize_projection=False,
                 category_bins=5,
                 include_bc_token=False,
                 include_end_bc_token=False,
                 include_sil_token=False,
                 include_overlap_token=False,
                 include_yield_token=False,
                 individual_speaker_tokens=False,
                 interruption_thresh=0.2,
                 keep_partial_overlap=True,
                 filter_bc_overlap_token=False,
                 evaluate_on_full=False,
                 use_speaker_token_in_embedding=False,
                 datasets=["switchboard"],
                 **kwargs,
                 ):
        self.logger = logger
        self.device = device

        self.tokenizer = tokenizer
        self.datasets = []

        self.data = []

        self.split = split
        if self.split not in ["train", "val", "test"]:
            self.split = "all"

        self.basic_mode = basic_mode
        self.basic_with_special = basic_with_special

        if savepath is None:
            dirname = self.tokenizer.__str__(
            )[:self.tokenizer.__str__().index("(")]
            savepath = os.path.join(CACHE_PATH, dirname)
        self.savepath = savepath
        self.savedata = savedata

        self.overwrite = overwrite
        self.load_from_cache_file = load_from_cache_file

        self.max_length = max_length
        self.keep_length = keep_length
        self.overlap_length = overlap_length
        self.yield_overlap_thresh = yield_overlap_thresh

        self.dev_mode = dev_mode
        self.split_utt = split_utt

        self.no_emp_tokens = no_emp_tokens
        self.remove_overlaps = remove_overlaps
        self.remove_backchannels = remove_backchannels

        self.remove_start_tokens = remove_start_tokens

        self.use_speaker_token_in_embedding = use_speaker_token_in_embedding

        self.speakerA_token = self.tokenizer.convert_tokens_to_ids('<speakerA>')
        self.speakerB_token = self.tokenizer.convert_tokens_to_ids('<speakerB>')
        if not self.use_speaker_token_in_embedding or self.speakerA_token == self.tokenizer.pad_token_id:
            self.speakerA_token = 1
            self.speakerB_token = 2

        self.ts = []
        self.store_raw = True

        self.parse_dialogs = parse_dialogs

        # Can only be combining streams if in basic mode
        self.combine_speaker = combine_speaker
        self.basic_mode = True if self.combine_speaker else basic_mode

        self.load_metrics_from_all = load_metrics_from_all
        self.normalize_time = normalize_time
        self.categorize_projection = categorize_projection if not self.normalize_time else False

        # Either tokenize with emp tokens -> with/without overlap
        # Or without emp tokens
        # Without emp tokens overrules the other options
        self.tokenize = self.tokenize_without_overlap if self.basic_mode else self.tokenize_with_overlap
        self.tokenize = self.tokenize_without_emp if self.no_emp_tokens else self.tokenize

        self.include_overlap_token = include_overlap_token
        self.include_yield_token = include_yield_token
        self.filter_bc_overlap_token = filter_bc_overlap_token
        self.evaluate_on_full = evaluate_on_full

        self.keep_partial_overlap = keep_partial_overlap

        # Always have <ebc> if using <bc> for speed of implementation
        self.include_bc_token = include_bc_token
        self.include_end_bc_token = include_end_bc_token if not self.include_bc_token else True
        if self.include_bc_token and not self.include_end_bc_token:
            self.logger.warning("Include <endbc> should be set if <bc> is set. Setting true")
            self.include_end_bc_token = True

        self.include_sil_token = include_sil_token

        self.individual_ts = individual_speaker_tokens
        if self.individual_ts and not self.remove_start_tokens:
            self.logger.warning("If using individual <ts> for serialised requires no start tokens")
            self.remove_start_tokens = True

        self.store_raw = store_raw
        self.interruption_thresh = interruption_thresh

        self.bins = category_bins

        if self.basic_mode:
            self.include_overlap_token = False
            self.include_end_bc_token = False
            self.remove_overlaps = True
            self.remove_backchannels = True
            self.remove_start_tokens = True
        elif self.basic_with_special:
            self.include_overlap_token = False
            self.include_end_bc_token = False
            self.remove_overlaps = False
            self.remove_backchannels = False
            self.remove_start_tokens = True

        self.logger.info(
            f"Load Pairwise GenerationDM with: Overwrite {self.overwrite}, Basic Mode {self.basic_mode}, Split {self.split}, Datasets {datasets} with filename {self.get_save_load_path()}")

        # Loads each dataset in turn and appends to self.datasets
        for ds in datasets:
            if ds == "switchboard":
                if not self.basic_mode:
                    self.datasets.append(SwitchboardDataset(
                        split=self.split, pairwise=True, dev_mode=self.dev_mode, parse_dialogs=self.parse_dialogs))
                else:
                    self.datasets.append(SwitchboardDataset(
                        split=self.split, pairwise=True, dev_mode=self.dev_mode, parse_dialogs=self.parse_dialogs))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def __str__(self):
        remove_overlaps = "RemoveOverlaps-" if self.remove_overlaps else ""
        remove_backchannels = "RemoveBackchannels" if self.remove_backchannels else ""
        serialise = "Serialise-" if self.basic_mode else ""
        filter_bc_overlap = "FilterBcEint-" if self.filter_bc_overlap_token else ""
        return f"PairwiseGenerationDM-{self.split}-{serialise}{filter_bc_overlap}{remove_overlaps}{remove_backchannels}"

    def collate_fn(self, batch):
        def collate_fn_channel(batch):
            ret = self.tokenizer.pad(
                {"input_ids": [b["input_ids"][: self.max_length]
                               for b in batch]},
                padding='max_length', max_length=self.max_length)

            ret["token_type_ids"] = self.tokenizer.pad(
                {"input_ids": [b["token_type_ids"][: self.max_length]
                               for b in batch]},
                padding='max_length', max_length=self.max_length)['input_ids']

            ret["other_token_type_ids"] = self.tokenizer.pad(
                {"input_ids": [b["other_token_type_ids"][: self.max_length]
                               for b in batch]},
                padding='max_length', max_length=self.max_length)['input_ids']

            ret["turn_overlap"] = self.tokenizer.pad(
                {"input_ids": [b["turn_overlap"][: self.max_length]
                               for b in batch]},
                padding='max_length', max_length=self.max_length)['input_ids']

            ret["timings"] = torch.stack([torch.nn.functional.pad(
                b['timings'], (0, 0, 0, 256 - b['timings'].size(0))) for b in batch])

            ret['time_until_ts'] = self.tokenizer.pad(
                {"input_ids": [b["time_until_ts"][:self.max_length] for b in batch]},
                padding='max_length', max_length=self.max_length)['input_ids']

            if 'speaker_ids' in batch[0].keys():
                ret['speaker_ids'] = self.tokenizer.pad(
                    {"input_ids": [b["speaker_ids"][:self.max_length] for b in batch]},
                    padding = 'max_length', max_length = self.max_length)['input_ids']
            else:
                ret['speaker_ids'] = torch.ones_like(ret['input_ids'])

            for k, v in ret.items():
                ret[k] = v.clone().detach()

            ret['conv_id'] = [b['conv_id'] for b in batch]
            return ret

        if self.combine_speaker:
            return collate_fn_channel(batch)

        batchA = [x['speakerA'] for x in batch]
        batchB = [x['speakerB'] for x in batch]

        retA = collate_fn_channel(batchA)
        retB = collate_fn_channel(batchB)

        assert retA['input_ids'].shape == retB['input_ids'].shape
        assert retA['input_ids'].shape[1] == self.max_length

        return {'speakerA': retA, 'speakerB': retB}

    def get_save_load_path(self, dir="", file_type="", ext=""):
        save_load_dir = get_abs_path(self.savepath)
        if dir != "":
            save_load_dir = get_abs_path(dir)

        ext = "." + ext if len(ext) > 0 and ext[0] != "." else ext

        if not os.path.exists(save_load_dir):
            os.mkdir(save_load_dir)

        basic = "Basic" if self.basic_mode else ""
        combine = "Combine" if self.combine_speaker else ""
        split_utt = "NotSplitLength" if not self.split_utt else ""
        add_emp_tokens = "NoEmpTokens" if self.no_emp_tokens else ""
        remove_overlaps = "RemoveOverlaps" if self.remove_overlaps else ""
        remove_backchannels = "RemoveBackchannels" if self.remove_backchannels else ""
        remove_start_tokens = "RemoveStartTokens" if self.remove_start_tokens else ""
        yield_tokens = "IncludeYield" if self.include_yield_token else ""
        bc_tokens = "IncludeBc" if self.include_bc_token else ""
        end_bc_tokens = "IncludeEndBc" if self.include_end_bc_token else ""
        end_int_tokens = "IncludeEndInt" if self.include_overlap_token else ""
        use_speaker_token_embedding= "UseSpeakerTokenEmbedding" if self.use_speaker_token_in_embedding else ""
        individual_ts = "IdvTS" if self.individual_ts else ""
        filter_bc_overlap = "FilterBcOverlap" if self.filter_bc_overlap_token else ""

        if ext != "":
            # Saving Metrics/Bins so remove some detail
            bins = ""
            if file_type == "category_bins":
                bins = f"BINS{self.bins}"
            return os.path.join(save_load_dir, "Pairwise" + basic + "".join(
                str(x) for x in self.datasets) + file_type + bins + ext)

        return os.path.join(save_load_dir,
                            "Pairwise" + basic + combine + remove_overlaps + remove_backchannels + remove_start_tokens + split_utt +
                            add_emp_tokens + yield_tokens + bc_tokens + end_bc_tokens + end_int_tokens + use_speaker_token_embedding + individual_ts +
                            filter_bc_overlap +
                            "".join(str(x) for x in self.datasets) + self.split.capitalize() + file_type + ext)

    """
    Main function for generating the data
    """
    def prepare_data(self):
        if self.load_from_cache_file or not self.overwrite:
            self.logger.info(f"data: loading {self.get_save_load_path()} from cache")
            if self.setup():
                self.log_info()
                return
            self.logger.info("data: setup failed so overwrite")

        # Load metrics for standardisation and binning from files if
        # in test split
        if self.split != "train":
            # Experimental Work
            if self.normalize_time:
                file_path = self.get_save_load_path('metrics', ext='pkl')
                if not os.path.exists(file_path):
                    self.logger.info(
                        f"data: failed to find metrics at {file_path}")
                    return
                with open(self.get_save_load_path('metrics', ext='pkl'), 'rb') as f:
                    self.metrics = pickle.load(f)
            if self.categorize_projection:
                file_path = self.get_save_load_path('category_bins', ext='pkl')
                if not os.path.exists(file_path):
                    self.logger.info(
                        f"data: failed to find metrics at {file_path}")
                    return

                with open(file_path, 'rb') as f:
                    self.category_bins = pickle.load(f)

        # Initialise datasets individually
        for ds in self.datasets:
            ds()

        self.logger.debug(f"data: remove overlap/backchannel: {self.remove_overlaps}/{self.remove_backchannels}")

        # Combine into one set 
        self.dataset = ConcatDataset(self.datasets)
        self.data, metrics = self.tokenize()
        self.category_bins = self.get_categories(
            columns=['time_until_ts', 'time_until_other_ts'], num_bins=self.bins)

        if self.split == "train":
            self.metrics = metrics

        # Experimental Work
        if self.normalize_time:
            self.normalize(self.metrics)

        # Experimental Work
        if self.categorize_projection:
            self.categorize()

        # For the serialised version to ensure consistency with how we generate for fully time aligned
        if self.combine_speaker:
            self.data = self.combine_speaker_channels()
        elif self.basic_mode and self.evaluate_on_full:
            # We need to fix the data so that <eot>/<emp> pairs do not exist 
            # to be consistent with the combined version 
            self.data = self.remove_eot_emp_pair()

        if self.split_utt:
            self.data = self.split_to_length(pairwise=not self.combine_speaker)

        if self.filter_bc_overlap_token:
            self.data = self.filter_special_tokens()

        self.log_info()
        self.save_to_disk()

    def setup(self):
        filename = self.get_save_load_path()
        try:
            saved_ds = torch.load(filename)
            self.dataset = saved_ds.dataset
            self.data = saved_ds.data

            if self.store_raw:
                with open(self.get_save_load_path("ts", ext=".pkl"), 'rb') as f:
                    self.ts = pickle.load(f)

            return True

        except FileNotFoundError:
            self.logger.error(f"File not found: {filename}")

        return False

    def save_to_disk(self):
        if self.split == "train":
            self.logger.info(f"data: save metrics")
            with open(self.get_save_load_path('metrics', ext='pkl'), 'wb') as f:
                pickle.dump(self.metrics, f)

            with open(self.get_save_load_path('category_bins', ext='pkl'), 'wb') as f:
                pickle.dump(self.category_bins, f)

        if not self.savedata:
            self.logger.info(
                f"data {self.split}: not saving transcript with savedata: {self.savedata}")
            return

        self.logger.info(
            f"data {self.split}: saving combined transcript at {self.get_save_load_path()}")
        torch.save(self, self.get_save_load_path())

        if self.store_raw:
            self.logger.info(f"data: saving <ts> data")
            with open(self.get_save_load_path('ts', ext='pkl'), 'wb') as f:
                pickle.dump(self.ts, f)


    def match_tokens_words(self, tokens, dialog, sentence):
        """
        Builds a list of dict (start_idx, end_idx, id, tokens) from tokens:
            output from tokenizer; dialog: features from dataset that contain
            word timings; sentence: string of the input

        This is done to split words within sentence into their respective and assign
        timings to each of these subtokens
        """
        dialog_word_idx = 0
        dialog_idx = 0

        decoded = ""
        new_tokens = []
        token_timings = []

        utt_outputs = []
        outputs = []

        if len(dialog) == 0:
            return outputs

        conv_id = dialog[0]['conv_id']

        for token_id, offset in zip(tokens['input_ids'][0], tokens['offset_mapping'][0]):
            word = sentence[offset[0]:offset[1]].strip()
            decoded += word
            new_tokens.append(token_id.item())

            if dialog_word_idx >= len(dialog[dialog_idx]['wfeats']):
                dialog_idx += 1
                dialog_word_idx = 0
                outputs.append(utt_outputs)
                utt_outputs = []

            curr_wfeat = dialog[dialog_idx]['wfeats'][dialog_word_idx]
            if decoded == curr_wfeat['word']:
                curr_wfeat['tokens'] = new_tokens

                # Add timings for start and end of each subtoken
                token_timings = []
                step_length = 1 / len(new_tokens)
                duration = curr_wfeat['end'] - \
                           curr_wfeat['start']
                for idx, token in enumerate(new_tokens):
                    offset_start = step_length * idx * duration
                    offset_end = step_length * (idx + 1) * duration
                    start_time = round(
                        curr_wfeat['start'] + offset_start, 5)
                    end_time = round(
                        curr_wfeat['start'] + offset_end, 5)
                    token_timings.append((start_time, end_time))

                curr_wfeat['token_timings'] = token_timings
                curr_wfeat['start'] = round(
                    curr_wfeat['start'], 3)
                curr_wfeat['end'] = round(
                    curr_wfeat['end'], 3)

                for idx, token in enumerate(new_tokens):
                    output = {}

                    output['conv_id'] = conv_id
                    output['start'] = token_timings[idx][0]
                    output['end'] = token_timings[idx][1]
                    output['tokens'] = token
                    output['word'] = word

                    utt_outputs.append(output)

                new_tokens = []
                decoded = ""
                dialog_word_idx += 1

        outputs.append(utt_outputs)

        assert all(
            all('tokens' in word for word in key['wfeats']) for key in dialog), f"{dialog}"
        assert sum([len(x) for x in outputs]) == len(tokens['input_ids'][0])

        return outputs

    def add_ts_token_as_type(self, output, ts, tokens, speaker='A'):
        """
        Operates on one stream of speaker utterances and adds `token_types` and
        `time_until_ts` values for each token in the stream

        `token_types` is the TurnType for which the token belongs to
        `time_until_ts` is the time until the current speaker's end of final
        word in their current utterance
        """
        word_idx = 0
        curr_ts_idx = 0
        in_ts = False
        in_yield = False

        token_types = []
        time_until_ts = []
        time_until_other_ts = []

        idx = 0
        in_ts_idx = 0
        while idx < len(output['tokens']):
            if curr_ts_idx >= len(ts):
                break

            if output['tokens'][idx] in tokens.values():
                if in_ts:
                    #token_types.append(TurnType.NORMAL)
                    token_types.append(ts[curr_ts_idx]['turn_type'])
                    time_until_ts.append(
                        ts[curr_ts_idx]['time_until_ts'][in_ts_idx])
                else:
                    token_types.append(TurnType.NONE)
                    time_until_ts.append(0)

                idx += 1
                continue

            if not in_ts:
                if ts[curr_ts_idx]['start_idx'] == word_idx:
                    token_types.append(ts[curr_ts_idx]['turn_type'])
                    time_until_ts.append(
                        ts[curr_ts_idx]['time_until_ts'][in_ts_idx])

                    if ts[curr_ts_idx]['start_idx'] == ts[curr_ts_idx]['end_idx']:
                        if idx < len(output['tokens'])-1 and output['tokens'][idx+1] in tokens.values():
                            token_types.append(ts[curr_ts_idx]['turn_type'])
                            time_until_ts.append(
                                ts[curr_ts_idx]['time_until_ts'][in_ts_idx])
                            idx += 1
                        else:
                            pass
                        curr_ts_idx += 1
                    else:
                        in_ts = True
                else:
                    token_types.append(TurnType.NONE)
                    time_until_ts.append(0)
            else:
                if ts[curr_ts_idx]['end_idx'] == word_idx:
                    token_types.append(ts[curr_ts_idx]['turn_type'])
                    time_until_ts.append(
                        ts[curr_ts_idx]['time_until_ts'][in_ts_idx])
                    if idx < len(output['tokens'])-1 and output['tokens'][idx+1] in tokens.values():
                        token_types.append(ts[curr_ts_idx]['turn_type'])
                        time_until_ts.append(
                            ts[curr_ts_idx]['time_until_ts'][in_ts_idx])
                        idx += 1

                    in_ts = False
                    in_ts_idx = 0
                    in_yield = False
                    curr_ts_idx += 1
                else:
                    token_types.append(ts[curr_ts_idx]['turn_type'])
                    time_until_ts.append(
                        ts[curr_ts_idx]['time_until_ts'][in_ts_idx])

            word_idx += 1
            idx += 1

        for i in range(idx, len(output['tokens'])):
            token_types.append(TurnType.NONE)
            time_until_ts.append(0)

        output['token_type_ids'] = [int(x) for x in token_types]
        output['time_until_ts'] = time_until_ts

        # At most should be one item greater than length of tokens
        if len(output['token_type_ids']) > len(output['tokens']):
            output['token_type_ids'] = output['token_type_ids'][:-1]
            output['time_until_ts'] = output['time_until_ts'][:-1]

        assert len(output['token_type_ids']) == len(
            output['tokens']), f"{len(output['tokens'])}, {len(output['token_type_ids'])}"
        assert len(output['time_until_ts']) == len(output['token_type_ids']), f"{len(output['time_until_ts']), len(output['token_type_ids'])}"
        return output, []

    def add_ts_token_dialog(self, output, ts, tokens, speaker='A'):
        """
        Adds turn-shift tokens in appopriate places (<emp>) within the
        transcript and keeps track of where this is not possible for later insertion

        Also we deal with adding `other_token_type_ids` so whether the
        turn-shift has occured after some interjection from the other speaker,
        the entire utterance is labelled with this token

        'turn_overlap' is added as the time between end of current utterance
        and the stat of the other speaker's utterance
        """
        word_idx = 0
        curr_ts_idx = 0
        in_ts = False

        emp_token_idx = tokens['emp']
        eot_token_idx = tokens['eot']
        sot_token_idx = tokens['sot']
        sbc_token_idx = tokens['sbc']
        ebc_token_idx = tokens['ebc']
        sint_token_idx = tokens['sint']
        eint_token_idx = tokens['eint']
        yield_token_idx = tokens['yield']

        unadded_idxs = []
        idx = 0
        in_yield = False

        # Add YIELD/NORMAL token_type_ids
        output['other_token_type_ids'] = [
            TurnType.NORMAL for _ in range(len(output['tokens']))]
        output['turn_overlap'] = [0 for _ in range(len(output['tokens']))]

        # Essentially loops through tokens and deals with adding turn shift
        # tokens that correspond to the next entry in the `ts` structure
        while idx < len(output['tokens']):
            if curr_ts_idx >= len(ts):
                break

            start_token = sot_token_idx
            end_token = eot_token_idx
            if self.include_end_bc_token and ts[curr_ts_idx]['turn_type'] == TurnType.BACKCHANNEL:
                start_token = sbc_token_idx
                end_token = ebc_token_idx
            elif self.include_overlap_token and ts[curr_ts_idx]['turn_type'] == TurnType.OVERLAP:
                start_token = sint_token_idx
                end_token = eint_token_idx
            elif self.include_yield_token and ts[curr_ts_idx]['other_turn_type'][0] == TurnType.YIELD:
                # Might've started as interruption but ends in the same way
                start_token = sot_token_idx
                end_token = yield_token_idx

            if in_yield and ts[curr_ts_idx]['other_turn_type'][1] <= output['timings'][idx][0]:
                output['other_token_type_ids'][idx] = TurnType.YIELD

            if in_ts:
                output['turn_overlap'][idx] = ts[curr_ts_idx]['turn_overlap']

            if output['tokens'][idx] in tokens.values():
                idx += 1
                continue

            if not in_ts:
                if ts[curr_ts_idx]['start_idx'] == word_idx:
                    if idx == 0:
                        if not self.remove_start_tokens:
                            unadded_idxs.append(
                                ('start', idx, speaker, start_token, ts[curr_ts_idx]))
                            word_idx += 1
                            idx += 1
                        in_ts = True
                        in_yield = ts[curr_ts_idx]['other_turn_type'][0] == TurnType.YIELD
                        continue

                    if output['tokens'][idx - 1] == emp_token_idx:
                        if not self.remove_start_tokens:
                            output['tokens'][idx - 1] = start_token
                            output['timings'][idx - 1] = (-1, -1)
                    else:
                        if not self.remove_start_tokens:
                            unadded_idxs.append(
                                ('start', idx, speaker, start_token, ts[curr_ts_idx]))

                    # This is done in case the utterance is one word so need to
                    # reconsider this word again
                    idx -= 1
                    word_idx -= 1
                    in_ts = True
                    in_yield = ts[curr_ts_idx]['other_turn_type'][0] == TurnType.YIELD
            else:
                if idx == len(output['tokens']) - 1:
                    unadded_idxs.append(('end', idx + 1, speaker, end_token, ts[curr_ts_idx]))
                    idx += 1
                    in_yield = False
                    continue

                if ts[curr_ts_idx]['end_idx'] == word_idx:
                    include_bc = self.include_bc_token and ts[curr_ts_idx]['turn_type'] == TurnType.BACKCHANNEL
                    if output['tokens'][idx + 1] == emp_token_idx:
                        output['tokens'][idx + 1] = end_token
                        output['other_token_type_ids'][idx + 1] = ts[curr_ts_idx]['other_turn_type'][0]
                        output['timings'][idx + 1] = (-1, -1)
                    else:
                        unadded_idxs.append(('end', idx + 1, speaker, end_token, ts[curr_ts_idx]))
                    in_ts = False
                    in_yield = False
                    curr_ts_idx += 1

            word_idx += 1
            idx += 1

        assert len(output['tokens']) == len(output['other_token_type_ids'])
        assert 'turn_overlap' in output
        return output, unadded_idxs

    def fix_ts_dialog(self, output, unadded, tokens):
        """
        Adds turn tokens where there is <emp> token to fill. Since both
        streams must be of equal length if a token is added to one stream
        an <emp> token is added in the other
        """
        unadded.sort(key=lambda x: x[1], reverse=True)

        emp_token_id = tokens['emp']
        eint_token_id = tokens['eint']
        offset = 0

        for idx, (type, word_idx, speaker, token, ts) in enumerate(unadded):
            speaker_tag = 'speakerA' if speaker == 'A' else 'speakerB'
            non_speaker_tag = 'speakerB' if speaker == 'A' else 'speakerA'

            # TODO: Check this -> might cause bug of interruption mislabelled
            token = token if token != eint_token_id else tokens['eot']

            if idx - 1 >= 0 and unadded[idx - 1][1] == word_idx:
                word_idx += 1

            output[speaker_tag]['tokens'].insert(word_idx, token)
            output[speaker_tag]['timings'].insert(word_idx, (-1, -1))
            output[speaker_tag]['token_type_ids'].insert(
                word_idx, ts['turn_type'] if type == "end" else TurnType.NONE)
            if self.remove_start_tokens:
                other = ts['other_turn_type'][0]
            else:
                other = TurnType.NONE
            output[speaker_tag]['other_token_type_ids'].insert(word_idx, other)
            output[speaker_tag]['turn_overlap'].insert(word_idx, 0)

            output[non_speaker_tag]['tokens'].insert(word_idx, emp_token_id)
            output[non_speaker_tag]['timings'].insert(word_idx, (-1, -1))
            # BREAKING CHANGE?
            output[non_speaker_tag]['token_type_ids'].insert(word_idx, TurnType.NONE)
            if self.remove_start_tokens:
                other = ts['other_turn_type'][0]
            else:
                other = TurnType.NONE
            output[non_speaker_tag]['other_token_type_ids'].insert(word_idx, other)
            output[non_speaker_tag]['turn_overlap'].insert(word_idx, 0)

            offset += 1

        # Fix other_token_type_ids so that only set to YIELD when other speaker
        # is actually speaking so not for the entire turn
        check_list = torch.tensor([TurnType.INTERRUPT, TurnType.OVERLAP])

        return output

    """
    Tokenizes each speaker's dialog and adds tokens to each channel 
    in increasing time where one or both speakers can speak at the same time. 
    If one speaker is speaking then the other speaker has a <emp> token.
    It is aligned based on when the current earliest token begins speaking

    Tokenizes each speaker's dialog and for each turn it is classified as either
    a NORMAL, INTERRUPT, OVERLAP or BACKCHANNEL turn (ts['turn_type']). 
    OVERLAP and BACKCHANNELis identified previously by SwitchboardDataset 
    as completely overlapped utterances
    Additionally, ts['other_turn_type'] is used to label the turn endings,
    so either NORMAL or YIELD. A YIELD is when the other speaker overlaps
    with the end-of-turn by at least `interruption_thresh` seconds or an 
    overlap is within `overlap_thresh` seconds of the end of the turn 

    Additional, time until the end of the turn is added for each token and 
    debugging responses. 

    Afterwards, we work on the token-level scope by adding words one at a time
    from each speaker depending on their degree of overlap
    End-of-turn tokens are added to the end of each speaker's dialog according
    to turn types. 
    """
    def tokenize_with_overlap(self):
        self.logger.info(f"data ({self.split}): tokenizing data")

        result = []
        pbar = tqdm.tqdm(total=len(self.dataset), desc="Tokenizing")

        tokens_dict = {
            'emp': self.tokenizer.convert_tokens_to_ids("<emp>"),
            'sot': self.tokenizer.convert_tokens_to_ids("<sot>"),
            'eot': self.tokenizer.convert_tokens_to_ids("<eot>"),
            'sbc': self.tokenizer.convert_tokens_to_ids("<sbc>"),
            'ebc': self.tokenizer.convert_tokens_to_ids("<ebc>"),
            'sint': self.tokenizer.convert_tokens_to_ids("<sint>"),
            'eint': self.tokenizer.convert_tokens_to_ids("<eint>"),
            'yield': self.tokenizer.convert_tokens_to_ids("<yield>"),
            'sil': self.tokenizer.convert_tokens_to_ids("<sil>"),
        }
        emp_token_id = tokens_dict['emp']
        sil_token_id = tokens_dict['sil'] if self.include_sil_token else tokens_dict['emp']

        metric_ts = {
            'mean': 0,
            'std': 0,
        }
        metric_other_ts = {
            'mean': 0,
            'std': 0,
        }
        n1, n2 = 0, 0
        for dataset in self.dataset:
            output = {
                "speakerA": {
                    'dialog': '',
                    'tokens': [],
                    'timings': [],
                    'conv_id': None
                },
                "speakerB": {
                    'dialog': '',
                    'tokens': [],
                    'timings': [],
                    'conv_id': None
                }
            }

            if len(dataset['dialog']) != 2:
                self.logger.warn("Requires channel splitting")

            """
            Helper function to extract tokens from each speaker's dialog
            """
            def get_tokens(speakerA, speakerB):
                sentenceA = " ".join(feature['text'] for feature in speakerA)
                sentenceB = " ".join(feature['text'] for feature in speakerB)

                tokensA = self.tokenize_sentence(sentenceA)
                tokensB = self.tokenize_sentence(sentenceB)

                dialogA = self.match_tokens_words(tokensA, speakerA, sentenceA)
                dialogB = self.match_tokens_words(tokensB, speakerB, sentenceB)

                return sentenceA, sentenceB, tokensA, tokensB, dialogA, dialogB

            speakerA = dataset['dialog']['speakerA']
            speakerB = dataset['dialog']['speakerB']
            sentenceA, sentenceB, tokensA, tokensB, dialogA, dialogB = get_tokens(speakerA, speakerB)

            speakerbcA = dataset['backchannel']['speakerA']
            speakerbcB = dataset['backchannel']['speakerB']
            _, _, _, _, bcA, bcB = get_tokens(speakerbcA, speakerbcB)

            speaker_overlapA = dataset['overlap']['speakerA']
            speaker_overlapB = dataset['overlap']['speakerB']
            _, _, _, _, overlapA, overlapB = get_tokens(speaker_overlapA, speaker_overlapB)

            ipusA = self._get_ipus(dialogA)
            ipusB = self._get_ipus(dialogB)

            """
            turn shift level structure that is based on TurnGPT
            perform all augmentations required on turn level so that 
            we can simply walk though the list of tokens and align appropriately
            """
            tsA, tsB = self._get_ts(dialogA, ipusA, dialogB, ipusB)
            if not self.remove_backchannels:
                dialogA, tsA, _, dialogB, tsB, _ = self._insert_bc(dialogA, tsA, bcA, dialogB, tsB, bcB)
            if not self.remove_overlaps:
                dialogB, tsA, _, dialogB, tsB, _ = self._insert_overlap(dialogA, tsA, overlapA, dialogB, tsB, overlapB,
                                                                        allow_joins=False)

            tsA, tsB = self._add_special_turn_types(dialogA, tsA, dialogB, tsB)
            tsA, tsB = self._add_turn_lengths(dialogA, tsA, dialogB, tsB)
            tsA = self.add_time_until_ts(dialogA, tsA)
            tsB = self.add_time_until_ts(dialogB, tsB)

            metric_ts, metric_other_ts, n1, n2 = self.calculate_metrics(tsA + tsB, metric_ts, metric_other_ts, n1, n2)

            if self.store_raw:
                self.ts.append({
                    'speakerA': tsA,
                    'speakerB': tsB,
                })
            if isinstance(dialogA[0], list):
                dialogA = [word for sentence in dialogA for word in sentence]
                dialogB = [word for sentence in dialogB for word in sentence]

            # Aligning individual tokens
            k, l = 0, 0
            while k < len(dialogA) and l < len(dialogB):
                # Add A dialog first
                if dialogA[k]['start'] <= dialogB[l]['start']:
                    output['speakerA']['tokens'].append(
                        dialogA[k]['tokens'])
                    output['speakerA']['timings'].append(
                        [dialogA[k]['start'], dialogA[k]['end']])

                    # Adding overlaps
                    # Current B word is closer to end of current A word then the start of next A word
                    # But still should start prior to end of current A word
                    # By at least half of the duration
                    gap_currB_currA = abs(
                        dialogB[l]['start'] - dialogA[k]['start'])
                    if k < len(dialogA) - 2:
                        gap_currB_nextA = abs(
                            dialogB[l]['start'] - dialogA[k + 1]['start'])

                    durationA = dialogA[k]['end'] - dialogA[k]['start']
                    durationB = dialogB[l]['end'] - dialogB[l]['start']
                    half_duration = durationA / 2
                    if durationA > durationB:
                        half_duration = durationB / 2

                    overlap = dialogA[k]['end'] - dialogB[l]['start']
                    coverage = overlap > 0 and overlap > half_duration
                    # B is closer to current A then the next A word
                    if coverage:  # and gap_currB_currA < gap_currB_nextA and starts_before_currA:
                        output['speakerB']['tokens'].append(
                            dialogB[l]['tokens'])
                        output['speakerB']['timings'].append(
                            [dialogB[l]['start'], dialogB[l]['end']])

                        l += 1
                    else:
                        output['speakerB']['tokens'].append(emp_token_id)
                        output['speakerB']['timings'].append(
                            [dialogA[k]['start'], dialogA[k]['end']])

                    k += 1

                # Add B dialog first
                elif dialogB[l]['start'] < dialogA[k]['start']:
                    output['speakerB']['tokens'].append(
                        dialogB[l]['tokens'])
                    output['speakerB']['timings'].append(
                        [dialogB[l]['start'], dialogB[l]['end']])

                    # Current A word is closer to the end of the current B word than the start of the next B word
                    # But still should start prior to end of current A word

                    durationA = dialogA[k]['end'] - dialogA[k]['start']
                    durationB = dialogB[l]['end'] - dialogB[l]['start']
                    half_duration = durationA / 2
                    if durationA > durationB:
                        half_duration = durationB / 2

                    overlap = dialogB[l]['end'] - dialogA[k]['start']
                    coverage = overlap > 0 and overlap > half_duration

                    if coverage:  # and gap_currA_currB < gap_currA_nextB and starts_before_currA:
                        output['speakerA']['tokens'].append(
                            dialogA[k]['tokens'])
                        output['speakerA']['timings'].append(
                            [dialogA[k]['start'], dialogA[k]['end']])

                        k += 1
                    else:
                        output['speakerA']['tokens'].append(emp_token_id)
                        output['speakerA']['timings'].append(
                            [dialogB[l]['start'], dialogB[l]['end']])
                    l += 1

            # Add leftover tokens
            while k < len(dialogA):
                output['speakerA']['tokens'].append(dialogA[k]['tokens'])
                output['speakerA']['timings'].append(
                    [dialogA[k]['start'], dialogA[k]['end']])
                output['speakerB']['tokens'].append(
                    emp_token_id)
                output['speakerB']['timings'].append(
                    [dialogA[k]['start'], dialogA[k]['end']])
                k += 1
            while l < len(dialogB):
                output['speakerB']['tokens'].append(dialogB[l]['tokens'])
                output['speakerB']['timings'].append(
                    [dialogB[l]['start'], dialogB[l]['end']])

                output['speakerA']['tokens'].append(
                    emp_token_id)
                output['speakerA']['timings'].append(
                    [dialogB[l]['start'], dialogB[l]['end']])
                l += 1

            output['speakerA'], unaddedA = self.add_ts_token_dialog(
                output['speakerA'], tsA, tokens_dict, speaker='A')
            output['speakerB'], unaddedB = self.add_ts_token_dialog(
                output['speakerB'], tsB, tokens_dict, speaker='B')

            output['speakerA'], _ = self.add_ts_token_as_type(
                output['speakerA'], tsA, tokens_dict, speaker='A')
            output['speakerB'], _ = self.add_ts_token_as_type(
                output['speakerB'], tsB, tokens_dict, speaker='B')

            output = self.fix_ts_dialog(
                output, unaddedA + unaddedB, tokens_dict)

            output['speakerA']['input_ids'] = torch.tensor(
                output['speakerA']['tokens'])
            output['speakerA']['token_type_ids'] = torch.tensor(
                output['speakerA']['token_type_ids'])
            output['speakerA']['attention_mask'] = (
                    output['speakerA']['input_ids'] != emp_token_id).long()
            output['speakerA']['other_token_type_ids'] = torch.tensor(output['speakerA']['other_token_type_ids'])
            output['speakerA']['turn_overlap'] = torch.tensor(output['speakerA']['turn_overlap'])
            output['speakerA']['time_until_ts'] = torch.tensor(output['speakerA']['time_until_ts'])
            output['speakerA']['speaker_ids'] = torch.where(
                torch.ne(output['speakerA']['token_type_ids'], TurnType.NONE),
                torch.tensor(self.speakerA_token, device=output['speakerA']['input_ids'].device),
                torch.tensor(0, device=output['speakerA']['input_ids'].device))
            output['speakerA']['conv_id'] = speakerA[0]['conv_id']

            output['speakerB']['input_ids'] = torch.tensor(
                output['speakerB']['tokens'])
            output['speakerB']['token_type_ids'] = torch.tensor(
                output['speakerB']['token_type_ids'])
            output['speakerB']['other_token_type_ids'] = torch.tensor(output['speakerB']['other_token_type_ids'])
            output['speakerB']['attention_mask'] = (
                    output['speakerB']['input_ids'] != emp_token_id).long()
            output['speakerB']['turn_overlap'] = torch.tensor(output['speakerB']['turn_overlap'])
            output['speakerB']['time_until_ts'] = torch.tensor(output['speakerB']['time_until_ts'])
            output['speakerB']['speaker_ids'] = torch.where(
                torch.ne(output['speakerB']['token_type_ids'], TurnType.NONE),
                torch.tensor(self.speakerB_token, device=output['speakerB']['input_ids'].device),
                torch.tensor(0, device=output['speakerB']['input_ids'].device))
            output['speakerB']['conv_id'] = speakerB[0]['conv_id']

            assert output['speakerA']['input_ids'].shape == output['speakerB'][
                'input_ids'].shape, f"not matching shape {output['speakerA']['input_ids'].shape} == {output['speakerB']['input_ids'].shape}"
            assert output['speakerA']['input_ids'].shape == output['speakerA'][
                'token_type_ids'].shape, f"{output['speakerA']['input_ids'].shape} {output['speakerA']['token_type_ids'].shape}"
            assert 'turn_overlap' in output['speakerA']
            assert 'turn_overlap' in output['speakerB']

            result.append(output)
            pbar.update(1)

        pbar.close()
        self.logger.info(f"data ({self.split}): finished tokenizing dataset")

        metrics = {
            'time_until_ts': metric_ts,
            'time_until_other_ts': metric_other_ts,
        }
        return result, metrics

    """
    Tokenizes data like in `tokenize_with_overlap` on the turn-level, so 
    we get the correct turn types for each utterance.
    However, after getting the turns and the yields as if the overlaps are present
    they are not added into the actual token-level transcripts.
    """
    def tokenize_without_overlap(self):
        results = []

        emp_token_id = self.tokenizer.convert_tokens_to_ids('<emp>')

        eot_token_id = self.tokenizer.convert_tokens_to_ids('<eot>')
        eot_token_idA = self.tokenizer.convert_tokens_to_ids('<speakerA>')
        eot_token_idB = self.tokenizer.convert_tokens_to_ids('<speakerB>')

        sot_token_id = self.tokenizer.convert_tokens_to_ids('<sot>')
        sot_token_idA = self.tokenizer.convert_tokens_to_ids('<speakerA>')
        sot_token_idB = self.tokenizer.convert_tokens_to_ids('<speakerB>')

        sint_token_id = self.tokenizer.convert_tokens_to_ids('<sint>')
        eint_token_id = self.tokenizer.convert_tokens_to_ids('<eint>')
        ebc_token_id = self.tokenizer.convert_tokens_to_ids('<ebc>')
        yield_token_id = self.tokenizer.convert_tokens_to_ids('<yield>')

        if not self.individual_ts:
            eot_token_idA = eot_token_id
            eot_token_idB = eot_token_id

            sot_token_idA = sot_token_id
            sot_token_idB = sot_token_id
        if self.remove_start_tokens:
            sot_token_idA = emp_token_id
            sot_token_idB = emp_token_id

        tokens_dict = {
            'emp': self.tokenizer.convert_tokens_to_ids("<emp>"),
            'sot': self.tokenizer.convert_tokens_to_ids("<sot>"),
            'eot': self.tokenizer.convert_tokens_to_ids("<eot>"),
            'sbc': self.tokenizer.convert_tokens_to_ids("<sbc>"),
            'ebc': self.tokenizer.convert_tokens_to_ids("<ebc>"),
            'sint': self.tokenizer.convert_tokens_to_ids("<sint>"),
            'eint': self.tokenizer.convert_tokens_to_ids("<eint>"),
            'yield': self.tokenizer.convert_tokens_to_ids("<yield>"),
            'sil': self.tokenizer.convert_tokens_to_ids("<sil>"),
        }

        def get_tokens(speakerA, speakerB):

            sentenceA = " ".join(feature['text'] for feature in speakerA)
            sentenceB = " ".join(feature['text'] for feature in speakerB)

            tokensA = self.tokenize_sentence(sentenceA)
            tokensB = self.tokenize_sentence(sentenceB)

            dialogA = self.match_tokens_words(tokensA, speakerA, sentenceA)
            dialogB = self.match_tokens_words(tokensB, speakerB, sentenceB)

            return sentenceA, sentenceB, tokensA, tokensB, dialogA, dialogB

        def get_other_ts_turn_type(ts, l):
            for i in range(l-1, -1, -1):
                if ts[i]['turn_type'] not in {TurnType.BACKCHANNEL, TurnType.OVERLAP}:
                    return ts[i]['other_turn_type'][0]
            return 0

        def get_ts_turn_type(ts, l):
            for i in range(l-1, -1, -1):
                if ts[i]['turn_type'] not in {TurnType.BACKCHANNEL, TurnType.OVERLAP}:
                    return ts[i]['turn_type']
            return 0

        def add_to_speaker(output, dialog, ts, other_ts, k: int, l: int, speaker="speakerA", other_speaker="speakerB",
                           last=False):

            if speaker == "speakerA":
                eot_id = eot_token_idB
                sot_id = sot_token_idA
            else:
                eot_id = eot_token_idA
                sot_id = sot_token_idB

            # Add <eot> if not the first utterance
            if len(output[speaker]['tokens']) == 0 or output[speaker]['tokens'][-1] in tokens_dict.values():
                output[speaker]['tokens'].append(sot_id)
                output[speaker]['timings'].append([-1, -1])
                output[speaker]['token_type_ids'].append(0)
                output[speaker]['turn_overlap'].append(ts[k]['turn_overlap'])
                output[speaker]['other_token_type_ids'].append(0)
                output[speaker]['time_until_ts'].append(0)
                output[speaker]['time_until_other_ts'].append(0)
            else:
                turn_type_other = 0
                turn_type = 1
                if k > 0 and k <= len(ts):
                    turn_type_other = get_other_ts_turn_type(ts, k)
                    turn_type = get_ts_turn_type(ts, k)

                    if self.include_overlap_token and ts[k - 1]['turn_type'] == TurnType.OVERLAP:
                        eot_id = eint_token_id
                    elif self.include_end_bc_token and ts[k - 1]['turn_type'] == TurnType.BACKCHANNEL:
                        eot_id = ebc_token_id

                output[speaker]['tokens'].append(eot_id)
                output[speaker]['timings'].append([-1, -1])
                output[speaker]['token_type_ids'].append(turn_type)
                output[speaker]['turn_overlap'].append(0)
                output[speaker]['other_token_type_ids'].append(turn_type_other)
                output[speaker]['time_until_ts'].append(0)
                output[speaker]['time_until_other_ts'].append(0)

            turn_type_other = 0
            turn_type = 1
            if l <= len(other_ts) and l > 0:
                turn_type_other = get_other_ts_turn_type(other_ts, l)
                turn_type = get_ts_turn_type(other_ts, l)

                if self.include_overlap_token and output[other_speaker]['token_type_ids'] == TurnType.OVERLAP:
                    eot_id = eint_token_id
                elif self.include_end_bc_token and output[other_speaker]['token_type_ids'] == TurnType.BACKCHANNEL:
                    eot_id = ebc_token_id

            if len(output[other_speaker]['tokens']) == 0 or output[other_speaker]['tokens'][
                -1] not in tokens_dict.values():
                output[other_speaker]['tokens'].append(eot_id)
                output[other_speaker]['timings'].append([-1, -1])
                output[other_speaker]['token_type_ids'].append(turn_type)
                output[other_speaker]['turn_overlap'].append(0)
                output[other_speaker]['other_token_type_ids'].append(turn_type_other)
                output[other_speaker]['time_until_ts'].append(0)
                output[other_speaker]['time_until_other_ts'].append(0)
            else:
                output[other_speaker]['tokens'].append(sot_id)
                output[other_speaker]['timings'].append([-1, -1])
                output[other_speaker]['token_type_ids'].append(0)
                output[other_speaker]['turn_overlap'].append(0)
                output[other_speaker]['other_token_type_ids'].append(turn_type_other)
                output[other_speaker]['time_until_ts'].append(0)
                output[other_speaker]['time_until_other_ts'].append(0)

            if k == 0 and l == 0 and len(output[other_speaker]['tokens']) > 0:
                output[other_speaker]['tokens'][-1] = emp_token_id

            # Add A <ts> as all turns should be in order ot <ts> and known should be overlapping aside from
            # those that are properly labelled so are pruned
            idxs = [idx for idx in range(ts[k]['start_idx'], ts[k]['end_idx'] + 1)]
            if idxs[-1] >= len(dialog):
                pass
            output[speaker]['tokens'].extend([dialog[idx]['tokens'] for idx in idxs])
            output[speaker]['timings'].extend([[dialog[idx]['start'], dialog[idx]['end']] for idx in idxs])
            output[speaker]['token_type_ids'].extend([ts[k]['turn_type'] for _ in idxs])

            turn_overlap_speaker = [ts[k]['turn_overlap'] for _ in idxs]
            if last:
                turn_overlap_speaker = [0 for _ in idxs]

            output[speaker]['turn_overlap'].extend(turn_overlap_speaker)
            output[other_speaker]['turn_overlap'].extend([0 for _ in idxs])

            output[speaker]['time_until_ts'].extend(ts[k]['time_until_ts'])
            output[speaker]['time_until_other_ts'].extend(ts[k]['time_until_other_ts'])

            output[other_speaker]['time_until_ts'].extend([0 for _ in idxs])
            output[other_speaker]['time_until_other_ts'].extend([0 for _ in idxs])

            # Essentially where other speaker is speaking in actual speech
            output[speaker]['other_token_type_ids'].extend([TurnType.NONE for _ in idxs])
            output[other_speaker]['other_token_type_ids'].extend([TurnType.NONE for _ in idxs])

            output[other_speaker]['tokens'].extend([emp_token_id for _ in idxs])
            output[other_speaker]['timings'].extend([[dialog[idx]['start'], dialog[idx]['end']] for idx in idxs])
            output[other_speaker]['token_type_ids'].extend([TurnType.NONE for _ in idxs])
            k += 1

            return output, k

        def update_ts(output):
            if not self.include_yield_token:
                return output

            eot_token_id = self.tokenizer.convert_tokens_to_ids('<eot>')
            end_token_id = self.tokenizer.convert_tokens_to_ids('<yield>') if self.include_yield_token else eot_token_id

            sot_token_id = self.tokenizer.convert_tokens_to_ids('<sot>')
            start_token_id = self.tokenizer.convert_tokens_to_ids(
                '<sint>') if self.include_yield_token else sot_token_id

            for idx in range(len(output['tokens'])):
                if idx < len(output) - 1 and output['tokens'][idx] == sot_token_id:
                    if output['token_type_ids'][idx + 1] == TurnType.INTERRUPT:
                        output[idx]['tokens'] = start_token_id
                if idx > 0 and output['tokens'][idx] == eot_token_id:
                    if output['other_token_type_ids'][idx - 1] == TurnType.YIELD:
                        output['tokens'][idx] = end_token_id

            return output

        def add_other_turn_overlaps(output, ts, dialog, speaker="speakerA", other_speaker="speakerB"):
            k = 0
            in_ts = False
            ts_idx = 0
            while ts_idx < len(ts) and k < len(output[speaker]['tokens']):
                utt = ts[ts_idx]

                if utt['turn_type'] not in {TurnType.BACKCHANNEL, TurnType.INTERRUPT, TurnType.YIELD, TurnType.OVERLAP}:
                    ts_idx += 1
                    continue

                # Check if word timing of B is in the current A word
                # If 50 % of B is covered by A then it can be seen as the same time
                A_timing = output[speaker]['timings'][k]
                if A_timing[0] + A_timing[1] == -2:
                    k += 1
                    continue

                if not in_ts:
                    B_timing = dialog[utt['start_idx']]['start'], dialog[utt['start_idx']]['end']
                    # First check if there is any overlap
                    if A_timing[0] < B_timing[0]:
                        if A_timing[1] < B_timing[0]:
                            # A starts and ends prior to beginning of B
                            k += 1
                            continue
                        # A starts but ends prior to beginning of B
                    elif B_timing[0] <= A_timing[0]:
                        # B started prior to A so should just be added
                        pass
                    in_ts = True

                # Check if utt_b is over
                # So if A has passed the end of B
                B_timing = dialog[utt['end_idx']]['start'], dialog[utt['end_idx']]['end']
                if B_timing[1] < A_timing[0]:
                    ts_idx += 1
                    in_ts = False

                # Just add current B word as token_type to A
                if output[speaker]['tokens'][k] != emp_token_id and in_ts:
                    output[speaker]['other_token_type_ids'][k] = TurnType.OVERLAP

                k += 1
            return output

        def add_other_token_type_ids(output, ts, dialog, speaker="speakerA", other_speaker="speakerB"):
            idx = 0
            in_ts = False
            in_yield = False
            curr_ts_idx = 0
            word_idx = 0

            added_ts = False

            while curr_ts_idx < len(ts) and idx < len(output[speaker]['tokens']):
                # idx tracks position in tokens that contains <emp>
                # word_idx tracks position with respect to one speaer
                # <ts> corresponds to turns but some do not appear within speaker_tokens
                # Just iterate through dialogs while keeping track of current turn shift and
                # Wait for the interruption
                if curr_ts_idx < len(ts) and ts[curr_ts_idx]['other_turn_type'][0] != TurnType.YIELD:
                    curr_ts_idx += 1
                    continue

                # Yield point should be between start and end. Time within <ts> is the point of interruption
                # That is definitely after the start of the interruption
                if output[speaker]['tokens'][idx] in tokens_dict.values():
                    idx += 1
                    continue

                if output[speaker]['timings'][idx][1] > dialog[ts[curr_ts_idx]['end_idx']]['end']:
                    curr_ts_idx += 1
                    idx += 1
                    continue

                if ts[curr_ts_idx]['other_turn_type'][1] <= output[speaker]['timings'][idx][0]:
                    output[speaker]['other_token_type_ids'][idx] = TurnType.YIELD

                idx += 1

            return output

        metric_ts = {
            'mean': 0,
            'std': 0,
        }
        metric_other_ts = {
            'mean': 0,
            'std': 0,
        }
        n1, n2 = 0, 0

        for dataset in self.dataset:
            output = {
                "speakerA": {
                    'dialog': '',
                    'tokens': [],
                    'token_type_ids': [],
                    'other_token_type_ids': [],
                    'timings': [],
                    'turn_overlap': [],
                    'time_until_ts': [],
                    'time_until_other_ts': [],
                    'key': None
                },
                "speakerB": {
                    'dialog': '',
                    'tokens': [],
                    'token_type_ids': [],
                    'other_token_type_ids': [],
                    'timings': [],
                    'turn_overlap': [],
                    'time_until_ts': [],
                    'time_until_other_ts': [],
                    'key': None
                }
            }

            speakerA = dataset['dialog']['speakerA']
            speakerB = dataset['dialog']['speakerB']
            sentenceA, sentenceB, tokensA, tokensB, dialogA, dialogB = get_tokens(speakerA, speakerB)

            speakerbcA = dataset['backchannel']['speakerA']
            speakerbcB = dataset['backchannel']['speakerB']
            _, _, _, _, bcA, bcB = get_tokens(speakerbcA, speakerbcB)

            speaker_overlapA = dataset['overlap']['speakerA']
            speaker_overlapB = dataset['overlap']['speakerB']
            _, _, _, _, overlapA, overlapB = get_tokens(speaker_overlapA, speaker_overlapB)

            ipusA = self._get_ipus(dialogA)
            ipusB = self._get_ipus(dialogB)

            # Assuming we have utterances from transcriptions
            # Just have to deal with overlaps
            tsA, tsB = self._get_ts(dialogA, ipusA, dialogB, ipusB)
            dialogA, tsA, _, dialogB, tsB, _ = self._insert_bc(dialogA, tsA, bcA, dialogB, tsB, bcB)
            dialogA, tsA, _, dialogB, tsB, _ = self._insert_overlap(dialogA, tsA, overlapA, dialogB, tsB, overlapB,
                                                                    allow_joins=False)

            tsA, tsB = self._add_special_turn_types(dialogA, tsA, dialogB, tsB, add_interruption_idx=True)
            tsA, tsB = self._add_turn_lengths(dialogA, tsA, dialogB, tsB)
            tsA = self.add_time_until_ts(dialogA, tsA)
            tsB = self.add_time_until_ts(dialogB, tsB)


            metric_ts, metric_other_ts, n1, n2 = self.calculate_metrics(tsA + tsB, metric_ts, metric_other_ts, n1, n2)

            k, l = 0, 0

            if isinstance(dialogA[0], list):
                dialogA = [word for sentence in dialogA for word in sentence]
                dialogB = [word for sentence in dialogB for word in sentence]

            current_speaker = 'B' if dialogA[0]['start'] < dialogB[0]['start'] else 'A'

            while k < len(tsA) and l < len(tsB):
                if self.remove_overlaps and tsA[k]['turn_type'] not in {TurnType.NORMAL, TurnType.INTERRUPT}:
                    k += 1
                    continue
                if self.remove_overlaps and tsB[l]['turn_type'] not in {TurnType.NORMAL, TurnType.INTERRUPT}:
                    l += 1
                    continue

                if dialogA[tsA[k]['start_idx']]['start'] < dialogB[tsB[l]['start_idx']]['start']:
                    output, k = add_to_speaker(output, dialogA, tsA, tsB, k, l)
                else:
                    output, l = add_to_speaker(output, dialogB, tsB, tsA, l, k, speaker="speakerB",
                                               other_speaker="speakerA")

            for i in range(k, len(tsA)):
                if tsA[i]['turn_type'] not in {TurnType.NORMAL, TurnType.INTERRUPT}:
                    continue

                output, _ = add_to_speaker(output, dialogA, tsA, tsB, i, l, last=k == len(tsA) - 1)

            for i in range(l, len(tsB)):
                if tsB[i]['turn_type'] not in {TurnType.NORMAL, TurnType.INTERRUPT}:
                    continue

                output, _ = add_to_speaker(output, dialogB, tsB, tsA, i, l, speaker="speakerB",
                                           other_speaker="speakerA",
                                           last=l == len(tsB) - 1)

            # Add 'other_token_type_ids' by looping through tokens and timings of one speaker and finding where to insert
            # other speaker's `actual` time of utterance to find points of overlap

            output = add_other_turn_overlaps(output, tsB, dialogB, speaker="speakerA", other_speaker="speakerB")
            output = add_other_turn_overlaps(output, tsA, dialogA, speaker="speakerB", other_speaker="speakerA")

            output = add_other_token_type_ids(output, tsA, dialogA, speaker="speakerA", other_speaker="speakerB")
            output = add_other_token_type_ids(output, tsB, dialogB, speaker="speakerB", other_speaker="speakerA")

            output['speakerA'] = update_ts(output['speakerA'])
            output['speakerB'] = update_ts(output['speakerB'])

            # Add last
            output['speakerA']['tokens'].append(eot_token_idA)
            output['speakerA']['token_type_ids'].append(0)
            output['speakerA']['other_token_type_ids'].append(0)
            output['speakerA']['timings'].append([-1, -1])
            output['speakerA']['time_until_ts'].append(0)
            output['speakerA']['time_until_other_ts'].append(0)
            output['speakerA']['turn_overlap'].append(0)

            output['speakerB']['tokens'].append(eot_token_idB)
            output['speakerB']['token_type_ids'].append(0)
            output['speakerB']['other_token_type_ids'].append(0)
            output['speakerB']['timings'].append([-1, -1])
            output['speakerB']['time_until_ts'].append(0)
            output['speakerB']['time_until_other_ts'].append(0)
            output['speakerB']['turn_overlap'].append(0)
            if output['speakerA']['tokens'][-2] == emp_token_id:
                output['speakerA']['tokens'][-1] = emp_token_id
            elif output['speakerB']['tokens'][-2] == emp_token_id:
                output['speakerB']['tokens'][-1] = emp_token_id

            # Add buffer
            output['speakerA']['tokens'].append(emp_token_id)
            output['speakerA']['token_type_ids'].append(0)
            output['speakerA']['other_token_type_ids'].append(0)
            output['speakerA']['timings'].append([-1, -1])
            output['speakerA']['time_until_ts'].append(0)
            output['speakerA']['time_until_other_ts'].append(0)
            output['speakerA']['turn_overlap'].append(0)

            output['speakerB']['tokens'].append(emp_token_id)
            output['speakerB']['token_type_ids'].append(0)
            output['speakerB']['other_token_type_ids'].append(0)
            output['speakerB']['timings'].append([-1, -1])
            output['speakerB']['time_until_ts'].append(0)
            output['speakerB']['time_until_other_ts'].append(0)
            output['speakerB']['turn_overlap'].append(0)

            output['speakerA']['input_ids'] = output['speakerA']['tokens']
            output['speakerB']['input_ids'] = output['speakerB']['tokens']

            assert len(output['speakerA']['input_ids']) == len(output['speakerA']['token_type_ids'])
            assert len(output['speakerA']['input_ids']) == len(output['speakerA']['other_token_type_ids'])
            assert len(output['speakerA']['input_ids']) == len(output['speakerA']['turn_overlap'])

            assert len(output['speakerA']['input_ids']) == len(output['speakerB']['input_ids'])

            assert len(output['speakerB']['input_ids']) == len(output['speakerB']['token_type_ids'])
            assert len(output['speakerB']['input_ids']) == len(output['speakerB']['other_token_type_ids'])
            assert len(output['speakerB']['input_ids']) == len(output['speakerB']['turn_overlap'])

            if len(output['speakerA']['time_until_ts']) != len(output['speakerA']['input_ids']):
                pass
            if len(output['speakerB']['time_until_ts']) != len(output['speakerB']['input_ids']):
                pass

            output['speakerA']['input_ids'] = torch.tensor(
                output['speakerA']['tokens'])
            output['speakerA']['token_type_ids'] = torch.tensor(
                output['speakerA']['token_type_ids'])
            output['speakerA']['attention_mask'] = (
                    output['speakerA']['input_ids'] != emp_token_id).long()
            output['speakerA']['other_token_type_ids'] = torch.tensor(output['speakerA']['other_token_type_ids'])
            output['speakerA']['turn_overlap'] = torch.tensor(output['speakerA']['turn_overlap'])
            output['speakerA']['time_until_ts'] = torch.tensor(output['speakerA']['time_until_ts'])
            output['speakerA']['time_until_other_ts'] = torch.tensor(output['speakerA']['time_until_other_ts'])
            output['speakerA']['timings'] = torch.tensor(output['speakerA']['timings'])
            output['speakerA']['speaker_ids'] = torch.where(
                torch.ne(output['speakerA']['token_type_ids'], TurnType.NONE),
                torch.tensor(1, device=output['speakerA']['input_ids'].device),
                torch.tensor(0, device=output['speakerA']['input_ids'].device))
            output['speakerA']['conv_id'] = speakerA[0]['conv_id']

            output['speakerB']['input_ids'] = torch.tensor(
                output['speakerB']['tokens'])
            output['speakerB']['token_type_ids'] = torch.tensor(
                output['speakerB']['token_type_ids'])
            output['speakerB']['attention_mask'] = (
                    output['speakerB']['input_ids'] != emp_token_id).long()
            output['speakerB']['other_token_type_ids'] = torch.tensor(output['speakerB']['other_token_type_ids'])
            output['speakerB']['turn_overlap'] = torch.tensor(output['speakerB']['turn_overlap'])
            output['speakerB']['time_until_ts'] = torch.tensor(output['speakerB']['time_until_ts'])
            output['speakerB']['time_until_other_ts'] = torch.tensor(output['speakerB']['time_until_other_ts'])
            output['speakerB']['timings'] = torch.tensor(output['speakerB']['timings'])
            output['speakerB']['speaker_ids'] = torch.where(
                torch.ne(output['speakerB']['token_type_ids'], TurnType.NONE),
                torch.tensor(2, device=output['speakerB']['input_ids'].device),
                torch.tensor(0, device=output['speakerB']['input_ids'].device))
            output['speakerB']['conv_id'] = speakerB[0]['conv_id']

            results.append(output)

        metrics = {
            'time_until_ts': metric_ts,
            'time_until_other_ts': metric_other_ts,
        }
        return results, metrics

    def tokenize_without_emp(self):
        results = []
        sot_token_id = self.tokenizer.convert_tokens_to_ids("<sot>")
        eot_token_id = self.tokenizer.convert_tokens_to_ids("<eot>")
        eint_token_id = self.tokenizer.convert_tokens_to_ids("<eint>")
        ebc_token_id = self.tokenizer.convert_tokens_to_ids("<ebc>")
        emp_token_id = self.tokenizer.convert_tokens_to_ids("<emp>")

        tokens_dict = {
            'emp': self.tokenizer.convert_tokens_to_ids("<emp>"),
            'sot': self.tokenizer.convert_tokens_to_ids("<sot>"),
            'eot': self.tokenizer.convert_tokens_to_ids("<eot>"),
            'sbc': self.tokenizer.convert_tokens_to_ids("<sbc>"),
            'ebc': self.tokenizer.convert_tokens_to_ids("<ebc>"),
            'sint': self.tokenizer.convert_tokens_to_ids("<sint>"),
            'eint': self.tokenizer.convert_tokens_to_ids("<eint>"),
            'yield': self.tokenizer.convert_tokens_to_ids("<yield>"),
            'sil': self.tokenizer.convert_tokens_to_ids("<sil>"),
        }

        results, metrics = self.tokenize_without_overlap()
        for result in results:
            for speaker_key, speaker in result.items():
                mask = torch.logical_and(torch.ne(speaker['input_ids'], emp_token_id),
                                         torch.ne(speaker['input_ids'], sot_token_id))
                result[speaker_key]['input_ids'] = speaker['input_ids'][mask]
                result[speaker_key]['token_type_ids'] = speaker['token_type_ids'][mask]
                result[speaker_key]['other_token_type_ids'] = speaker['other_token_type_ids'][mask]
                result[speaker_key]['time_until_ts'] = speaker['time_until_ts'][mask]
                result[speaker_key]['time_until_other_ts'] = speaker['time_until_other_ts'][mask]
                result[speaker_key]['timings'] = speaker['timings'][mask]
                result[speaker_key]['turn_overlap'] = speaker['turn_overlap'][mask]
                result[speaker_key]['speaker_ids'] = speaker['speaker_ids'][mask]
                result[speaker_key]['attention_mask'] = speaker['attention_mask'][mask]

        return results, metrics

    def tokenize_sentence(self, dialog):
        tokens = self.tokenizer(
            dialog,
            padding="do_not_pad",
            truncation=True,
            max_length=12400,
            return_offsets_mapping=True,
            return_tensors="pt")
        return tokens

    """
    Splits each dialog into sequences of `max_length` tokens or whatever is left with overlap length in token overlap
    between sequences.
    Also pads to max_length
    """

    def split_to_length(self, pairwise=True):
        # Augment entirety of datasets
        result = []
        start_idx = 0
        end_idx = self.overlap_length

        col_names = {'input_ids', 'token_type_ids', 'other_token_type_ids', 'turn_overlap', 'time_until_ts',
                     'time_until_other_ts'}

        # self.dataset iterates data of each dataset

        def split_dialog(channel, min_length=-1, overlap_length=self.overlap_length, keep_length=self.keep_length,
                         max_length=self.max_length):
            output = {}
            result = []

            tokens = channel['input_ids']
            types = channel['token_type_ids']
            other = channel['other_token_type_ids']
            overlap = channel['turn_overlap']
            speaker_ids = channel['speaker_ids']
            if 'time_until_ts' in channel:
                time_until_ts = channel['time_until_ts']
            else:
                time_until_ts = torch.ones_like(overlap)
            if 'time_until_other_ts' in channel:
                time_until_other_ts = channel['time_until_other_ts']
            else:
                time_until_other_ts = torch.ones_like(overlap)

            timings = channel['timings']
            if isinstance(timings, list):
                timings = torch.tensor(timings)

            start_idx = 0
            end_idx = overlap_length

            if min_length == -1:
                min_length = len(tokens)

            while end_idx <= min_length:
                # Split to appropriate lengths
                start_idx = end_idx - overlap_length
                # End either with a maximum length sequence or whatever is left
                end_idx = min(max_length + start_idx, min_length)

                if end_idx - start_idx < keep_length:
                    end_idx = overlap_length
                    break

                output['input_ids'] = tokens[start_idx:end_idx].clone(
                ).detach()
                output['token_type_ids'] = types[start_idx:end_idx].clone(
                ).detach()
                output['attention_mask'] = torch.ones(
                    end_idx - start_idx)
                output['timings'] = timings[start_idx:end_idx].clone(
                ).detach()
                output['other_token_type_ids'] = other[start_idx:end_idx].clone(
                ).detach()
                output['turn_overlap'] = overlap[start_idx:end_idx].clone().detach()
                output['time_until_ts'] = time_until_ts[start_idx:end_idx].clone().detach()
                output['time_until_other_ts'] = time_until_other_ts[start_idx:end_idx].clone().detach()
                output['speaker_ids'] = speaker_ids[start_idx:end_idx].clone().detach()

                output['conv_id'] = channel['conv_id']

                assert output['token_type_ids'].shape == output['input_ids'].shape, f"{types.shape} {tokens.shape}"
                assert output['input_ids'].shape[
                           0] <= 256, f"{output['input_ids'].shape}"

                yield copy.deepcopy(output)

            yield None

        for dialog in self.data:
            if pairwise:
                min_length = min(len(dialog['speakerA']['tokens']), len(dialog['speakerB']['tokens']))

                iterA = split_dialog(dialog['speakerA'], min_length=min_length)
                iterB = split_dialog(dialog['speakerB'], min_length=min_length)

                outA = next(iterA)
                outB = next(iterB)

                while outA is not None and outB is not None:
                    result.append({
                        'speakerA': outA,
                        'speakerB': outB,
                    })

                    outA = next(iterA)
                    outB = next(iterB)

            else:
                iterator = split_dialog(dialog)
                for out in iterator:
                    if out is None:
                        break

                    result.append(out)

        return result

    """
    Returns list of (start_idx,end_idx) for each speaker where the start and end indexes correspond to
    IPUs.

    Here, we simply update indexes to track the index of the start word and 
    the end word index, with respext to the whole dialogue, in each turn 
    dialog is list of dict with each word feature
    """

    def _get_ipus(self, dialog):
        ipus = []

        curr_ipu = {
            'start_idx': 0,
            'end_idx': 0,
            'word': '',
        }

        total_word_idx = 0
        for sentence_idx, sentence in enumerate(dialog):
            curr_ipu = {
                'start_idx': total_word_idx,
                'end_idx': total_word_idx,
                'word': "",
            }

            total_word_idx += len(sentence) - 1
            curr_ipu['end_idx'] = total_word_idx
            curr_ipu['word'] += " ".join(x['word'] for x in sentence)
            ipus.append(curr_ipu)

            total_word_idx += 1

        assert len(ipus) != 0, f"no IPU length {dialog[0]['conv_id']}"

        return ipus

    """
    Combines IPUs to form TSs (turnshifts) by combining consecutive IPUs
    Labels each turn with a type: NORMAL, INTERRUPT via ts['turn_type']
    Also, we find yield end of turns via ts['other_turn_type'] where the 
    end of the turn corresponds with the other speaker's INTERRUPT
    """
    def _get_ts(self, dialogA, ipuA, dialogB, ipuB):
        tsA = []
        tsB = []

        def handle_channel(dialog1, ts1, ipu1, curr_ts1, idx1, dialog2, ts2, ipu2, curr_ts2, idx2):
            # Check that
            # If these are both true than B can be added as A is definetly a new turn that requires a turn shift
            # or is_utt_continue2):
            overlap = None

            if len(curr_ts2['word']) != 0:
                ts2.append(curr_ts2)
                curr_ts2 = reset_ts(
                    ipu2[idx2]['start_idx'] if idx2 < len(ipu2) else -1)

            # Check if B overlaps with A so it starts prior to end of A by some threshold 
            if dialog2[ipu2[idx2]['start_idx']]['start'] + self.interruption_thresh < dialog1[ipu1[idx1]['end_idx']][
                'end']:
                # This case should not occur as overlaps should be preprocssed 
                # and inserted later
                if dialog2[ipu2[idx2]['end_idx']]['end'] < dialog1[ipu1[idx1]['end_idx']]['end']:
                    # A ----------------          |        -------------
                    # B       ------      ----    | -----    ------

                    # Speaker B's utterance could be a continuation of a previous turn if words have been added
                    # So it cannot be a bc so we only consider as an overlap so therefore an interruption
                    pass

                else:

                    # A ----------------     |     ---------
                    # B           ---------- |  ------   -------
                    curr_ts1['end_idx'] = ipu1[idx1]['end_idx']

                    # In this case when expanding curr_ts1 we have check if it is a backchannel 
                    # as if we are continuing it then it is no longer a backchannel
                    # and it is now a yield as B is interrupting
                    # NOTE may not be relevant anymore as BACKCHANNEL should not be present here
                    if curr_ts1['word'] != "" and curr_ts1['turn_type'] == TurnType.BACKCHANNEL:
                        curr_ts1['turn_type'] = TurnType.NORMAL
                    curr_ts1['other_turn_type'] = TurnType.YIELD

                    curr_ts1['word'] += (" " + ipu1[idx1]['word'])
                    idx1 += 1

                    # Automatically assign as INTERRUPT as B speaks during A's turn
                    # Account for case where B is continuing previous speech and retaining original information except
                    # for in the case of BC as continuing a BC is not possible
                    if curr_ts2['word'] == "":
                        curr_ts2['turn_type'] = TurnType.INTERRUPT
                    else:
                        # Update from overlap to interrupt as now turn ends after end of current speaker's turn
                        curr_ts2['turn_type'] = TurnType.INTERRUPT if curr_ts2['turn_type'] == TurnType.OVERLAP else \
                            curr_ts2['turn_type']
            else:
                # A --------              |        ----              |       ----------------
                # B           ----------  |  ------      ----------- |              ------      -----------
                # Easy case as no input from B yet and either B is speaking next after some silence
                # OR A has another utterance and then B speaks
                curr_ts1['end_idx'] = ipu1[idx1]['end_idx']
                if curr_ts1['word'] != "" and curr_ts1['turn_type'] == TurnType.BACKCHANNEL:
                    curr_ts1['turn_type'] = TurnType.NORMAL
                curr_ts1['word'] += (" " + ipu1[idx1]['word'])
                idx1 += 1

            return ts1, ipu1, curr_ts1, idx1, ts2, ipu2, curr_ts2, idx2

        def reset_ts(start_idx=0):
            return {
                'start_idx': start_idx,
                'end_idx': -1,
                'word': '',
                'turn_type': TurnType.NORMAL,
                'other_turn_type': TurnType.NORMAL,
            }

        curr_tsA = reset_ts()
        curr_tsB = reset_ts()

        dialogA = [word for sentence in dialogA for word in sentence]
        dialogB = [word for sentence in dialogB for word in sentence]

        i, j = 0, 0
        while i < len(ipuA) and j < len(ipuB):

            # Keep updating the turns one at a time from both speakers until complete
            if dialogA[ipuA[i]['start_idx']]['start'] < dialogB[ipuB[j]['start_idx']]['start']:
                tsA, ipuA, curr_tsA, i, tsB, ipuB, curr_tsB, j = handle_channel(
                    dialogA, tsA, ipuA, curr_tsA, i, dialogB, tsB, ipuB, curr_tsB, j)
            else:
                tsB, ipuB, curr_tsB, j, tsA, ipuA, curr_tsA, i = handle_channel(
                    dialogB, tsB, ipuB, curr_tsB, j, dialogA, tsA, ipuA, curr_tsA, i, )

        if curr_tsA['word'] != "":
            tsA.append(curr_tsA)
            curr_tsA = reset_ts(curr_tsA['end_idx'])
        if curr_tsB['word'] != "":
            tsB.append(curr_tsB)
            curr_tsB = reset_ts(curr_tsB['end_idx'])

        for x in range(i, len(ipuA)):
            curr_tsA['word'] += (" " + ipuA[x]['word'])
            curr_tsA['end_idx'] = ipuA[i]['end_idx']
            curr_tsA['turn_type'] = TurnType.NORMAL

        for x in range(j, len(ipuB)):
            curr_tsB['word'] += (" " + ipuB[x]['word'])
            curr_tsB['end_idx'] = ipuB[j]['end_idx']
            curr_tsB['turn_type'] = TurnType.NORMAL

        if curr_tsA['word'] != "":
            tsA.append(curr_tsA)
        if curr_tsB['word'] != "":
            tsB.append(curr_tsB)

        return tsA, tsB

    def _insert_bc(self, dialogA, tsA, bcA, dialogB, tsB, bcB):
        """
        Insert <bc> as a part of the dialogA at the correct timings location, and then add <bc> as a turn
        within `ts` with the appropriate turn shift type

        Will also have to update the `start_idx` and `end_idx` of `ts` elements. Can just do this after everything else is done

        We assume backchannels are part of the other's utterance so just add when it is prior to the next turn
        """

        def _insert_bc_channel(dialog, ts, bc, use_bc_token=False):
            curr_bc_idx = 0
            total_word_idx = 0
            curr_ts_idx = 0

            new_dialog = []
            new_ts = []
            turn_idx = 0
            turn = None

            while turn_idx < len(dialog):
                turn = dialog[turn_idx]
                if curr_bc_idx < len(bc) and turn[0]['start'] > bc[curr_bc_idx][0]['start']:
                    end_word_idx = total_word_idx + (0 if use_bc_token else len(bc[curr_bc_idx]) - 1)
                    bc_word = '<bc>' if use_bc_token else " ".join(x['word'] for x in bc[curr_bc_idx])

                    new_ts.append({
                        'start_idx': total_word_idx,
                        'end_idx': end_word_idx,
                        'word': bc_word,
                        'turn_type': TurnType.BACKCHANNEL
                    })

                    if self.include_bc_token:
                        bc[curr_bc_idx][0]['word'] = bc_word
                        bc[curr_bc_idx][0]['end'] = bc[curr_bc_idx][-1]['end']
                        bc[curr_bc_idx][0]['tokens'] = self.tokenizer.convert_tokens_to_ids('<bc>')
                        bc[curr_bc_idx] = bc[curr_bc_idx][:1]

                    new_dialog.append(bc[curr_bc_idx])

                    total_word_idx = end_word_idx + 1
                    curr_bc_idx += 1
                    continue

                new_dialog.append(turn)

                turn_type = ts[curr_ts_idx]['turn_type']
                end_word_idx = total_word_idx + (0 if use_bc_token and turn_type == TurnType.BACKCHANNEL else len(turn) - 1)
                bc_word = '<bc>'
                word = " ".join(x['word'] for x in turn)
                new_ts.append({
                    'start_idx': total_word_idx,
                    'end_idx': end_word_idx,
                    'word': bc_word if turn_type == TurnType.BACKCHANNEL else word,
                    'turn_type': turn_type
                })
                total_word_idx = end_word_idx + 1
                curr_ts_idx += 1
                turn_idx += 1

            while curr_bc_idx < len(bc):
                new_dialog.append(bc[curr_bc_idx])

                bc_word = '<bc>' if use_bc_token else " ".join(x['word'] for x in turn)
                end_word_idx = total_word_idx + (0 if use_bc_token else len(bc[curr_bc_idx]) - 1)
                new_ts.append({
                    'start_idx': total_word_idx,
                    'end_idx': end_word_idx,
                    'word': bc_word,
                    'turn_type': TurnType.BACKCHANNEL
                })
                total_word_idx = end_word_idx + 1
                curr_bc_idx += 1

            assert curr_bc_idx == len(bc), f"{curr_bc_idx} {len(bc)}"

            return new_dialog, new_ts, bc

        if len(dialogB) != len(tsB):
            for idx in range(min(len(dialogB), len(tsB))):
                print(f"dialogB: {dialogB[idx]}")
                print(f"\ttsB: {tsB[idx]}\n\n")

        assert len(dialogA) == len(tsA), f"{len(dialogA)} {len(tsA)}"
        assert len(dialogB) == len(tsB), f"{len(dialogB)} {len(tsB)}"

        old_len = len(dialogA)
        dialogA, tsA, bcA = _insert_bc_channel(dialogA, tsA, bcA, use_bc_token=self.include_bc_token)
        assert old_len + len(bcA) == len(dialogA), f"{old_len}, {len(bcA)}, {len(dialogA)}"

        old_len = len(dialogB)
        dialogB, tsB, bcB = _insert_bc_channel(dialogB, tsB, bcB, use_bc_token=self.include_bc_token)
        assert old_len + len(bcB) == len(dialogB), f"{old_len}, {len(bcB)}, {len(dialogB)}"

        assert len(dialogA) == len(tsA)
        assert len(dialogB) == len(tsB)

        assert all(dialogA[idx][0]['start'] < dialogA[idx + 1][0]['start'] for idx in range(len(dialogA) - 1))
        assert all(dialogB[idx][0]['start'] < dialogB[idx + 1][0]['start'] for idx in range(len(dialogB) - 1))

        return dialogA, tsA, bcA, dialogB, tsB, bcB

    def _insert_overlap(self, dialogA, tsA, overlapA, dialogB, tsB, overlapB, allow_joins=False):
        """
        Insert overlap either as a standalone overlap . This can be on
        either side so joining onto prior or post utterance
        A   --------   ----       |  ----------      ----   ----------
        B            -----------  |             ----------

        If joining on prior than B will now be interrupting A, if joining on post then A is interrupting B
        If no join than A is simply an overlap

        Join on Prior when: starts less than 1s from the end of A's utterance
        Join on Post when: overlap ends less than 1s from the start of A's next utterance
        All of the above is conditioned on whether start/end is closer to start/end of that parent utterance so only
        consider prior when start of overlap is closer to start of parent than the end of overlap is closer to end of parent

        If the conditions fail then we can consider the case as a pure overlap
        
        Note: allow_joins has to be set to consider the case where we can join overlaps
            Therefore, ignore this case as it introduced unnecessary complexity
            that could be solved when initially reading from the datasets
        """

        def _insert_overlap_channel(dialog, ts, overlap, other_dialog, pre_thresh=1, post_thresh=1, allow_joins=False):
            curr_overlap_idx = 0
            curr_turn_idx = 0

            # Each element will be (idx, OVERLAP_TYPE, TURN_TYPE)
            # 0 -> Overlap
            # 1 -> Prior Interruption
            # 2 -> Post Interruption
            overlapped_idx = []
            overlapper_idx = []

            other_turn_idx = 0
            other_turn = None
            while other_turn_idx < len(other_dialog):
                other_turn = other_dialog[other_turn_idx]
                if len(overlap) <= curr_overlap_idx:
                    break

                # We want to find the member of other dialog which overlaps the current overlap
                if other_turn[0]['start'] > overlap[curr_overlap_idx][0]['start']:
                    other_turn_idx += 1
                    continue
                if other_turn[-1]['end'] < overlap[curr_overlap_idx][-1]['end']:
                    other_turn_idx += 1
                    continue

                # Closer to start than to the end so consider prior case
                if overlap[curr_overlap_idx][0]['start'] - other_turn[0]['start'] < other_turn[-1]['end'] - \
                        overlap[curr_overlap_idx][-1]['end']:
                    prior_turn_idx = get_turn(dialog, compare=(
                        overlap[curr_overlap_idx][0]['start'], overlap[curr_overlap_idx][-1]['end']), when="prior")
                    turn = dialog[prior_turn_idx]
                    if allow_joins and prior_turn_idx > 0 and overlap[curr_overlap_idx][0]['start'] - turn[-1][
                        'end'] < pre_thresh:
                        # Join onto prior
                        turn.extend(overlap[curr_overlap_idx])
                        overlapped_idx.append(prior_turn_idx)
                        overlapper_idx.append(other_turn_idx)

                        if len(overlapped_idx) > 1 and prior_turn_idx == overlapped_idx[-2]:
                            overlapped_idx = overlapped_idx[:-1]
                            overlapper_idx = overlapper_idx[:-1]

                    else:
                        dialog.insert(prior_turn_idx + 1, overlap[curr_overlap_idx])

                        if len(overlapped_idx) == 0 or prior_turn_idx + 1 != overlapped_idx[-1]:
                            overlapped_idx.append(prior_turn_idx + 1)
                            overlapper_idx.append(other_turn_idx)
                else:
                    post_turn_idx = get_turn(dialog, compare=(
                        overlap[curr_overlap_idx][0]['start'], overlap[curr_overlap_idx][-1]['end']), when="post")
                    turn = dialog[post_turn_idx] if post_turn_idx < len(dialog) else None

                    # Handle case where turn joins onto the next turn
                    if allow_joins and turn is not None and turn[0]['start'] - overlap[curr_overlap_idx][-1][
                        'end'] < post_thresh:
                        new_turn = overlap[curr_overlap_idx]
                        new_turn.extend(turn)
                        dialog[post_turn_idx] = new_turn

                        overlapped_idx.append(post_turn_idx)
                        overlapper_idx.append(other_turn_idx)

                        # Occurs where index of next overlap matches current (two overlaps joining)
                        # Decide which values to copy over
                        if len(overlapped_idx) > 1 and post_turn_idx == overlapped_idx[-2]:
                            overlapped_idx = overlapped_idx[:-1]
                            overlapper_idx = overlapper_idx[:-1]
                    else:
                        dialog.insert(post_turn_idx, overlap[curr_overlap_idx])

                        if len(overlapped_idx) == 0 or post_turn_idx != overlapped_idx[-1]:
                            overlapped_idx.append(post_turn_idx)
                            overlapper_idx.append(other_turn_idx)

                curr_overlap_idx += 1

            while curr_overlap_idx < len(overlap):
                curr_overlap_idx += 1
                pass

            assert all(overlapped_idx[idx] != overlapped_idx[idx + 1] for idx in
                       range(len(overlapped_idx) - 1)), overlapped_idx
            return dialog, overlapped_idx, overlapper_idx

        def get_turn(dialog, compare=(0, 1), when="prior"):
            for idx, utt in enumerate(dialog):
                if when == "prior":
                    if utt[0]['start'] > compare[0]:
                        return idx - 1

                if when == "post":
                    if utt[0]['start'] > compare[0]:
                        return idx

            if when == "prior":
                return len(dialog) - 1

            return len(dialog)

        def update_ts(dialog, ts, curr_overlaps, other_overlaps, other_dialog, allow_joins=True):
            total_word_idx = 0
            curr_overlap_idx = 0
            other_overlap_idx = 0
            other_turn_idx = 0
            curr_ts_idx = 0

            new_ts = []
            other_update_ts = []
            for turn_idx, turn in enumerate(dialog):
                while other_overlap_idx < len(other_overlaps) and other_turn_idx != other_overlaps[other_overlap_idx]:
                    other_turn_idx += 1

                adding_new = False
                # Turn is modified from previous
                if curr_overlap_idx < len(curr_overlaps) and turn_idx == curr_overlaps[curr_overlap_idx]:
                    # Check if still completely enveloped by other turn if joined past start or end of the other turn
                    if other_dialog[other_turn_idx][0]['start'] <= dialog[turn_idx][0]['start']:
                        if not allow_joins or dialog[turn_idx][-1]['end'] <= other_dialog[other_turn_idx][-1]['end']:
                            turn_type = TurnType.OVERLAP
                        elif dialog[turn_idx][0]['start'] < other_dialog[other_turn_idx][-1]['end']:
                            # Ensures that an interruption occurs only dialog ends within overlap portion as it ain't an overlap

                            # New overlap ends after the overlapping dialog so now current dialog is interrupting
                            # the other's dialog
                            turn_type = TurnType.INTERRUPT
                        else:
                            adding_new = True
                    elif other_dialog[other_turn_idx][0]['start'] < dialog[turn_idx][-1]['end']:
                        # Now a prior join that means that the other speaker interrupted the utterance
                        # So whatever was previous
                        turn_type = ts[curr_ts_idx]['turn_type']
                        # Also have to add to other_ts
                        # Keep track for now and add as a post step
                        other_update_ts.append((other_turn_idx, other_dialog[other_turn_idx]))

                    other_overlap_idx += 1
                    end_word_idx = total_word_idx + len(turn) - 1
                    new_ts.append({
                        'start_idx': total_word_idx,
                        'end_idx': end_word_idx,
                        'word': " ".join(x['word'] for x in turn),
                        'turn_type': turn_type
                    })

                    if turn_type != TurnType.OVERLAP and not adding_new:
                        #  Turn was normally added that exists in ts
                        curr_ts_idx += 1

                    total_word_idx = end_word_idx + 1
                    curr_overlap_idx += 1

                    continue

                if curr_ts_idx >= len(ts):
                    print(dialog[0][0]['conv_id'])
                    pass

                turn_type = ts[curr_ts_idx]['turn_type']

                end_word_idx = total_word_idx + len(turn) - 1
                new_ts.append({
                    'start_idx': total_word_idx,
                    'end_idx': end_word_idx,
                    'word': " ".join(x['word'] for x in turn),
                    'turn_type': turn_type
                })

                total_word_idx = end_word_idx + 1
                curr_ts_idx += 1

            return new_ts, other_update_ts

        def update_int(dialog, ts, others):
            for other in others:
                ts[other[0]]['turn_type'] = TurnType.INTERRUPT
            return ts

        if dialogA[0][0]['conv_id'] == 'sw2152A-ms98-a-0002':
            pass

        dialogA, overlappedA, overlapperA = _insert_overlap_channel(dialogA, tsA, overlapA, dialogB,
                                                                    allow_joins=allow_joins)
        tsA, other_interruptionA = update_ts(dialogA, tsA, overlappedA, overlapperA, dialogB, allow_joins=allow_joins)

        dialogB, overlappedB, overlapperB = _insert_overlap_channel(dialogB, tsB, overlapB, dialogA,
                                                                    allow_joins=allow_joins)
        tsB, other_interruptionB = update_ts(dialogB, tsB, overlappedB, overlapperB, dialogA, allow_joins=allow_joins)

        assert len(dialogA) == len(tsA), f"{len(dialogA)} {len(tsA)}"
        assert len(dialogB) == len(tsB), f"{len(dialogB)} {len(tsB)}"

        tsA = update_int(dialogA, tsA, other_interruptionB)
        tsB = update_int(dialogB, tsB, other_interruptionA)
        return dialogA, tsA, overlapA, dialogB, tsB, overlapB

    def _add_special_turn_types(self, dialogA, tsA, dialogB, tsB, add_interruption_idx=True):
        """
        Adds special turn type: yield; to the previous turn if an interrupt 
        occurs in the current speaker's channel. Alternatively, a yield is 
        added if an overlap is close enough to the end of the speaker's turn
        """
        def get_interruption_time(curr_dialog, next_dialog, curr_ts, next_ts):
            """
            Finds the index of curr_dialog which is definitely after the beginning of the
            interruption by next_ts
            We will use in combination with overlap_mask after adding <emp> alignment to give the final 
            locations
            """
            if not add_interruption_idx:
                return -1

            if next_ts['start_idx'] >= len(curr_dialog):
                pass

            next_start = next_dialog[next_ts['start_idx']]['start']
            for idx in range(curr_ts['start_idx'], curr_ts['end_idx'] + 1):
                if curr_dialog[idx]['start'] > next_start:
                    return curr_dialog[idx]['start']

            return curr_dialog[curr_ts['end_idx']]['start'] - 0.00001

        i, j = 0, 0

        dialogA = [word for sentence in dialogA for word in sentence]
        dialogB = [word for sentence in dialogB for word in sentence]

        # 'end_idx' of last full turn
        non_bc_turnsA = [idx for idx, x in enumerate(tsA) if
                         x['turn_type'] in [TurnType.NORMAL, TurnType.INTERRUPT, TurnType.OVERLAP]] + [-1]
        non_bc_turnsB = [idx for idx, x in enumerate(tsB) if
                         x['turn_type'] in [TurnType.NORMAL, TurnType.INTERRUPT, TurnType.OVERLAP]] + [-1]

        normal_turnsA = [idx for idx, x in enumerate(tsA) if
                         x['turn_type'] in [TurnType.NORMAL, TurnType.INTERRUPT]] + [-1]
        normal_turnsB = [idx for idx, x in enumerate(tsA) if
                         x['turn_type'] in [TurnType.NORMAL, TurnType.INTERRUPT]] + [-1]
        normal_turn_i = 0
        normal_turn_j = 0

        while i < len(non_bc_turnsA) and j < len(non_bc_turnsB):
            """
            If current utterance from A is an interruption then add turn type 
            to B that shows that they're end of turn is a yield.
            Also store the location of the interruption
            (TYPE, IDX)

            Also a yield type is added if an overlap pcurs within
            self.yield_overlap_thresh of the end of the turn
            """
            prev_ts_idxA = non_bc_turnsA[i - 1] if i > 0 else None
            ts_idxA = non_bc_turnsA[i]
            prev_ts_idxB = non_bc_turnsB[j - 1] if j > 0 else None
            ts_idxB = non_bc_turnsB[j]

            if dialogA[tsA[ts_idxA]['start_idx']]['start'] < dialogB[tsB[ts_idxB]['start_idx']]['start']:
                if 'other_turn_type' not in tsA[ts_idxA] or not isinstance(tsA[ts_idxA]['other_turn_type'], tuple):
                    tsA[ts_idxA]['other_turn_type'] = (TurnType.NORMAL, -1)

                tsA[ts_idxA]['turn_length'] = dialogA[tsA[ts_idxA]['end_idx']]['end'] - \
                                              dialogA[tsA[ts_idxA]['start_idx']]['start']

                if prev_ts_idxB is not None and tsA[ts_idxA]['turn_type'] == TurnType.INTERRUPT:
                    int_idx = get_interruption_time(dialogB, dialogA, tsB[prev_ts_idxB], tsA[ts_idxA])
                    tsB[prev_ts_idxB]['other_turn_type'] = (TurnType.YIELD, int_idx)
                elif ts_idxB != -1 and tsB[ts_idxB]['turn_type'] in {TurnType.OVERLAP, TurnType.BACKCHANNEL}:
                    int_idx = get_interruption_time(dialogA, dialogB, tsA[ts_idxA], tsB[ts_idxB])
                    if abs(dialogA[tsA[ts_idxA]['end_idx']]['end'] - dialogB[tsB[ts_idxB]['start_idx']][
                        'start']) < self.yield_overlap_thresh:
                        if 'other_turn_type' not in tsA[ts_idxA] or tsA[ts_idxA]['other_turn_type'][1] == -1:
                            tsA[ts_idxA]['other_turn_type'] = (TurnType.YIELD, int_idx)
                    j += 1
                    continue

                if ts_idxA != -1 and tsA[ts_idxA]['turn_type'] in {TurnType.NORMAL, TurnType.INTERRUPT}\
                        and tsB[ts_idxB]['turn_type'] in {TurnType.NORMAL, TurnType.INTERRUPT}:
                    tsA[ts_idxA]['turn_overlap'] = dialogB[tsB[ts_idxB]['start_idx']]['start'] - \
                                                   dialogA[tsA[ts_idxA]['end_idx']]['end']
                else:
                    tsA[ts_idxA]['turn_overlap'] = 0
                i += 1
            else:
                if 'other_turn_type' not in tsB[ts_idxB] or not isinstance(tsB[ts_idxB]['other_turn_type'], tuple):
                    tsB[ts_idxB]['other_turn_type'] = (TurnType.NORMAL, -1)

                tsB[ts_idxB]['turn_length'] = dialogB[tsB[ts_idxB]['end_idx']]['end'] - \
                                              dialogB[tsB[ts_idxB]['start_idx']]['start']

                if prev_ts_idxA is not None and tsB[ts_idxB]['turn_type'] == TurnType.INTERRUPT:
                    int_idx = get_interruption_time(dialogA, dialogB, tsA[prev_ts_idxA], tsB[ts_idxB])
                    tsA[prev_ts_idxA]['other_turn_type'] = (TurnType.YIELD, int_idx)
                elif ts_idxA != -1 and tsA[ts_idxA]['turn_type'] in {TurnType.OVERLAP, TurnType.BACKCHANNEL}:
                    int_idx = get_interruption_time(dialogA, dialogB, tsA[ts_idxA], tsB[ts_idxB])
                    if abs(dialogB[tsB[ts_idxB]['end_idx']]['end'] - dialogA[tsA[ts_idxA]['start_idx']][
                        'start']) < self.yield_overlap_thresh:
                        if 'other_turn_type' not in tsB[ts_idxB] or tsB[ts_idxB]['other_turn_type'][1] == -1:
                            tsB[ts_idxB]['other_turn_type'] = (TurnType.YIELD, int_idx)
                    i += 1
                    continue

                if ts_idxB != -1 and tsA[ts_idxA]['turn_type'] in {TurnType.NORMAL, TurnType.INTERRUPT} and \
                        tsB[ts_idxB]['turn_type'] in {TurnType.NORMAL, TurnType.INTERRUPT}:
                    tsB[ts_idxB]['turn_overlap'] = dialogA[tsA[ts_idxA]['start_idx']]['start'] - \
                                                   dialogB[tsB[ts_idxB]['end_idx']]['end']
                else:
                    tsB[ts_idxB]['turn_overlap'] = 0
                j += 1


        for x in range(i, len(non_bc_turnsA)):
            ts = non_bc_turnsA[x]
            tsA[ts]['other_turn_type'] = (TurnType.NORMAL, -1)
            tsA[ts]['turn_overlap'] = 0

        for x in range(j, len(non_bc_turnsB)):
            ts = non_bc_turnsB[x]
            tsB[ts]['other_turn_type'] = (TurnType.NORMAL, -1)
            tsB[ts]['turn_overlap'] = 0

        for turn_shifts in [tsA, tsB]:
            for ts in turn_shifts:
                if 'other_turn_type' not in ts:
                    ts['other_turn_type'] = (TurnType.NORMAL, -1)
                if 'turn_overlap' not in ts:
                    ts['turn_overlap'] = 0

        if not add_interruption_idx:
            ts['other_turn_type'] = [x[0] if isinstance(x, tuple) else x for x in ts['other_turn_type']]

        assert all('other_turn_type' in x for x in tsA), [x for x in tsA if "other_turn_type" not in x]
        assert all('other_turn_type' in x for x in tsB), [x for x in tsB if "other_turn_type" not in x]
        assert all(isinstance(x['other_turn_type'], tuple) for x in tsA), [(idx,x['turn_type'], x['turn_overlap']) for idx,x in enumerate(tsA) if not isinstance(x["other_turn_type"],tuple)] + [len(tsA)]
        assert all(isinstance(x['other_turn_type'], tuple) for x in tsB), [(idx,x['turn_type'], x['turn_overlap']) for idx,x in enumerate(tsB) if not isinstance(x["other_turn_type"],tuple)] + [len(tsB)]

        assert all('turn_overlap' in x for x in tsA), [x for x in tsA if "turn_overlap" not in x]
        assert all('turn_overlap' in x for x in tsB), [x for x in tsB if "turn_overlap" not in x]
        return tsA, tsB

    def _add_turn_lengths(self, dialogA, tsA, dialogB, tsB):
        dialogA = [word for sentence in dialogA for word in sentence]
        dialogB = [word for sentence in dialogB for word in sentence]

        for ts in tsA:
            ts['turn_length'] = dialogA[ts['end_idx']]['end'] - dialogA[ts['start_idx']]['start']

        for ts in tsB:
            ts['turn_length'] = dialogB[ts['end_idx']]['end'] - dialogB[ts['start_idx']]['start']

        return tsA, tsB

    def combine_speaker_channels(self):
        result = []

        def remove_empty_tensor(data, dim=0):
            if dim == 2:
                return [x[0, 0, ...] for x in data if x.shape[0] > 0]

            return [x for x in data if x.shape[0] > 0]

        emp_token_id = self.tokenizer.convert_tokens_to_ids('<emp>')
        sot_token_idA = self.tokenizer.convert_tokens_to_ids('<sot>')
        sot_token_idB = self.tokenizer.convert_tokens_to_ids('<sot>')
        eot_token_idA = self.tokenizer.convert_tokens_to_ids('<eot>')
        eot_token_idB = self.tokenizer.convert_tokens_to_ids('<eot>')
        if self.individual_ts:
            eot_token_idA = self.tokenizer.convert_tokens_to_ids('<speakerA>')
            eot_token_idB = self.tokenizer.convert_tokens_to_ids('<speakerB>')

        for dialog in self.data:
            output = {}

            tokensA = dialog['speakerA']['input_ids']
            maskA = torch.logical_and(tokensA != emp_token_id, tokensA != sot_token_idA)
            tokensA = tokensA[maskA]

            eot_maskA = (tokensA == eot_token_idA).nonzero(as_tuple=True)[-1] + 1
            tokensA = remove_empty_tensor(tokensA.tensor_split(eot_maskA))
            typesA = remove_empty_tensor(dialog['speakerA']['token_type_ids'][maskA].tensor_split(eot_maskA))
            otherA = remove_empty_tensor(dialog['speakerA']['other_token_type_ids'][maskA].tensor_split(eot_maskA))
            overlapA = remove_empty_tensor(dialog['speakerA']['turn_overlap'][maskA].tensor_split(eot_maskA))
            time_until_tsA = remove_empty_tensor(dialog['speakerA']['time_until_ts'][maskA].tensor_split(eot_maskA))
            time_until_other_tsA = remove_empty_tensor(
                dialog['speakerA']['time_until_other_ts'][maskA].tensor_split(eot_maskA))
            all_timingsA = remove_empty_tensor(
                torch.tensor(dialog['speakerA']['timings'])[maskA].tensor_split(eot_maskA))
            timingsA = remove_empty_tensor(torch.tensor(dialog['speakerA']['timings'])[maskA].tensor_split(eot_maskA),
                                           dim=2)
            speaker_idsA = remove_empty_tensor(dialog['speakerA']['speaker_ids'][maskA].tensor_split(eot_maskA))

            tokensB = dialog['speakerB']['input_ids']
            maskB = torch.logical_and(tokensB != emp_token_id, tokensB != sot_token_idB)
            tokensB = tokensB[maskB]

            eot_maskB = (tokensB == eot_token_idB).nonzero(as_tuple=True)[-1] + 1
            tokensB = remove_empty_tensor(tokensB.tensor_split(eot_maskB))
            typesB = remove_empty_tensor(dialog['speakerB']['token_type_ids'][maskB].tensor_split(eot_maskB))
            otherB = remove_empty_tensor(dialog['speakerB']['other_token_type_ids'][maskB].tensor_split(eot_maskB))
            overlapB = remove_empty_tensor(dialog['speakerB']['turn_overlap'][maskB].tensor_split(eot_maskB))
            time_until_tsB = remove_empty_tensor(dialog['speakerB']['time_until_ts'][maskB].tensor_split(eot_maskB))
            time_until_other_tsB = remove_empty_tensor(
                dialog['speakerB']['time_until_other_ts'][maskB].tensor_split(eot_maskB))
            all_timingsB = remove_empty_tensor(
                torch.tensor(dialog['speakerB']['timings'])[maskB].tensor_split(eot_maskB))
            timingsB = remove_empty_tensor(torch.tensor(dialog['speakerB']['timings'])[maskB].tensor_split(eot_maskB),
                                           dim=2)
            speaker_idsB = remove_empty_tensor(dialog['speakerB']['speaker_ids'][maskB].tensor_split(eot_maskB))

            tokens = tokensA + tokensB
            types = typesA + typesB
            other = otherA + otherB
            overlap = overlapA + overlapB
            time_until_ts = time_until_tsA + time_until_tsB
            time_until_other_ts = time_until_other_tsA + time_until_other_tsB
            timings = torch.tensor(timingsA + timingsB)
            all_timings = all_timingsA + all_timingsB
            speaker_ids = speaker_idsA + speaker_idsB

            sorted_timings = timings.sort()
            sort_idx = sorted_timings.indices
            tokens = torch.cat([tokens[sort_idx[idx]] for idx in range(len(tokens))])
            types = torch.cat([types[sort_idx[idx]] for idx in range(len(types))])
            other = torch.cat([other[sort_idx[idx]] for idx in range(len(other))])
            overlap = torch.cat([overlap[sort_idx[idx]] for idx in range(len(overlap))])
            time_until_ts = torch.cat([time_until_ts[sort_idx[idx]] for idx in range(len(time_until_ts))])
            time_until_other_ts = torch.cat(
                [time_until_other_ts[sort_idx[idx]] for idx in range(len(time_until_other_ts))])
            timings = torch.cat([all_timings[sort_idx[idx]] for idx in range(len(all_timings))])
            speaker_ids = torch.cat([speaker_ids[sort_idx[idx]] for idx in range(len(speaker_ids))])

            output['input_ids'] = tokens
            output['token_type_ids'] = types
            output['other_token_type_ids'] = other
            output['turn_overlap'] = overlap
            output['timings'] = timings
            output['time_until_ts'] = time_until_ts
            output['time_until_other_ts'] = time_until_other_ts
            output['speaker_ids'] = speaker_ids
            output['conv_id'] = dialog['speakerA']['conv_id']

            result.append(output)

        return result

    def pp_item(self, conv_id):
        if self.combine_speaker:
            dialogs = [x for x in self.data if x['conv_id'] == conv_id]
        else:
            dialogs = [x for x in self.data if x['speakerA']['conv_id'] == conv_id]

        if len(dialogs) == 0:
            print(f"ERROR: conv_id {conv_id} not found")
            if self.combine_speaker:
                print([x['conv_id'] for x in self.data])
            else:
                print([x['speakerA']['conv_id'] for x in self.data])

            return

        for batch in dialogs:
            if not self.combine_speaker:
                input_idsA = batch['speakerA']['input_ids']
                input_idsB = batch['speakerB']['input_ids']

                timingsA = batch['speakerA']['timings']
                timingsB = batch['speakerB']['timings']

                typesA = batch['speakerA']['token_type_ids']
                typesB = batch['speakerB']['token_type_ids']

                otherA = batch['speakerA']['other_token_type_ids']
                otherB = batch['speakerB']['other_token_type_ids']

                overlapA = batch['speakerA']['turn_overlap']
                overlapB = batch['speakerB']['turn_overlap']

                speaker_idsA = batch['speakerA']['speaker_ids']
                speaker_idsB = batch['speakerB']['speaker_ids']

                start = 0
                offset = 5
                end = start + len(input_idsA)
                curr = [start, start + offset]

                while True:
                    _, _ = pp_pair_dialogs(tokenizer, input_idsA,
                                           timings=timingsA, curr=curr, token_types=typesA,
                                           others={"Yield Type": otherA, "Overlap": overlapA,"Speaker Type": speaker_idsA}, speaker='A')
                    curr, _ = pp_pair_dialogs(
                        tokenizer, input_idsB, timings=timingsB, curr=curr, token_types=typesB,
                        others={"Yield Type": otherB, "Overlap": overlapB,"Speaker Type": speaker_idsB}, speaker='B')
                    print()

                    if curr[0] > end:
                        break

                print("------------------------------------")
            else:
                end = len(batch['input_ids'])
                curr = [0, 5]

                types = batch['token_type_ids']
                other = batch['other_token_type_ids']
                overlap = batch['turn_overlap']
                timings = batch['timings']
                time_until_ts = batch['time_until_ts']
                speaker_ids = batch['speaker_ids']

                while True:
                    curr, _ = pp_single_dialogs(tokenizer, batch['input_ids'], curr,
                                                timings, others=[types, other, overlap, speaker_ids, time_until_ts])
                    print()
                    if curr[0] > end:
                        break

    def add_time_until_ts(self, dialog, turn_shifts):
        if isinstance(dialog[0], list):
            dialog = [word for sentence in dialog for word in sentence]

        for ts in turn_shifts:
            time_until_ts = [ts['turn_length']]
            time_until_other_ts = [ts['turn_length'] + ts['turn_overlap']]
            last_mid = dialog[ts['start_idx']]['start'] + (
                    dialog[ts['start_idx']]['end'] - dialog[ts['start_idx']]['start']) / 2

            for idx in range(ts['start_idx'] + 1, ts['end_idx'] + 1):
                mid = dialog[idx]['start'] + (dialog[idx]['end'] - dialog[idx]['start']) / 2
                new_time = round(time_until_ts[-1] - (mid - last_mid), 4)
                new_other_time = round(time_until_other_ts[-1] - (mid - last_mid), 4)

                time_until_ts.append(new_time)
                time_until_other_ts.append(new_other_time)
                last_mid = mid

            ts['time_until_ts'] = time_until_ts
            ts['time_until_other_ts'] = time_until_other_ts

            assert len(ts['time_until_ts']) == 1 + ts['end_idx'] - ts['start_idx']
            assert len(ts['time_until_other_ts']) == 1 + ts['end_idx'] - ts['start_idx']

        return turn_shifts

    def calculate_metrics(self, data, metric_ts, metric_other_ts, n1, n2):
        for timings in data:
            time_ts = torch.tensor(timings['time_until_ts'])
            time_other_ts = torch.tensor(timings['time_until_other_ts'])
            metric_ts, n1 = update_metric(metric_ts, n1, time_ts)
            metric_other_ts, n2 = update_metric(metric_other_ts, n2, time_other_ts)

        return metric_ts, metric_other_ts, n1, n2

    def normalize(self, metrics):
        for data in self.data:
            for col_name, metric in metrics.items():
                if col_name not in data['speakerA']:
                    continue

                data['speakerA'][col_name] = (data['speakerA'][col_name] - metric['mean']) / metric['std']
                data['speakerB'][col_name] = (data['speakerB'][col_name] - metric['mean']) / metric['std']

        return self.data

    def get_categories(self, columns=['time_until_ts', 'time_until_other_ts'], num_bins=20):
        bins = {}
        for col in columns:
            val = torch.tensor([])
            for data in self.data:
                for speaker in ['speakerA', 'speakerB']:
                    if col not in data[speaker]:
                        print(f"col {col} not found")
                        continue

                    data_adj = data[speaker][col]
                    while len(data_adj.shape) > 1:
                        data_adj = data_adj[0]

                    val = torch.cat((val, data_adj))

            if val.shape[0] == 0:
                break

            sorted_val = val.sort()
            percentiles = np.linspace(0, 100, num_bins + 1)
            bin_edges = np.percentile(sorted_val.values, percentiles)

            bins[col] = bin_edges

        return bins

    def categorize(self):
        for col_name, bin_edges in self.category_bins.items():
            col = torch.tensor([])
            for data in self.data:
                for speaker in ['speakerA', 'speakerB']:
                    data_adj = data[speaker][col_name]
                    while len(data_adj.shape) > 1:
                        data_adj = data_adj[0]

                    col = torch.cat((col, data_adj))

            # Assign observations to bins
            bin_indices = np.digitize(col, bin_edges, right=True)

            # Adjust bin indices to be 0-based
            bin_indices -= 1

            # Ensure bin indices are within the range [0, num_bins-1]
            bin_indices = np.clip(bin_indices, 0, len(bin_edges) - 1)
            bin_indices = bin_indices.tolist()

            curr_idx = 0
            for idx, data in enumerate(self.data):
                for speaker in ['speakerA', 'speakerB']:
                    length = data[speaker]['input_ids'].shape[-1]
                    data[speaker]['time_until_ts'] = torch.tensor([bin_indices[curr_idx: curr_idx + length]]).long()[0]
                    curr_idx += length

    def remove_bc_and_overlaps(self, dialog, ts):
        assert len(dialog) == len(ts)

        new_dialog = []
        new_ts = []
        for i in range(len(dialog)):
            if ts[i]['turn_type'] == TurnType.BACKCHANNEL:
                continue

            new_dialog.append(dialog[i])
            new_ts.append(ts[i])

        return new_dialog, new_ts

    def filter_special_tokens(self, tokens=['<ebc>', '<eint>']):
        ids = torch.tensor(self.tokenizer.convert_tokens_to_ids(tokens))
        emp = self.tokenizer.convert_tokens_to_ids('<emp>')
        new_data = []
        for data in self.data:
            maskA = torch.isin(data['speakerA']['input_ids'], ids)
            maskB = torch.isin(data['speakerB']['input_ids'], ids)

            for key in data['speakerA'].keys():
                if key not in ['input_ids', 'speaker_ids', 'token_type_ids']:
                    continue

                value = 0
                if key == 'input_ids':
                    value = emp

                data['speakerA'][key][maskA] = value
                data['speakerB'][key][maskB] = value


        return self.data

    def log_info(self):
        stats = {
            'ebc_tokens': 0,
            'eint_tokens': 0,
            'yield_tokens': 0,
            'eot_tokens': 0,
            'yield_turns': 0,
            'normal_turns': 0,
        }

        ebc_token_id, eint_token_id, yield_token_id, eot_token_id = self.tokenizer.convert_tokens_to_ids(['<ebc>', '<eint>', '<yield>', '<eot>'])

        for data in self.data:
            ddata = data
            if list(data.keys())[0] != 'speakerA':
                ddata = {'speakerA': data}

            for speaker, row in ddata.items():
                stats['ebc_tokens'] += (row['input_ids'] == ebc_token_id).sum()
                stats['eint_tokens'] += (row['input_ids'] == eint_token_id).sum()
                stats['eot_tokens'] += (row['input_ids'] == eot_token_id).sum()
                stats['yield_tokens'] += (row['input_ids'] == yield_token_id).sum()

        self.logger.info(self.__str__())
        for key, value in stats.items():
            self.logger.info(f"{key}: {value}")
        self.logger.info(f"loaded {self.get_save_load_path()} from cache")

    def remove_eot_emp_pair(self):
        """
        Remove the <emp> token that is paired with <eot>
        Previously had just for easy combination for TurnGPT.
        """
        eot_token_id = self.tokenizer.convert_tokens_to_ids('<eot>')
        emp_token_id = self.tokenizer.convert_tokens_to_ids('<emp>')
        for data in self.data:
            speakerA = data['speakerA']
            speakerB = data['speakerB']

            input_idsA = speakerA['input_ids']
            input_idsB = speakerB['input_ids']

            maskB = input_idsB == eot_token_id
            maskB_roll = torch.roll(maskB, shifts=1, dims=0)
            # maskB_roll = torch.cat((maskB_roll, maskB[-1:])) if maskB[-1] else maskB_roll
            maskB_roll[0] = False


            assert torch.all(input_idsA[maskB] == emp_token_id), f"{input_idsA[maskB]}"

            for key in speakerA.keys():
                if key in ['conv_id' ,'dialog', 'key', 'tokens']:
                    continue 

                speakerA[key] = speakerA[key][torch.logical_not(maskB)]
                speakerB[key] = speakerB[key][torch.logical_not(maskB_roll)]

            maskA = speakerA['input_ids'] == eot_token_id
            maskA_roll = torch.roll(maskA, shifts=1, dims=0)
            # maskA_roll = torch.cat((maskA_roll, maskA[-1:])) if maskA[-1] else maskA_roll
            maskA_roll[0] = False
            for key in speakerB.keys():
                if key in ['conv_id' ,'dialog', 'key', 'tokens']:
                    continue

                speakerB[key] = speakerB[key][torch.logical_not(maskA)]
                speakerA[key] = speakerA[key][torch.logical_not(maskA_roll)]


            maskA = data['speakerA']['input_ids'] == eot_token_id
            maskB = data['speakerB']['input_ids'] == eot_token_id


            start = 0
            offset = 5
            end = start + len(input_idsA)
            curr = [start, start + offset]

            while False:
                _, _ = pp_pair_dialogs(self.tokenizer, speakerA['input_ids'],
                                       timings=speakerA['timings'], curr=curr, token_types=speakerA['token_type_ids'],
                                       speaker='A')
                curr, _ = pp_pair_dialogs(
                    tokenizer, speakerB['input_ids'], timings=speakerB['timings'], curr=curr, token_types=speakerB['token_type_ids'],
                    speaker='B')
                print()

                if curr[0] > end:
                    break

                print("------------------------------------")

            maskA[-1] = 0
            maskB[-1] = 0
            assert torch.all(speakerA['input_ids'][maskB] != emp_token_id)
            assert torch.all(speakerB['input_ids'][maskA] != emp_token_id)

        return self.data




def update_metric(metric, n, array):
    prev_n = n
    length = array.size(-1)
    n += length
    new_mean = metric['mean'] + (array.mean(-1) - metric['mean']) * (
            length / n)

    ssd = metric['std'] * (prev_n - 1)
    ssd_new = array.var(
        -1, unbiased=False) * (length - 1)

    ssd_comb = ssd + ssd_new + \
               (array.mean(-1) -
                metric['mean']).pow(2) * prev_n * length / n
    metric['std'] = ssd_comb / (n - 1)
    metric['mean'] = new_mean

    return metric, n


if __name__ == "__main__":
    key = "4617"
    tokenizer = SpokenDialogTokenizer(tokens=[
        '<bc>', '<yield>', '<emp>', '<eot>', '<eint>', '<ebc>', '<speakerA>', '<speakerB>'
    ])
    ts = PairwiseGenerationDM(tokenizer=tokenizer, split="test", overwrite=True, remove_start_tokens=True,
                              include_yield_token=True,
                              combine_speaker=False, basic_mode=True, datasets=['switchboard'],
                              dev_mode=False, remove_overlaps=False, remove_backchannels=False, no_emp_tokens=False, savedata=False, include_overlap_token=True,
                              normalize_time=False,
                              include_bc_token=False, include_end_bc_token=True, individual_speaker_tokens=False, filter_bc_overlap_token=False,
                              parse_dialogs=[key]
                              )
    ts.prepare_data()

    tsb = PairwiseGenerationDM(tokenizer=tokenizer, overwrite=True, combine_speaker=False,
                               include_bc_token=True, include_overlap_token=True,
                               split="train", basic_mode=False, dev_mode=True, datasets=['switchboard'], savedata=True)
    #tsb.prepare_data()

    dl = DataLoader(ts, batch_size=4, collate_fn=ts.collate_fn)
    # dlb = DataLoader(tsb, batch_size=4, collate_fn=tsb.collate_fn)

    it = iter(dl)
    batch = next(it)


    # itb = iter(dlb)
    # batchB = next(itb)

    def show_input(batch):
        print(batch['speakerA']['conv_id'])
        input_idsA = batch['speakerA']['input_ids']
        input_idsB = batch['speakerB']['input_ids']

        timingsA = batch['speakerA']['timings']
        timingsB = batch['speakerB']['timings']

        typesA = batch['speakerA']['token_type_ids']
        typesB = batch['speakerB']['token_type_ids']

        otherA = batch['speakerA']['other_token_type_ids']
        otherB = batch['speakerB']['other_token_type_ids']

        overlapA = batch['speakerA']['turn_overlap']
        overlapB = batch['speakerB']['turn_overlap']

        start = 0
        offset = 5
        end = start + len(input_idsA)
        curr = [start, start + offset]

        while True:
            _, _ = pp_pair_dialogs(tokenizer, input_idsA,
                                   timings=timingsA, curr=curr, token_types=typesA, others=[otherA, overlapA],
                                   speaker='A')
            curr, _ = pp_pair_dialogs(
                tokenizer, input_idsB, timings=timingsB, curr=curr, token_types=typesB, others=[otherB, overlapB],
                speaker='B')
            print()

            if curr[0] > end:
                break

        print("------------------------------------")


    ts.pp_item(f'sw{key}A-ms98-a-0001')

    # show_input(batchspeaker_ids
    # show_input(batchB)
