from enum import IntEnum
from data.utils import pp_pair_dialogs
from pairwisegpt.utils import get_turn_after_bc
from pairwisegpt.train import build_parser, get_latest_model
from pairwisegpt.model import PairwiseGPT
from pairwise_generation_dm import TurnType, PairwiseGenerationDM
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch
import matplotlib.pyplot as plt
from time import sleep
from argparse import Namespace
import os
import logging
import json
import copy


class EvaluateType(IntEnum):
    OVERLAP_GENERATION = 1
    OVERLAP_EFFECT = 2


def get_abs_path(filepath):
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), filepath)


def get_overlap_utterance(model, dl, overlap_tokens={'eint', 'ebc'}, eot_items=None):
    """
    Finds utterances within dl that predict TRP well with threshold >0.2, that
    also contain overlaps or backchannels

    Identify locations of utterances with overlap or back channel and return list of keys:
        [(batch_number, number_in_batch, seq_index,
          turn_type (NORMAL/YIELD), distance_to_overlap)]
    """
    if eot_items is None:
        eot_items = torch.tensor([model.tokenizer.eos_token_id])
    if not isinstance(eot_items, torch.Tensor):
        eot_items = torch.tensor(eot_items)

    eot_overlapsA = []
    eot_overlapsB = []

    eint = model.tokenizer.convert_tokens_to_ids('<eint>')
    ebc = model.tokenizer.convert_tokens_to_ids('<ebc>')
    if 'eint' not in overlap_tokens:
        eint = ebc
    if 'ebc' not in overlap_tokens:
        ebc = eint

    for idx, batch in enumerate(dl):
        tokensA = batch['speakerA']['input_ids']
        tokensB = batch['speakerB']['input_ids']

        turn_typeA = batch['speakerA']['token_type_ids']
        turn_typeB = batch['speakerB']['token_type_ids']

        eot_typesA = batch['speakerA']['other_token_type_ids']
        eot_typesB = batch['speakerB']['other_token_type_ids']

        time_untilA = batch['speakerA']['time_until_ts']
        time_untilB = batch['speakerB']['time_until_ts']

        eotA = torch.isin(tokensA, eot_items)
        eotB = torch.isin(tokensB, eot_items)

        overlapA = torch.logical_or(torch.eq(
            tokensA, eint), torch.eq(tokensA, ebc))
        overlapB = torch.logical_or(torch.eq(
            tokensB, eint), torch.eq(tokensB, ebc))

        cumsum_overlapA = torch.cumsum(overlapA, dim=-1)
        cumsum_overlapB = torch.cumsum(overlapB, dim=-1)

        new_overlapA = cumsum_overlapA
        new_overlapB = cumsum_overlapB

        first_eotA = new_overlapB * eotA
        first_eotB = new_overlapA * eotB

        # Extract active turn regions
        for eot in first_eotA.nonzero():
            overlap = overlapB.nonzero(as_tuple=True)[0]
            if len(eot_overlapsA) == 0 or idx != eot_overlapsA[-1][0]:
                overlap = overlap[(overlap < eot[0])][-1]
                distance = eot[0] - overlap
                eot_overlapsA.append(
                    (idx, eot[0].item(), eot_typesA[eot[0]], distance))
                continue
            elif overlapB[eot_overlapsA[-1][1]:eot[0]].sum() > 0:
                overlap = overlap[(overlap > eot_overlapsA[-1][1])
                                  & (overlap < eot[0])][-1]
                distance = eot[0] - overlap

                eot_overlapsA.append(
                    (idx, eot[0].item(), eot_typesA[eot[0]], distance))
        for eot in first_eotB.nonzero():
            overlap = overlapA.nonzero(as_tuple=True)[0]
            if len(eot_overlapsB) == 0 or idx != eot_overlapsB[-1][0]:
                overlap = overlap[(overlap < eot[0])][-1]
                distance = eot[0] - overlap

                eot_overlapsB.append(
                    (idx, eot[0].item(), eot_typesB[eot[0]], distance))
                continue
            elif overlapA[eot_overlapsB[-1][1]:eot[0]].sum() > 0:
                overlap = overlap[(overlap > eot_overlapsB[-1][1])
                                  & (overlap < eot[0])][-1]
                distance = eot[0] - overlap

                eot_overlapsB.append(
                    (idx, eot[0].item(), eot_typesB[eot[0]], distance))

    return eot_overlapsA, eot_overlapsB


def predict_for_batch(model, sequence, masks=None, device="cuda"):
    speakerA = sequence['speakerA']
    speakerB = sequence['speakerB']

    emp_token_id = model.tokenizer.convert_tokens_to_ids('<emp>')

    tokensA = speakerA['input_ids'].to(device)
    tokensB = speakerB['input_ids'].to(device)
    attention_maskA = speakerA['attention_mask'].to(device)
    attention_maskB = speakerB['attention_mask'].to(device)
    token_type_idsA = speakerA['speaker_ids'].to(device)
    token_type_idsB = speakerB['speaker_ids'].to(device)

    if masks is not None:
        tokensA = torch.where(masks[0], tokensA, emp_token_id)
        tokensB = torch.where(masks[1], tokensB, emp_token_id)
        attention_maskA = torch.where(masks[0], attention_maskA, 1)
        attention_maskB = torch.where(masks[1], attention_maskB, 1)
        token_type_idsA = torch.where(masks[0], token_type_idsA, 0)
        token_type_idsB = torch.where(masks[1], token_type_idsB, 0)

    labelsA = torch.where(
        attention_maskA != 0,
        tokensA,
        torch.tensor(-100, device=device))
    labelsB = torch.where(
        attention_maskB != 0,
        tokensB,
        torch.tensor(-100, device=device))

    out = model.forward(
        input_idsA=tokensA,
        input_idsB=tokensB,
        attention_maskA=attention_maskA,
        attention_maskB=attention_maskB,
        token_type_idsA=token_type_idsA,
        token_type_idsB=token_type_idsB,
        labelsA=labelsA,
        labelsB=labelsB,
        output_attentions=True,
    )

    return out


def step_score(model, dl, batch_idx, seq_idx, type, distance, device="cuda", speaker='A', overlap_type=[], eot_items=[], last_batch_idx=-1, raw_out=None, remove_overlaps=False):
    sequence = dl[batch_idx]
    token_type_idsA = sequence['speakerA']['token_type_ids'].to(device)
    token_type_idsB = sequence['speakerB']['token_type_ids'].to(device)

    masks = None
    if remove_overlaps:
        masks = (torch.logical_not(torch.isin(token_type_idsA, overlap_type)),
                 torch.logical_not(torch.isin(token_type_idsB, overlap_type)))

    if last_batch_idx != batch_idx:
        raw_out = predict_for_batch(model, sequence, masks)

    out_logits = raw_out.logits[0]
    if speaker == 'B':
        out_logits = raw_out.logits[1]

    probs = out_logits.softmax(-1)

    eot_prob = max(probs[seq_idx - 1, x.long()] for x in eot_items)
    sum_self_attention = sum(
        [sum([y.detach().cpu().sum() for y in x]) for x in raw_out.attentions])
    sum_cross_attention = sum(
        [sum([y.detach().cpu().sum() for y in x]) for x in raw_out.cross_attentions])
    return eot_prob.detach().cpu(), batch_idx, seq_idx, raw_out, type, distance, sum_self_attention, sum_cross_attention


def get_scores_for_ts(model, eot, dl, eot_items=None, speaker='A', remove_overlaps=False, device="cuda:0"):
    if eot_items is None:
        eot_items = [model.tokenizer.eos_token_id]
    if not isinstance(eot_items, torch.Tensor):
        eot_items = torch.tensor(eot_items)

    overlap_type = torch.tensor(
        [TurnType.OVERLAP, TurnType.BACKCHANNEL], device=device)

    last_batch_idx1 = -1
    last_batch_idx2 = -1
    scores = []
    emp_token_id = model.tokenizer.convert_tokens_to_ids('<emp>')
    speaker_key = "speaker" + speaker

    diffs = []
    raw_out1 = None
    raw_out2 = None

    progress_bar = tqdm(eot)
    for batch_idx, seq_idx, type, distance in progress_bar:
        scores_model1 = step_score(model, dl, batch_idx, seq_idx, type, distance, speaker=speaker,
                                   eot_items=eot_items, last_batch_idx=last_batch_idx1, overlap_type=overlap_type, raw_out=raw_out1)
        scores_model2 = step_score(model, dl, batch_idx, seq_idx, type, distance, speaker=speaker, eot_items=eot_items,
                                   last_batch_idx=last_batch_idx2, overlap_type=overlap_type, remove_overlaps=True, raw_out=raw_out2)
        diff = (scores_model1[0] - scores_model2[0], scores_model1[0],  scores_model2[0], batch_idx, seq_idx,
                scores_model1[-4], scores_model1[-3], scores_model1[-2], scores_model2[-2], scores_model1[-1], scores_model2[-1])

        raw_out1 = scores_model1[3]
        raw_out2 = scores_model2[3]
        last_batch_idx1 = scores_model1[2]
        last_batch_idx2 = scores_model2[2]

        diffs.append(diff)

    return diffs


class Analyser:
    def __init__(self,
                 base_config,
                 compare_config=None,
                 device="cuda:0",
                 name="Full",
                 split='test',
                 **kwargs):
        """
        Should only load one model at a time to avoid memory issues
        """
        self.base_config = base_config
        self.new_config = compare_config

        self.device = device
        self.name = name

        self.split = split

    def __call__(self, run=EvaluateType.OVERLAP_EFFECT, *args, **kwargs):
        if run == EvaluateType.OVERLAP_EFFECT:
            return self.compare_overlap_caused_ts(**kwargs)
        elif run == EvaluateType.OVERLAP_GENERATION:
            return self.overlap_generation()

    def generate_plot(self, batch_index, seq_index, step=40, remove_overlap=False, overlap_tokens={'ebc', 'eint'}, device="cuda:0"):
        seq_index += 1

        overlap_types = []
        if 'eint' in overlap_tokens:
            overlap_types.append(TurnType.INTERRUPT)
        if 'ebc' in overlap_tokens:
            overlap_types.append(TurnType.BACKCHANNEL)

        special_tokens = [
            token_id for token_id in self.model1.tokenizer.special_tokens if token_id != "<emp>"]
        special_tokens.append("<eot>")

        tokens = [self.model1.tokenizer.convert_tokens_to_ids(
            token_id) for token_id in self.model1.tokenizer.special_tokens if token_id != "<emp>"]
        tokens.append(self.model1.tokenizer.convert_tokens_to_ids("<eot>"))
        masks = None

        if remove_overlap:
            overlap_type = torch.tensor(overlap_types, device=device)

            token_type_idsA = self.ds1[batch_index]['speakerA']['token_type_ids'].to(
                device)
            token_type_idsB = self.ds1[batch_index]['speakerB']['token_type_ids'].to(
                device)
            masks = (torch.logical_not(torch.isin(token_type_idsA, overlap_type)),
                     torch.logical_not(torch.isin(token_type_idsB, overlap_type)))

        out = predict_for_batch(
            self.model1, self.ds1[batch_index], masks=masks)
        logitsA = [out.logits[0][..., token] for token in tokens]
        logitsB = [out.logits[1][..., token] for token in tokens]

        input_idsA = self.ds1[batch_index]['speakerA']['input_ids'].detach(
        ).cpu()
        input_idsB = self.ds1[batch_index]['speakerB']['input_ids'].detach(
        ).cpu()
        if remove_overlap:
            input_idsA = torch.where(
                masks[0].cpu(), input_idsA, self.model1.tokenizer.convert_tokens_to_ids('<emp>'))
            input_idsB = torch.where(
                masks[1].cpu(), input_idsB, self.model1.tokenizer.convert_tokens_to_ids('<emp>'))
        input_idsA = input_idsA[seq_index-step:seq_index]
        input_idsB = input_idsB[seq_index-step:seq_index]

        probsA = [log.softmax(dim=-1).detach().cpu() for log in logitsA]
        probsB = [log.softmax(dim=-1).detach().cpu() for log in logitsB]

        pA = [x[seq_index-step:seq_index] for x in probsA]
        pB = [x[seq_index-step:seq_index] for x in probsB]

        fig, (_, ax) = plot_trp(trp=pA, text=self.model1.tokenizer.convert_ids_to_tokens(
            input_idsA), eos_token='<eot>', special_tokens=special_tokens)
        _, _ = plot_trp(trp=pB, text=self.model1.tokenizer.convert_ids_to_tokens(
            input_idsB), eos_token='<eot>', special_tokens=special_tokens, fig=fig, ax=ax, show=True)
        return fig

    def compare_overlap_caused_ts(self, eint=True, ebc=True, **kwargs):
        """
        Find samples that predict a TRP with an overlap or backchannel in the utterance then without

        Use function above to find TS with overlaps or backchannels
        Iterate with base model scores on those turn-shifts

        Iterate with model trained without bc or overlaps. Remove overlaps/backchannels from dataset.
        Do not use other one as we reuqire the same token indexes so just mask out

        Evaluate performance of model A and model B on the same TS with and without overlaps or backchannels
        Could also do so on the same model
        """
        overlap_tokens = set()
        if eint:
            overlap_tokens.add('eint')
        if ebc:
            overlap_tokens.add('ebc')

        self.model1, self.ds1, self.dl1 = load_model(self.base_config)

        overlap_tsA, overlap_tsB = get_overlap_utterance(
            self.model1, self.ds1, overlap_tokens=overlap_tokens)

        scoresA = get_scores_for_ts(self.model1, overlap_tsA, self.ds1)
        scoresB = get_scores_for_ts(
            self.model1, overlap_tsB, self.ds1, speaker='B')

        return scoresA, scoresB

    def turn_shift_projection(self):
        """
        Test how well the model is able to generate responses.
            - Generate to test the number of tokens until a <ts> token

        Over entire dataset, generate 1000 dialogues and check whether model is able to predict <ts>
        and calculate error compared to ground truth
        """
        return

    def sample_attention(self):
        pass

    def overlap_generation(self):
        """
        Test whether the model is able to generate overlaps and how close this is to
        the ground truth

        Over entire dataset, find instances of <emp> tokens (of at least 4 tokens) with dialog_type TurnType.NONE and generate dialog
        until the other speaker ends their turn, as this means no more chance for backchannel/overlap/interruption

        Count instances of generated utterances: backchannels and overlaps and interruptions
        If time: check error as compared to ground truth
        """

        def speaker_overlap_generation(speaker, other_speaker, speaker_id='A'):
            mask_emp = torch.logical_and(speaker['input_ids'] == emp_token_id,
                                         speaker['token_type_ids'] == TurnType.NONE)

            mask = torch.logical_and(
                mask_emp, other_speaker['input_ids'] != emp_token_id)
            mask_nz = mask.nonzero()

            prior_seq = []
            post_seq = []

            speaker1 = "speakerA" if speaker_id == 'A' else "speakerB"
            speaker2 = "speakerA" if speaker_id != 'A' else "speakerB"
            device = speaker['input_ids'].device

            self.model1.tokenizer.set_padding_side('left')

            for (batch_idx, seq_idx) in mask_nz:
                minimum = max(0, seq_idx - 50)
                maximum = min(256, seq_idx + 50)

                if seq_idx - minimum < 50:
                    continue

                prior_batches = torch.tensor(
                    [batch_idx.item() for _ in range(minimum, seq_idx + 1)], device=device)
                prior_idx = torch.tensor(
                    [idx for idx in range(minimum, seq_idx + 1)], device=device)

                post_batches = torch.tensor(
                    [batch_idx.item() for _ in range(seq_idx + 1, maximum)], device=device)
                post_idx = torch.tensor(
                    [idx for idx in range(seq_idx + 1, maximum)], device=device)

                prior_batch = {
                    speaker1: {
                        k: v[(prior_batches, prior_idx)].unsqueeze(0) for k, v in speaker.items() if isinstance(v, torch.Tensor)
                    },
                    speaker2: {
                        k: v[(prior_batches, prior_idx)].unsqueeze(0) for k, v in other_speaker.items() if
                        isinstance(v, torch.Tensor)
                    }
                }
                """
                prior_batch = {
                    speaker1: model.tokenizer.pad(prior_batch[speaker1], padding='max_length', max_length=256),
                    speaker2: model.tokenizer.pad(prior_batch[speaker2], padding='max_length', max_length=256),
                }
                prior_batch['speakerA']['conv_id'] = speaker['conv_id']
                prior_batch['speakerB']['conv_id'] = other_speaker['conv_id']

                for speaker_key in prior_batch.keys():
                    for k, v in prior_batch[speaker_key].items():
                        if isinstance(v, torch.Tensor):
                            if v.shape[-1] != 256:
                                prior_batch[speaker_key][k] = model.tokenizer.pad({'input_ids':
                                                              prior_batch[speaker_key][k]},
                                                                                   padding='max_length',
                                                                                   max_length=256)['input_ids']

                            assert prior_batch[speaker_key][
                                k].shape[-1] == 256, f"{v.shape}, {k} {v}"
                """

                generated = self.model1.generate(
                    context=prior_batch, stop_at_eos=True)
                self.print_dialogs(prior_batch, generated, post_batches)

            return prior_seq, post_seq

        self.model1, self.ds1, self.dl1 = load_model(self.base_config)

        progress_bar = tqdm(self.dl1, desc="Overlap Generation Batch     ")
        emp_token_id = self.model1.tokenizer.convert_tokens_to_ids('<emp>')

        for step, batch in enumerate(progress_bar):
            speakerA = batch['speakerA']
            speakerB = batch['speakerB']

            priorA, postA = speaker_overlap_generation(speakerA, speakerB)
            priorB, postB = speaker_overlap_generation(
                speakerB, speakerA, speaker_id='B')

        return

    def show_yields(self, step=20, name="Serialised"):
        self.model1, self.ds1, self.dl1 = load_model(self.base_config)
        progress_bar = tqdm(self.dl1, desc="Overlap Generation Batch     ")
        eot_token_id = self.model1.tokenizer.convert_tokens_to_ids('<eot>')
        post = 4

        count = 0

        for idx, batch in enumerate(progress_bar):
            speakerA = batch['speakerA']
            speakerB = batch['speakerB']

            input_idsA = speakerA['input_ids'].to(self.device)
            input_idsB = speakerB['input_ids'].to(self.device)

            eot_typesA = speakerA["other_token_type_ids"].to(
                self.device)
            eot_typesB = speakerB["other_token_type_ids"].to(
                self.device)
            timingsA = speakerA["timings"].to(
                self.device)
            timingsB = speakerB["timings"].to(
                self.device)

            out = predict_for_batch(self.model1, batch, device=self.device)

            yield_maskA = torch.logical_and(
                torch.eq(eot_typesA, TurnType.YIELD), torch.eq(input_idsA, eot_token_id))
            yield_maskB = torch.logical_and(
                torch.eq(eot_typesB, TurnType.YIELD), torch.eq(input_idsB, eot_token_id))

            input_ids = [input_idsA, input_idsB]
            yield_masks = [yield_maskA, yield_maskB]
            timings = [timingsA, timingsB]

            for speaker in range(2):
                other_speaker = (speaker + 1) % 2
                pred_curr = out.logits[speaker].clone()
                pred_other = out.logits[other_speaker].clone()
                yield_mask = yield_masks[speaker]

                input_idsA = input_ids[speaker]
                input_idsB = input_ids[other_speaker]

                prob_curr = torch.softmax(pred_curr, dim=-1)
                prob_other = torch.softmax(pred_other, dim=-1)

                # Find where full predicts trp but serialised does not at yield points
                prob_curr = prob_curr[yield_mask]
                prob_other = prob_other[yield_mask]

                yield_idxs = torch.nonzero(yield_mask, as_tuple=True)
                for idx, (batch_idx, seq_idx) in enumerate(zip(yield_idxs[0], yield_idxs[1])):
                    curr_logits_full = out.logits[speaker]
                    other_logits_full = out.logits[other_speaker]

                    curr_logits_full = curr_logits_full[batch_idx,
                                                        seq_idx-step: seq_idx+post, eot_token_id]
                    other_logits_full = other_logits_full[batch_idx,
                                                          seq_idx-step: seq_idx+post, eot_token_id]

                    curr_probs_full = torch.softmax(
                        curr_logits_full, dim=-1).detach().cpu()
                    other_probs_full = torch.softmax(
                        other_logits_full, dim=-1).detach().cpu()

                    curr_probs_full = torch.sqrt(curr_probs_full)
                    other_probs_full = torch.sqrt(other_probs_full)

                    # if curr_probs_full[batch]

                    # curr_probs_full = torch.where(torch.logical_and(curr_probs_full > 0.05, curr_probs_full < 0.2), curr_probs_full+0.1, curr_probs_full)
                    conv_id = speakerA['conv_id'][batch_idx]
                    timing = timings[speaker][batch_idx, seq_idx-1]

                    fig, (_, ax) = plot_trp([curr_probs_full],
                                            text=self.model1.tokenizer.convert_ids_to_tokens(
                                                input_idsA[batch_idx, seq_idx-step:seq_idx+post]),
                                            eos_token="<eot>", title=f"{conv_id}:{timing}")
                    plot_trp([other_probs_full], text=self.model1.tokenizer.convert_ids_to_tokens(
                        input_idsB[batch_idx, seq_idx - step:seq_idx + post]), eos_token="<eot>", fig=fig, ax=ax,
                        show=False)

                    if not os.path.exists(os.path.join("figures", conv_id)):
                        os.mkdir(os.path.join("figures", conv_id))

                    plt.savefig(os.path.join(
                        "figures", conv_id, f"{count}-{self.name}"))
                    plt.close()
                    count += 1
            count = 0

    def compare_bc_turns(self, thresh=0.5, step=20):
        """
        Compares performance of eot influenced by addition of bc in utterance like compare_yields
        Find where prediction fails without an inserted bc
        """

        self.model1, self.ds1, self.dl1 = load_model(
            self.base_config, set=self.split)
        progress_bar = tqdm(self.dl1, desc="Overlap Generation Batch     ")

        eot_token_id = self.model1.tokenizer.convert_tokens_to_ids('<eot>')
        values = []

        for step, batch in enumerate(progress_bar):
            speakerA = batch['speakerA']
            speakerB = batch['speakerB']

            input_idsA = speakerA['input_ids'].to(self.device)
            input_idsB = speakerB['input_ids'].to(self.device)

            typesA = speakerA['token_type_ids'].to(self.device)
            typesB = speakerB['token_type_ids'].to(self.device)

            eot_typesA = speakerA["other_token_type_ids"].to(
                self.device)
            eot_typesB = speakerB["other_token_type_ids"].to(
                self.device)

            # Find eot of other speaker which immediately follows a backchannel
            eot_items = torch.tenor([eot_token_id]).to(self.device)

            bc_maskA, bc_maskB = get_turn_after_bc(
                input_idsA, input_idsB, eot_items, typesA=typesA, typesB=typesB)

            out_serialised = predict_for_batch(
                self.model1, batch, device=self.device)

            if True:
                for i in range(4):
                    start = 0
                    offset = 10
                    end = start + 256
                    curr = [start, start + offset]
                    while True:
                        if (input_idsA[i, curr[0]:curr[1]] == eot_token_id).sum() == 0 and (input_idsB[i, curr[0]:curr[1]] == eot_token_id).sum() == 0:
                            curr = [curr[0] + offset, curr[1] + offset]
                            continue
                        _, _ = pp_pair_dialogs(self.model1.tokenizer, input_idsA[i],
                                               timings=speakerA['timings'][i], curr=curr, token_types=typesA[i],
                                               others={"Yield Type": eot_typesA[i], "BC Mask": bc_maskA[i]}, speaker='A')
                        curr, _ = pp_pair_dialogs(self.model1.tokenizer, input_idsB[i],
                                                  timings=speakerB['timings'][i], curr=curr, token_types=typesB[i],
                                                  others={"Yield Type": eot_typesB[i], "BC Mask": bc_maskB[i]}, speaker='B')
                        print()

                        if curr[0] > end:
                            print("---------------------------")
                            break
                break

    def compare_yields(self, thresh=0.5, step=20):
        """
        Compare yield performance between base and new config.
        For example, if base is Serialised and new is the full alignment why full alignment is superior.

        Where each fails in comparison in with the other so where base produces TP where new does not and
        where new produces TP where new cannot
        """
        self.model1, self.ds1, self.dl1 = load_model(
            self.base_config, set=self.split)

        progress_bar = tqdm(self.dl1, desc="Overlap Generation Batch     ")

        eot_token_id = self.model1.tokenizer.convert_tokens_to_ids('<eot>')
        values = []

        for step, batch in enumerate(progress_bar):
            speakerA = batch['speakerA']
            speakerB = batch['speakerB']

            input_idsA = speakerA['input_ids'].to(self.device)
            input_idsB = speakerB['input_ids'].to(self.device)

            eot_typesA = speakerA["other_token_type_ids"].to(
                self.device)
            eot_typesB = speakerB["other_token_type_ids"].to(
                self.device)

            out_serialised = predict_for_batch(
                self.model1, batch, device=self.device)

            yield_maskA = torch.logical_and(
                torch.ne(eot_typesA, TurnType.YIELD), torch.eq(input_idsA, eot_token_id))
            yield_maskB = torch.logical_and(
                torch.ne(eot_typesB, TurnType.YIELD), torch.eq(input_idsB, eot_token_id))

            input_ids = [input_idsA, input_idsB]
            yield_masks = [yield_maskA, yield_maskB]

            for speaker in range(2):
                other_speaker = (speaker + 1) % 2
                pred_curr = out_serialised.logits[speaker].clone()
                pred_other = out_serialised.logits[other_speaker].clone()
                yield_mask = yield_masks[speaker]

                prob_curr = torch.softmax(pred_curr, dim=-1)
                prob_other = torch.softmax(pred_other, dim=-1)

                # Find where full predicts trp but serialised does not at yield points
                prob_curr = prob_curr[yield_mask]
                prob_other = prob_other[yield_mask]

                values.append(
                    (speaker, batch, yield_masks, prob_curr, prob_other))

            break

        self.model1.cpu()
        self.model2, self.ds1, self.dl1 = load_model(self.new_config)
        progress_bar = tqdm(values, desc="Overlap Generation Batch     ")
        for step, (speaker, batch, yield_masks, curr_logits_serialised, other_logits_serialised) in enumerate(progress_bar):
            input_idsA = batch['speakerA' if speaker ==
                               0 else "speakerB"]['input_ids']
            input_idsB = batch['speakerB' if speaker ==
                               1 else "speakerA"]['input_ids']

            out_full = predict_for_batch(
                self.model2, batch, device=self.device)
            probs_fullA = [log.softmax(dim=-1).detach().cpu()
                           for log in out_serialised.logits[0]]
            probs_fullB = [log.softmax(dim=-1).detach().cpu()
                           for log in out_full.logits[1]]

            input_idsA, input_idsB = input_ids
            yield_maskA, yield_maskB = yield_masks

            other_speaker = (speaker + 1) % 2
            curr_logits_full = out_full.logits[speaker].clone()
            other_logits_full = out_full.logits[other_speaker].clone()
            yield_mask = yield_masks[speaker]

            probs_full = torch.softmax(curr_logits_full, dim=-1)

            # Find where full predicts trp but serialised does not at yield points
            yield_full_serialised = probs_full[yield_mask]
            diff = (yield_full_serialised -
                    curr_probs_serialised)[:, eot_token_id]

            yield_idxs = torch.nonzero(yield_mask, as_tuple=True)
            for idx, (batch_idx, seq_idx) in enumerate(zip(yield_idxs[0], yield_idxs[1])):
                if diff[idx] > thresh:
                    curr_probs_serialised = [log.softmax(
                        dim=-1).detach().cpu() for log in curr_logits_serialised]
                    other_probs_serialised = [log.softmax(
                        dim=-1).detach().cpu() for log in other_logits_serialised]

                    curr_probs_full = [log.softmax(
                        dim=-1).detach().cpu() for log in curr_logits_full]
                    other_probs_full = [log.softmax(
                        dim=-1).detach().cpu() for log in other_logits_full]

                    fig, (_, ax) = plot_dual_trp(curr_probs_serialised, curr_probs_full,
                                                 text=self.model1.tokenizer.convert_ids_to_tokens(
                                                     input_idsA),
                                                 eos_token="<eot>")
                    plot_dual_trp(other_probs_serialised, other_probs_full,
                                  text=self.model1.tokenizer.convert_ids_to_tokens(input_idsB), ax=ax, fig=fig,
                                  eos_token="<eot>")

    def rule_based_yield(self):
        """
        Rule-based Yield EOT prediction.
        Any point where two tokens are overlapping so, for whoever has the turn predict a <yield>
        """
        self.model1, self.ds1, self.dl1 = load_model(
            self.base_config, set=self.split, load=False, device="cpu")

        progress_bar = tqdm(self.dl1, desc="Overlap Generation Batch     ")

        eot_token_id = self.model1.tokenizer.convert_tokens_to_ids('<eot>')
        emp_token_id = self.model1.tokenizer.convert_tokens_to_ids('<emp>')
        values = []

        tp = 0
        tn = 0
        fn = 0
        fp = 0

        for step, batch in enumerate(progress_bar):
            speakerA = batch['speakerA']
            speakerB = batch['speakerB']

            input_idsA = speakerA['input_ids'].to(self.device)
            input_idsB = speakerB['input_ids'].to(self.device)

            turn_typesA = speakerA["token_type_ids"].to(
                self.device)
            turn_typesB = speakerB["token_type_ids"].to(
                self.device)

            eot_typesA = speakerA["other_token_type_ids"].to(
                self.device)
            eot_typesB = speakerB["other_token_type_ids"].to(
                self.device)

            yield_maskA = torch.logical_not(torch.logical_and(
                torch.ne(eot_typesA, TurnType.YIELD), torch.eq(input_idsA, eot_token_id)))
            yield_maskB = torch.logical_not(torch.logical_and(
                torch.ne(eot_typesB, TurnType.YIELD), torch.eq(input_idsB, eot_token_id)))

            yield_eot_mask = torch.logical_or(
                torch.logical_and(
                    turn_typesB != TurnType.NONE, eot_typesA == TurnType.YIELD
                ),
                torch.logical_and(
                    turn_typesA != TurnType.NONE, eot_typesB == TurnType.YIELD
                )
            )

            overlap_maskA = torch.logical_or(
                torch.logical_or(torch.eq(
                    turn_typesA, TurnType.OVERLAP), torch.eq(turn_typesA, TurnType.BACKCHANNEL)),
                torch.logical_or(torch.eq(turn_typesB, TurnType.OVERLAP), torch.eq(turn_typesB, TurnType.BACKCHANNEL)))
            overlap_maskB = torch.logical_or(
                torch.logical_or(torch.eq(
                    turn_typesA, TurnType.OVERLAP), torch.eq(turn_typesA, TurnType.BACKCHANNEL)),
                torch.logical_or(torch.eq(
                    turn_typesB, TurnType.OVERLAP), torch.eq(turn_typesB, TurnType.BACKCHANNEL)),
            )

            overlap_maskA = yield_eot_mask.logical_or(overlap_maskA)
            overlap_maskB = yield_eot_mask.logical_or(overlap_maskB)
            overlap_maskA = torch.logical_and(
                overlap_maskA, torch.ne(input_idsA, eot_token_id))
            overlap_maskB = torch.logical_and(
                overlap_maskB, torch.ne(input_idsB, eot_token_id))

            ignore_mask = torch.logical_not(torch.logical_or(
                input_idsA == self.model1.tokenizer.pad_token_id, input_idsB == self.model1.tokenizer.pad_token_id))

            eot_truthA = torch.eq(input_idsA, eot_token_id)
            eot_truthB = torch.eq(input_idsB, eot_token_id)
            eot_truthA = torch.logical_and(eot_truthA, yield_maskA)
            eot_truthB = torch.logical_and(eot_truthB, yield_maskB)
            # eot_truthA = torch.where(yield_maskA, torch.tensor(1), torch.tensor(0))
            # eot_truthB = torch.where(yield_maskB, torch.tensor(1), torch.tensor(0))

            if False:
                for i in range(4):
                    start = 0
                    offset = 10
                    end = start + 256
                    curr = [start, start + offset]
                    while True:
                        if (input_idsA[i, curr[0]:curr[1]] == eot_token_id).sum() == 0 and (input_idsB[i, curr[0]:curr[1]] == eot_token_id).sum() == 0:
                            curr = [curr[0] + offset, curr[1] + offset]
                            continue
                        _, _ = pp_pair_dialogs(self.model1.tokenizer, input_idsA[i],
                                               timings=speakerA['timings'][i], curr=curr, token_types=turn_typesA[i],
                                               others={"Yield Type": eot_typesA[i], "Prediction": overlap_maskA[i], "Truth": eot_truthA[i], "Yield Mask": yield_maskA[i]}, speaker='A')
                        curr, _ = pp_pair_dialogs(self.model1.tokenizer, input_idsB[i],
                                                  timings=speakerB['timings'][i], curr=curr, token_types=turn_typesB[i],
                                                  others={"Yield Type": eot_typesB[i], "Prediction": overlap_maskB[i], "Truth": eot_truthB[i], "Yield Mask": yield_maskB[i]}, speaker='B')
                        print()

                        if curr[0] > end:
                            print("---------------------------")
                            break

            yield_maskA = yield_maskA.logical_and(ignore_mask)[..., 1:]
            yield_maskB = yield_maskB.logical_and(ignore_mask)[..., 1:]

            eot_truthA = eot_truthA[..., 1:]
            # eot_truthA = eot_truthA[yield_maskA]
            eot_truthB = eot_truthB[..., 1:]
            # eot_truthB = eot_truthB[yield_maskB]

            overlap_maskA = overlap_maskA[..., :-1]
            # overlap_maskA = overlap_maskA[yield_maskA]
            overlap_maskB = overlap_maskB[..., :-1]
            # overlap_maskB = overlap_maskB[yield_maskB]

            tp += torch.logical_and(eot_truthA, overlap_maskA).sum()
            tn += torch.logical_and(torch.logical_not(eot_truthA),
                                    torch.logical_not(overlap_maskA)).sum()
            fn += torch.logical_and(eot_truthA,
                                    torch.logical_not(overlap_maskA)).sum()
            fp += torch.logical_and(torch.logical_not(eot_truthA),
                                    overlap_maskA).sum()

            tp += torch.logical_and(eot_truthB, overlap_maskB).sum()
            tn += torch.logical_and(torch.logical_not(eot_truthB),
                                    torch.logical_not(overlap_maskB)).sum()
            fn += torch.logical_and(eot_truthB,
                                    torch.logical_not(overlap_maskB)).sum()
            fp += torch.logical_and(torch.logical_not(eot_truthB),
                                    overlap_maskB).sum()

        recall = tp / (tp + fn) if tp + fn > 0 else 0
        specificity = tn / (tn + fp) if tn + fp > 0 else 0

        bacc = (recall + specificity) / 2

        print(f"rule based yield eot bacc={bacc}")

    def proportion_of_token_pairs(self):
        """
        For each token pair label with Complete Overlap, Partial Overlap, Backchannel, One Speaker
        """
        self.model1, self.ds1, self.dl1 = load_model(
            self.base_config, set=self.split, load=False, device="cpu")

        progress_bar = tqdm(self.dl1, desc="Overlap Generation Batch     ")

        eot_token_id = self.model1.tokenizer.convert_tokens_to_ids('<eot>')
        emp_token_id = self.model1.tokenizer.convert_tokens_to_ids('<emp>')

        stats = {
            TurnType.NORMAL: 0,
            TurnType.OVERLAP: 0,
            TurnType.YIELD: 0,
            TurnType.BACKCHANNEL: 0,
        }
        total = 0

        def handle_normal(x): return torch.logical_or(
            x == TurnType.NORMAL, x == TurnType.INTERRUPT)

        for step, batch in enumerate(progress_bar):
            speakerA = batch['speakerA']
            speakerB = batch['speakerB']

            input_idsA = speakerA['input_ids'].to(self.device).reshape(-1)
            input_idsB = speakerB['input_ids'].to(self.device).reshape(-1)

            turn_typesA = speakerA["token_type_ids"].to(
                self.device).reshape(-1)
            turn_typesB = speakerB["token_type_ids"].to(
                self.device).reshape(-1)

            eot_typesA = speakerA["other_token_type_ids"].to(
                self.device).reshape(-1)
            eot_typesB = speakerB["other_token_type_ids"].to(
                self.device).reshape(-1)

            attention_maskA = speakerA['attention_mask'].to(
                self.device).reshape(-1)
            attention_maskB = speakerB['attention_mask'].to(
                self.device).reshape(-1)

            turn_typesA = turn_typesA[attention_maskA]
            turn_typesB = turn_typesB[attention_maskB]
            eot_typesA = eot_typesA[attention_maskA]
            eot_typesB = eot_typesB[attention_maskB]

            turn_typesA = torch.where(
                input_idsA == eot_token_id, torch.tensor(TurnType.NONE), turn_typesA)
            turn_typesB = torch.where(
                input_idsB == eot_token_id, torch.tensor(TurnType.NONE), turn_typesB)
            eot_typesA = torch.where(
                input_idsA == eot_token_id, torch.tensor(TurnType.NONE), eot_typesA)
            eot_typesB = torch.where(
                input_idsB == eot_token_id, torch.tensor(TurnType.NONE), eot_typesB)

            # NORMAL exists where token_type_ids is 1 for one channel and 0 for other
            # Handle NORMAL yields by removing
            normalA = torch.logical_and(handle_normal(
                turn_typesA), turn_typesB == TurnType.NONE)
            normalA = normalA.logical_and(eot_typesA != TurnType.YIELD)
            normalB = torch.logical_and(handle_normal(
                turn_typesB), turn_typesA == TurnType.NONE)
            normalB = normalB.logical_and(eot_typesB != TurnType.YIELD)

            # BACKCHANNEL exists where token_type_ids is anything for one channel and BACKCHANNEL for other
            bcA = turn_typesA == TurnType.BACKCHANNEL
            bcB = turn_typesB == TurnType.BACKCHANNEL

            # YIELD exists where eot_types is YIELD
            yieldA = eot_typesA == TurnType.YIELD
            yieldB = eot_typesB == TurnType.YIELD

            # COMPLETE OVERLAP exists where speaker is OVERLAP
            overlapA = turn_typesA == TurnType.OVERLAP
            overlapB = turn_typesB == TurnType.OVERLAP

            stats[TurnType.NORMAL] += (normalA.sum() + normalB.sum()).item()
            stats[TurnType.YIELD] += (yieldA.sum() + yieldB.sum()).item()
            stats[TurnType.BACKCHANNEL] += (bcA.sum() + bcB.sum()).item()
            stats[TurnType.OVERLAP] += (overlapA.sum() + overlapB.sum()).item()
            total += torch.logical_and(attention_maskA,
                                       attention_maskB).sum().item()

        total = sum(stats.values())
        for k, v in stats.items():
            percen = round(v/total * 100, 2)
            print(f"{k.name}: {v} with percentage {percen}%")

    def print_dialogs(self, prior, gen, post):
        prior_length = 20
        curr = [40, 52]
        print("PRIOR")
        pp_pair_dialogs(self.model1.tokenizer,
                        prior['speakerA']['input_ids'][0], curr=curr, speaker="speakerA")
        pp_pair_dialogs(self.model1.tokenizer,
                        prior['speakerB']['input_ids'][0], curr=curr, speaker="speakerB")
        print()

        step = 7
        print("GENERATED")
        for idx in range(0, len(gen['input_idsA'][0]), step):
            pp_pair_dialogs(self.model1.tokenizer, input_ids=gen['input_idsA'][0], curr=[
                            idx, idx+step], speaker="speakerA")
            pp_pair_dialogs(self.model1.tokenizer, input_ids=gen['input_idsB'][0], curr=[
                            idx, idx+step], speaker="speakerB")
            print()

        print("--------------------")


def load_model(loaded_config, set='test', load=True, device="cuda:0"):
    load_path = get_abs_path(loaded_config.load_path)
    logging.getLogger(__name__).info(
        f"model: loading model from {load_path}")

    if os.path.exists(os.path.join(load_path, "config.json")):
        with open(os.path.join(load_path, "config.json")) as f:
            config = Namespace(**json.load(f))
    else:
        config = loaded_config

    logging.getLogger(__name__).info(f"Loaded config: {config}")
    config.device = loaded_config.device

    model = PairwiseGPT(
        **vars(config)
    )
    model.eval()

    if config.load_model:
        print("LOAD MODEL from ", load_path)
        load_model_file = get_latest_model(os.path.dirname(
            load_path))
        checkpoint = torch.load(load_model_file)
        model.load_state_dict(checkpoint['model_state_dict'])

    if config.cuda:
        model = model.cuda()
    else:
        model = model.cpu()

    assert set in {'train', 'test', 'val', 'all'}

    test_ds = PairwiseGenerationDM(
        split=set,
        tokenizer=model.get_tokenizer(),
        overwrite=loaded_config.overwrite,
        basic_mode=False,
        include_end_bc_token=True,
        include_overlap_token=True,
        remove_start_tokens=True,
        filter_bc_overlap_token=loaded_config.filter_bc_overlap_token,
        max_length=config.max_length,
        keep_length=config.keep_length,
        overlap_length=config.overlap_length,
        datasets=config.datasets,
    )
    test_ds.prepare_data()
    test_dl = DataLoader(
        test_ds,
        batch_size=config.batch_size,
        collate_fn=test_ds.collate_fn,
        num_workers=8,
        shuffle=False
    )

    return model, test_ds, test_dl


def unload_model(model):
    model = model.cpu()
    return model


if __name__ == "__main__":
    parser = build_parser()
    config = parser.parse_args()

    config.evaluate_on_full = True

    # SERIALISED
    config.device = "cpu"
    config.overwrite = False  # True
    # config.load_path = "trained_model/2024-05-16:02-55-40/"

    # FULL ALIGNMENT
    new_config = copy.deepcopy(config)
    # new_config.load_path = "trained_model/2024-05-16:10-52-49/"

    print(config)
    print("----------------")
    print(new_config)

    index = 1
    configs = (config, new_config)
    co = configs[index]
    name = ("Serialised", "Full")[index]

    analyser = Analyser(co, config, device=co.device, name=name, split="test")
    # analyser(EvaluateType.OVERLAP_EFFECT)
    # analyser.show_yields()
    analyser.compare_bc_turns()
