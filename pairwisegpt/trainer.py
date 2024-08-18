import torch
import os
import json
import numpy as np
from datetime import datetime

import logging

import wandb
from matplotlib import pyplot as plt
from tqdm import tqdm

import copy

from pairwise_generation_dm import TurnType
from pairwisegpt.utils import plot_trp
from pairwisegpt.metrics import Metrics, PR_AUC, ROC_AUC, F1_Score, Perplexity, MetricType


def get_abs_path(filepath):
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), filepath)


def get_new_filename(save_dir):
    now = datetime.now()
    current_time_str = now.strftime("%Y-%m-%d:%H-%M-%S")
    return os.path.join(save_dir, current_time_str)


def get_latest_model(path, before=None):
    list_dir = os.listdir(path)
    latest_model = None
    max_index = 100000

    if before is not None:
        before = before.split("/")[-1][:-3]
        max_index = int(before[6:])

    latest_index = -1
    for item in list_dir:
        if item[:5] == 'model':
            index = int(''.join(x for x in item if x.isdigit()))
            if latest_model is None or index > latest_index and index < max_index:
                latest_index = index
                latest_model = item

    if latest_model is None:
        raise RuntimeError("model file not found")

    return os.path.join(path, latest_model)


class Trainer:
    def __init__(self,
                 model=None,
                 criterion=None,
                 optimizer=None,
                 epochs=None,
                 load_from_checkpoint=None,
                 device=None,
                 log_interval=None,
                 save_path="./",
                 early_stop=5,
                 dev_mode=False,
                 config=None,
                 remove_start_tokens=False,
                 include_overlap_token=False,
                 include_yield_token=False,
                 include_bc_token=False,
                 include_end_bc_token=False,
                 mask_attention_emp=False,
                 no_loss_emp=False,
                 remove_emp_metric_generation=False,
                 save_model_allowed=False,
                 serialise_data=False,
                 no_emp_tokens=False,
                 evaluate_on_full=False,
                 filter_bc_overlap_token=False,
                 **kwargs,
                 ):

        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.epochs = epochs
        self.device = device
        self.log_interval = log_interval

        self.epoch = 0
        self.dev_mode = dev_mode

        self.train_history = {}
        self.val_history = {}
        self.test_history = {}

        self.best = {
            'epoch': 0,
            'model_state_dict': None,
            'optimizer_state_dict': None,
            'loss': None,
            'best_threshold': None,
        }
        self.last_best_path = None

        self.save_path = get_abs_path(get_new_filename(save_path))
        if not os.path.isdir(self.save_path):
            os.mkdir(self.save_path)

        with open(os.path.join(self.save_path, "config.json"), "w") as config_file:
            json.dump(vars(config), config_file)

        self.logger = logging.getLogger(__name__)

        self.early_stop = early_stop

        self.load_model_file = load_from_checkpoint
        if self.load_model_file is not None:
            self.load_from_checkpoint()

        self.eot_token_id = self.model.tokenizer.convert_tokens_to_ids('<eot>')
        self.sot_token_id = self.model.tokenizer.convert_tokens_to_ids('<sot>')
        self.sbc_token_id = self.model.tokenizer.convert_tokens_to_ids('<sbc>')
        self.bc_token_id = self.model.tokenizer.convert_tokens_to_ids('<bc>')
        self.ebc_token_id = self.model.tokenizer.convert_tokens_to_ids('<ebc>')
        self.sint_token_id = self.model.tokenizer.convert_tokens_to_ids(
            '<sint>')
        self.eint_token_id = self.model.tokenizer.convert_tokens_to_ids(
            '<eint>')
        self.yield_token_id = self.model.tokenizer.convert_tokens_to_ids(
            '<yield>')

        self.tokens_dict = {
            '<eot>': self.eot_token_id,
            '<sot>': self.sot_token_id,
            '<sbc>': self.sbc_token_id,
            '<bc>': self.bc_token_id,
            '<ebc>': self.ebc_token_id,
            '<sint>': self.sint_token_id,
            '<eint>': self.eint_token_id,
            '<yield>': self.yield_token_id
        }
        self.eot_tokens = torch.tensor([self.model.tokenizer.convert_tokens_to_ids(
            '<ebc>'), self.model.tokenizer.convert_tokens_to_ids('<eint>'), self.model.tokenizer.eos_token_id]).long().to(self.device)

        self.filter_bc_overlap_token = filter_bc_overlap_token
        self.val_metrics = self.init_metrics(
            include_overlap_token, include_bc_token, include_end_bc_token, include_yield_token, "val")
        self.test_metrics = self.init_metrics(
            include_overlap_token, include_bc_token, include_end_bc_token, include_yield_token, "test")

        self.global_step = 0
        self.remove_start_tokens = remove_start_tokens

        self.mask_attention_emp = mask_attention_emp
        self.no_loss_emp = no_loss_emp
        self.remove_emp_metric_generation = remove_emp_metric_generation

        self.serialise_data = serialise_data
        self.no_emp_tokens = no_emp_tokens
        self.evaluate_on_full = evaluate_on_full

        if self.no_emp_tokens:
            self.remove_emp_metric_generation = False
            self.mask_attention_emp = False
            self.no_loss_emp = False

        self.save_model_allowed = save_model_allowed

        self.include_yield_token = include_yield_token

        self.logger.info(
            f"using filter_bc_overlap_token: {self.filter_bc_overlap_token}")

    def init_metrics(self, include_overlap_token=False, include_bc_token=False, include_end_bc_token=False,
                     include_yield_token=False,
                     type="test", **kwargs):
        """
        (self.eot_token_id, 'eot', [
            MetricType.BACC], TurnType.NORMAL),
        (self.eot_token_id, 'eot', [
            MetricType.BACC], TurnType.OVERLAP),
        """
        metrics = [
            # Out of all tokens
            (self.eot_token_id, 'eot', [
                MetricType.BACC, MetricType.PR_AUC, MetricType.F1_SCORE, MetricType.NRR, MetricType.BR], TurnType.NONE),
            # Out of all tokens outside of overlap
            (self.eot_token_id, 'eot', [
                MetricType.BACC], TurnType.BACKCHANNEL),
            (-1, 'perplexity', [MetricType.PERPLEXITY], TurnType.NORMAL),
            (-1, 'perplexity', [MetricType.PERPLEXITY], TurnType.NON_OVERLAP),
            (-1, 'perplexity', [MetricType.PERPLEXITY], TurnType.NONE),
            (-1, 'perplexity', [MetricType.PERPLEXITY], TurnType.OVERLAP),
            (-1, 'perplexity', [MetricType.PERPLEXITY], TurnType.OVERLAP),

            # Predicting the start of the turn
            ('all', 'int', [MetricType.BACC,
                            MetricType.PR_AUC], TurnType.NONE),
            ('all', 'int', [MetricType.BACC], TurnType.NORMAL),
            ('all', 'int', [MetricType.BACC], TurnType.OVERLAP),
            ('all', 'int', [MetricType.BACC], TurnType.BACKCHANNEL),
            ('all', 'int', [MetricType.BACC], TurnType.INTERRUPT)
        ]

        if include_overlap_token and self.eint_token_id != self.model.tokenizer.pad_token_id:
            metrics.append(
                (self.eint_token_id, 'eint', [
                    MetricType.BACC, MetricType.PR_AUC], TurnType.NONE)
            )
        if include_end_bc_token and self.ebc_token_id != self.model.tokenizer.pad_token_id:
            metrics.append(
                (self.ebc_token_id, 'ebc', [
                    MetricType.BACC, MetricType.PR_AUC], TurnType.NONE)
            )
        if include_bc_token and self.bc_token_id != self.model.tokenizer.pad_token_id:
            metrics.append(
                (self.bc_token_id, 'bc', [
                    MetricType.BACC, MetricType.F1_SCORE, MetricType.PR_AUC], TurnType.NONE)
            )
        if include_yield_token and self.yield_token_id != self.model.tokenizer.pad_token_id:
            metrics.extend(
                [
                    (self.yield_token_id, 'yield', [
                        MetricType.BACC, MetricType.F1_SCORE, MetricType.PR_AUC], TurnType.NONE),
                    (self.yield_token_id, 'yield', [
                        MetricType.BACC, MetricType.F1_SCORE, MetricType.PR_AUC], TurnType.NORMAL),
                    (self.yield_token_id, 'yield', [
                        MetricType.BACC, MetricType.F1_SCORE, MetricType.PR_AUC], TurnType.OVERLAP),
                    (self.yield_token_id, 'yield', [
                        MetricType.BACC], TurnType.BACKCHANNEL),
                    ('rule', self.yield_token_id, 'eot',
                     [MetricType.BACC], TurnType.NONE),
                ]
            )
        else:
            # Metric to deal with how the model performs at predicting <eot> where there should be <yield>
            metrics.extend([
                (self.eot_token_id, 'eot', [
                    MetricType.BACC, MetricType.PR_AUC], TurnType.YIELD),
                (self.eot_token_id, 'eot', [
                    MetricType.BACC, MetricType.PR_AUC], TurnType.NON_YIELD),
                ('rule', self.eot_token_id, 'eot',
                 [MetricType.BACC], TurnType.YIELD)
            ])

        if not self.filter_bc_overlap_token:
            metrics.extend([
                (-1, 'perplexity', [MetricType.PERPLEXITY],
                 TurnType.NORMAL, None, True),
                (-1, 'perplexity', [MetricType.PERPLEXITY],
                 TurnType.NON_OVERLAP, None, True),
                (-1, 'perplexity', [MetricType.PERPLEXITY],
                 TurnType.NONE, None, True),
                (-1, 'perplexity', [MetricType.PERPLEXITY],
                 TurnType.OVERLAP, None, True),
            ])

            metrics.extend([
                ([self.ebc_token_id, self.eint_token_id, self.eot_token_id],
                 'eot_ebc_eint', [MetricType.BACC], TurnType.NONE)
            ])

        tokens = {
            k: self.model.tokenizer.convert_tokens_to_ids(k) for k in self.model.tokenizer.special_tokens
        }
        tokens['<|endoftext|>'] = self.model.tokenizer.pad_token_id

        return Metrics(metrics, tokens_dict=tokens, type=type)

    def load_from_checkpoint(self):
        try:
            checkpoint = torch.load(self.load_model_file)
        except:
            self.load_model_file = get_latest_model(os.path.dirname(
                self.load_model_file), before=self.load_model_file)
            self.load_from_checkpoint()
        else:
            self.logger.info(
                f"model: loading parameters for model {checkpoint.keys()}")
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epoch = checkpoint['epoch']
            self.global_step = checkpoint['global_step']

    def train(self, train_dl, val_dl, test_dl, scheduler):
        best_loss = float('Inf')
        best_PPL = float('Inf')

        # For early stopping, number of iterations without loss improvement
        not_improving_x = 0
        progress_bar = tqdm(range(self.epoch, self.epoch +
                                  self.epochs), desc='Epoch   ')

        self.scheduler = scheduler
        val_metrics, val_params = self.validate(val_dl)
        test_metrics = self.test(test_dl, val_params)

        for idx in progress_bar:
            train_metrics = self.train_epoch(train_dl)
            val_metrics, val_params = self.validate(val_dl)
            test_metrics = self.test(test_dl, val_params)

            for key in train_metrics:
                if key not in self.train_history:
                    self.train_history[key] = []
                self.train_history[key].append(train_metrics[key])
            for key in val_metrics:
                if key not in self.val_history:
                    self.val_history[key] = []
                self.val_history[key].append(val_metrics[key])
            for key in test_metrics:
                if key not in self.test_history:
                    self.test_history[key] = []
                self.test_history[key].append(test_metrics[key])

            self.save_history(self.save_path)

            avg_valid_loss = val_metrics['avg_loss']
            PPL = val_metrics.get('val_filter_PPL_NONE', -1)

            if avg_valid_loss < best_loss or PPL < best_PPL:
                not_improving_x = 0

                """
                self.trp_example_plots()
                self.text_generation_examples()
                """

                best_loss = avg_valid_loss
                best_PPL = PPL
                self.best['model_state_dict'] = self.model.state_dict()
                self.best['optimizer_state_dict'] = self.optimizer.state_dict()
                self.best['loss'] = best_loss
                self.best['epoch'] = idx + 1
                self.best['global_step'] = self.global_step
                self.best['best_threshold'] = val_params

                model_name = "model_" + str(self.best['epoch']) + ".pt"
                self.save_training(os.path.join(self.save_path, model_name))
                self.prune_training()

                self.last_best_path = os.path.join(self.save_path, model_name)

                self.logger.info(
                    f"train: saving model at {os.path.join(self.save_path, model_name)} of epoch {self.best['epoch']}")
            else:
                not_improving_x += 1

                if not_improving_x >= self.early_stop and self.early_stop > 0:
                    self.logger.info("train: early stop")
                    progress_bar.close()
                    return self.train_history

        progress_bar.close()
        return self.train_history

    def get_postfix_str(self, step, f1, loss, count, tp, fp, fn, tn):
        return (f'loss={loss / (step + 1): .4f}, f1={f1 / (step + 1): .4f}, accuracy={(tp + tn) / count: .4f} '
                f'precision={(tp / (tp + fp)) if tp + fp != 0 else 1: .4f}, recall={tp / (tp + fn) if tp + fn != 0 else 0: .4f}, '
                f'bAcc={0.5 * (tp / (tp + fn) + tn / (fp + tn)) if tp + fn != 0 and fp + tn != 0 else 0: .4f}')

    def metric_output(self, metrics):
        output = ""
        if 'avg_loss' in metrics.keys():
            output += f"loss={metrics['avg_loss'] :.4f}"
        if 'eot_f1' in metrics.keys():
            output += f"f1={metrics['eot_f1'] :.4f}"
        if 'eot_acc' in metrics.keys():
            output += f"acc={metrics['eot_acc'] :.4f}"
        if 'recall' in metrics.keys():
            output += f"loss={metrics['eot_recall'] :.4f}"
        if 'rouge_mean' in metrics.keys():
            output += f"rouge2={metrics['rouge_mean'] :.4f}"

        return output

    def train_epoch(self, train_dl):
        self.model.train()
        total_loss = 0

        progress_bar = tqdm(train_dl, desc='Training', unit="batch")
        padding = torch.zeros((8, 183)).to(self.device)
        padding[:, :5] = 1

        emp_token_id = self.model.tokenizer.convert_tokens_to_ids("<emp>")
        pred_label = []

        for step, batch in enumerate(progress_bar):
            self.optimizer.zero_grad()

            input_idsA = batch["speakerA"]["input_ids"].to(self.device)
            attention_maskA = batch["speakerA"]["attention_mask"].to(
                self.device)
            token_type_idsA = batch["speakerA"]["token_type_ids"].to(
                self.device)
            speaker_idsA = batch['speakerA']['speaker_ids'].to(self.device)

            input_idsB = batch["speakerB"]["input_ids"].to(self.device)
            attention_maskB = batch["speakerB"]["attention_mask"].to(
                self.device)
            token_type_idsB = batch["speakerB"]["token_type_ids"].to(
                self.device)
            speaker_idsB = batch['speakerB']['speaker_ids'].to(self.device)

            if self.no_loss_emp:
                label_loss_maskA = torch.logical_and(
                    attention_maskA, torch.ne(input_idsA, emp_token_id))
                label_loss_maskB = torch.logical_and(
                    attention_maskB, torch.ne(input_idsB, emp_token_id))
            else:
                label_loss_maskA = attention_maskA
                label_loss_maskB = attention_maskB

            labelsA = self.generate_labels(
                input_idsA, mask=label_loss_maskA)
            projection_labelsA = self.generate_projection_labels(labelsA)

            labelsB = self.generate_labels(
                input_idsB, mask=label_loss_maskB)
            projection_labelsB = self.generate_projection_labels(labelsB)

            if self.mask_attention_emp:
                attention_maskA = torch.logical_and(
                    attention_maskA, torch.ne(input_idsA, emp_token_id))
                attention_maskB = torch.logical_and(
                    attention_maskB, torch.ne(input_idsB, emp_token_id))

            out = self.model.forward(
                input_idsA=input_idsA,
                labelsA=labelsA,
                projection_labelsA=projection_labelsA,
                attention_maskA=attention_maskA,
                token_type_idsA=speaker_idsA,
                input_idsB=input_idsB,
                labelsB=labelsB,
                projection_labelsB=projection_labelsB,
                attention_maskB=attention_maskB,
                token_type_idsB=speaker_idsB,
            )

            loss = out.loss
            if out.mc_loss is not None:
                loss = out.loss + out.mc_loss

            loss.backward()
            self.optimizer.step()

            if step % self.log_interval == 0 and not self.dev_mode:
                wandb.log({"loss": loss,
                           "global_step": self.global_step})

            total_loss += loss.item()
            avg_loss = round(total_loss / (step + 1), 4)
            progress_bar.set_postfix_str(f"loss={avg_loss}")

            self.global_step += 1

        avg_loss = total_loss / len(train_dl)
        metrics = {}
        metrics['avg_loss'] = avg_loss

        wandb.log({"train_loss": avg_loss,
                   "global_step": self.global_step})
        try:
            self.trp_example_plots(
                input_idsA, input_idsB, out, name="TRP/train")
        except Exception as e:
            self.logger.warning(f"Failed to plot trps for train {e}")

        progress_bar.disable = False
        progress_bar.set_postfix_str(self.metric_output(metrics))
        progress_bar.close()

        return metrics

    def validate(self, val_dl):
        """
        Returns various peformance metrics and then the parameters (thresholds)
        required to obtain those metrics to be used at testing time
        """
        total_loss, total_count = 0, 0

        emp_token_id = self.model.tokenizer.convert_tokens_to_ids("<emp>")
        parameters = {}

        self.model.eval()
        self.metrics = self.val_metrics
        with torch.no_grad():
            progress_bar = tqdm(val_dl, desc='Validation')

            for step, batch in enumerate(progress_bar):
                input_idsA = batch["speakerA"]["input_ids"].to(self.device)
                attention_maskA = batch["speakerA"]["attention_mask"].to(
                    self.device)
                token_type_idsA = batch["speakerA"]["token_type_ids"].to(
                    self.device)
                eot_typesA = batch["speakerA"]["other_token_type_ids"].to(
                    self.device)
                turn_overlapA = batch["speakerA"]["turn_overlap"].to(
                    self.device)
                speaker_idsA = batch['speakerA']['speaker_ids'].to(self.device)

                input_idsB = batch["speakerB"]["input_ids"].to(self.device)
                attention_maskB = batch["speakerB"]["attention_mask"].to(
                    self.device)
                token_type_idsB = batch["speakerB"]["token_type_ids"].to(
                    self.device)
                eot_typesB = batch["speakerB"]["other_token_type_ids"].to(
                    self.device)
                turn_overlapB = batch["speakerB"]["turn_overlap"].to(
                    self.device)
                speaker_idsB = batch['speakerB']['speaker_ids'].to(self.device)

                if self.no_loss_emp:
                    label_loss_maskA = torch.logical_and(
                        attention_maskA, torch.ne(input_idsA, emp_token_id))
                    label_loss_maskB = torch.logical_and(
                        attention_maskB, torch.ne(input_idsB, emp_token_id))
                else:
                    label_loss_maskA = attention_maskA.detach().clone()
                    label_loss_maskB = attention_maskB.detach().clone()

                labelsA = self.generate_labels(
                    input_idsA, mask=label_loss_maskA)
                projection_labelsA = self.generate_projection_labels(labelsA)

                labelsB = self.generate_labels(
                    input_idsB, mask=label_loss_maskB)
                projection_labelsB = self.generate_projection_labels(labelsB)

                ignore_maskA = attention_maskA.detach().clone()
                ignore_maskB = attention_maskB.detach().clone()
                if self.mask_attention_emp:
                    attention_maskA = torch.logical_and(
                        attention_maskA, torch.ne(input_idsA, emp_token_id))
                    attention_maskB = torch.logical_and(
                        attention_maskB, torch.ne(input_idsB, emp_token_id))

                out = self.model.forward(
                    input_idsA=input_idsA,
                    labelsA=labelsA,
                    projection_labelsA=projection_labelsA,
                    attention_maskA=attention_maskA,
                    token_type_idsA=speaker_idsA,
                    input_idsB=input_idsB,
                    labelsB=labelsB,
                    projection_labelsB=projection_labelsB,
                    attention_maskB=attention_maskB,
                    token_type_idsB=speaker_idsB,
                )

                loss = out.loss
                total_loss += loss.item()

                # Switch overlap mask from if no empty token in both channels as this could leave out where there is an empty token
                # but this is because there is a slight pause in one speaker
                # Switch to where turn types are both of type not in a turn
                # Includes BC and overlaps on purpose as this is the false case where no turn shift but still overlapping occurs

                if self.remove_emp_metric_generation:
                    labels_not_empA = torch.logical_and(
                        torch.eq(input_idsA, emp_token_id), torch.eq(token_type_idsA, TurnType.NONE))
                    labels_not_empB = torch.logical_and(
                        torch.eq(input_idsB, emp_token_id), torch.eq(token_type_idsB, TurnType.NONE))
                    ignore_maskA = torch.logical_and(
                        torch.logical_not(labels_not_empA), ignore_maskA)
                    ignore_maskB = torch.logical_and(
                        torch.logical_not(labels_not_empB), ignore_maskB)

                metric_masksA, metric_masksB, all_metric_masks = self.generate_masks(
                    input_idsA,
                    input_idsB,
                    token_type_idsA,
                    token_type_idsB,
                    eot_typesA,
                    eot_typesB,
                    ignore_maskA,
                    ignore_maskB,
                    self.eot_tokens)

                self.add_to_metrics(out.logits[0].detach(), input_idsA.detach(
                ), **metric_masksA)
                self.add_to_metrics(out.logits[1].detach(), input_idsB.detach(
                ), **metric_masksB)

                # self.add_to_bacc(out.logits[0], labelsA)
                # self.add_to_bacc(out.logits[1], labelsB)

                avg_loss = round(total_loss / (step + 1), 4)
                progress_bar.set_postfix_str(f"loss={avg_loss}")

            metrics, counts, parameters = self.metrics.calculate()
            metrics['avg_loss'] = total_loss / len(val_dl)

            if not self.dev_mode:
                self.plot_metric_graphs()

                wandb_out = {"val_loss": round(total_loss / len(val_dl), 4),
                             "global_step": self.global_step}

                for key, count in counts.items():
                    if count is None:
                        continue

                    self.logger.info(f"count for {key} is {count}")

                for key, parameter in parameters.items():
                    if parameter is None:
                        continue

                    self.logger.info(f"threshold for {key} is {parameter}")

                wandb_out.update(metrics)

                # Figure out way to extrapolate metrics
                wandb.log(wandb_out)

            self.metrics.reset()
            progress_bar.disable = False
            progress_bar.set_postfix_str(self.metric_output(metrics))

        progress_bar.close()
        self.model.train()

        return metrics, parameters

    def test(self, test_dl, params):
        """
        Returns various peformance metrics and then the parameters (thresholds)
        required to obtain those metrics to be used at testing time
        """
        total_loss, total_count = 0, 0

        emp_token_id = self.model.tokenizer.convert_tokens_to_ids("<emp>")
        parameters = {}

        self.model.eval()
        self.metrics = self.test_metrics
        self.emp_metrics = copy.deepcopy(self.test_metrics)
        with torch.no_grad():
            progress_bar = tqdm(test_dl, desc='Test')

            for step, batch in enumerate(progress_bar):
                input_idsA = batch["speakerA"]["input_ids"].to(self.device)
                attention_maskA = batch["speakerA"]["attention_mask"].to(
                    self.device)
                token_type_idsA = batch["speakerA"]["token_type_ids"].to(
                    self.device)
                eot_typesA = batch["speakerA"]["other_token_type_ids"].to(
                    self.device)
                speaker_idsA = batch['speakerA']['speaker_ids'].to(self.device)

                input_idsB = batch["speakerB"]["input_ids"].to(self.device)
                attention_maskB = batch["speakerB"]["attention_mask"].to(
                    self.device)
                token_type_idsB = batch["speakerB"]["token_type_ids"].to(
                    self.device)
                eot_typesB = batch["speakerB"]["other_token_type_ids"].to(
                    self.device)
                speaker_idsB = batch['speakerB']['speaker_ids'].to(self.device)

                if self.no_loss_emp:
                    label_loss_maskA = torch.logical_and(
                        attention_maskA, torch.ne(input_idsA, emp_token_id))
                    label_loss_maskB = torch.logical_and(
                        attention_maskB, torch.ne(input_idsB, emp_token_id))
                else:
                    label_loss_maskA = attention_maskA.detach().clone()
                    label_loss_maskB = attention_maskB.detach().clone()

                labelsA = self.generate_labels(
                    input_idsA, mask=label_loss_maskA)
                projection_labelsA = self.generate_projection_labels(labelsA)

                labelsB = self.generate_labels(
                    input_idsB, mask=label_loss_maskB)
                projection_labelsB = self.generate_projection_labels(labelsB)

                ignore_maskA = attention_maskA.detach().clone()
                ignore_maskB = attention_maskB.detach().clone()
                if self.mask_attention_emp:
                    attention_maskA = torch.logical_and(
                        attention_maskA, torch.ne(input_idsA, emp_token_id))
                    attention_maskB = torch.logical_and(
                        attention_maskB, torch.ne(input_idsB, emp_token_id))

                out = self.model.forward(
                    input_idsA=input_idsA,
                    labelsA=labelsA,
                    projection_labelsA=projection_labelsA,
                    attention_maskA=attention_maskA,
                    token_type_idsA=speaker_idsA,
                    input_idsB=input_idsB,
                    labelsB=labelsB,
                    projection_labelsB=projection_labelsB,
                    attention_maskB=attention_maskB,
                    token_type_idsB=speaker_idsB,
                )

                loss = out.loss
                total_loss += loss.item()

                emp_metric_masksA, emp_metric_masksB, emp_all_metric_masks = self.generate_masks(
                    input_idsA,
                    input_idsB,
                    token_type_idsA,
                    token_type_idsB,
                    eot_typesA,
                    eot_typesB,
                    ignore_maskA,
                    ignore_maskB,
                    self.eot_tokens,)

                self.add_to_metrics(out.logits[0].detach(), input_idsA.detach(
                ).clone(), metric=self.emp_metrics, **emp_metric_masksA)
                self.add_to_metrics(out.logits[1].detach(), input_idsB.detach(
                ).clone(), metric=self.emp_metrics, **emp_metric_masksB)
                self.add_to_joint_metrics(
                    out.logits[0].detach(), out.logits[1].detach(),
                    input_idsA.detach().clone(), input_idsB.clone(), metric=self.emp_metrics,
                    **emp_all_metric_masks)

                if self.remove_emp_metric_generation:
                    labels_not_empA = torch.logical_and(
                        torch.eq(input_idsA, emp_token_id), torch.eq(token_type_idsA, TurnType.NONE))
                    labels_not_empB = torch.logical_and(
                        torch.eq(input_idsB, emp_token_id), torch.eq(token_type_idsB, TurnType.NONE))
                    ignore_maskA = torch.logical_and(
                        torch.logical_not(labels_not_empA), ignore_maskA)
                    ignore_maskB = torch.logical_and(
                        torch.logical_not(labels_not_empB), ignore_maskB)

                metric_masksA, metric_masksB, all_metric_masks = self.generate_masks(
                    input_idsA,
                    input_idsB,
                    token_type_idsA,
                    token_type_idsB,
                    eot_typesA,
                    eot_typesB,
                    ignore_maskA,
                    ignore_maskB,
                    self.eot_tokens,)

                self.add_to_metrics(out.logits[0].detach(), input_idsA.detach(
                ).clone(), **metric_masksA)
                self.add_to_metrics(out.logits[1].detach(), input_idsB.detach(
                ).clone(), **metric_masksB)

                self.add_to_joint_metrics(
                    out.logits[0].detach(), out.logits[1].detach(),
                    input_idsA.detach().clone(), input_idsB.clone(),
                    **all_metric_masks)

                avg_loss = round(total_loss / (step + 1), 4)
                progress_bar.set_postfix_str(f"loss={avg_loss}")

            metrics, counts, parameters = self.metrics.calculate(params)
            emp_metrics, _, _ = self.emp_metrics.calculate(
                params, prefix="emp")
            metrics['avg_loss'] = total_loss / len(test_dl)

            if not self.dev_mode:
                self.plot_metric_graphs()

                wandb_out = {"test_loss": round(total_loss / len(test_dl), 4),
                             "global_step": self.global_step}

                for key, count in counts.items():
                    if count is None:
                        continue

                    self.logger.info(f"count for {key} is {count}")

                for key, parameter in parameters.items():
                    if parameter is None:
                        continue

                    self.logger.info(f"threshold for {key} is {parameter}")

                for key, value in metrics.items():
                    self.logger.info(f"{key} = {value}")

                new_emp_metric = {}
                for key, value in emp_metrics.items():
                    new_emp_metric["emp_" + key] = value

                wandb_out.update(metrics)
                wandb_out.update(new_emp_metric)

                # Figure out way to extrapolate metrics
                wandb.log(wandb_out)

                self.trp_example_plots(
                    input_idsA, input_idsB, out, name="TRP/test")

            self.metrics.reset()
            self.emp_metrics.reset()
            progress_bar.disable = False
            progress_bar.set_postfix_str(self.metric_output(metrics))

        progress_bar.close()
        self.model.train()

        return metrics

    def generate_labels(self, input_ids, mask=None, pad_id=-100):
        labels = input_ids.clone()
        labels[torch.logical_not(mask)] = pad_id

        return labels

    def generate_projection_labels(self, labels):
        batch_size, num_labels = labels.size()

        mask = (labels == self.model.tokenizer.eos_token_id)
        distances = torch.full((batch_size, num_labels),
                               num_labels, device=labels.device)
        distances[mask] = 0

        for i in range(num_labels - 2, -1, -1):
            distances[:, i] = torch.minimum(
                distances[:, i], distances[:, i + 1] + 1)

        return distances

    def calculate_loss(self, output, labels, padding=None):
        mask = torch.ones(labels.shape).to(self.device)
        if padding is not None:
            mask = padding

        output = output[:, :100].contiguous()

        self.criterion.reduction = "none"
        loss = self.criterion(
            output.view(-1, output.size(-1)), labels.view(-1))
        loss = loss.view(labels.shape)

        loss *= mask
        return loss.sum() / mask.sum()

    def add_to_metrics(self, logits, labels, metric=None, **kwargs):
        probs = logits.softmax(dim=-1)
        if metric is None:
            self.metrics.add(probs, labels, logits=logits, **kwargs)
        else:
            metric.add(probs, labels, logits=logits, **kwargs)

    """
    Returns true/false if logits predict ground truth trp sufficiently wrong
    """

    def is_not_trp_example(self, logits, labels):
        probs = logits.softmax(dim=-1)
        trp_prob = probs[..., self.model.tokenizer.eos_token_id]
        trp_prob = trp_prob[..., :-1]

        labels = labels[..., 1:]
        is_trp = labels == self.model.tokenizer.eos_token_id
        not_trp = labels != self.model.tokenizer.eos_token_id

        return torch.max(trp_prob - is_trp.long()).item() > 0.5

    def save_training(self, path):
        if self.save_model_allowed:
            self.logger.info(f"save model at {path}")
            torch.save(self.best, path)

    def prune_training(self):
        if self.last_best_path is None or not os.path.exists(self.last_best_path):
            return

        os.remove(self.last_best_path)
        self.logger.info(f"Remove last model at {self.last_best_path}")

    def save_history(self, path):
        self.logger.info("trainer: save history")
        for key in self.train_history:
            try:
                np.save(os.path.join(
                    path, f"train_{key}"), self.train_history[key])
            except TypeError as e:
                print(f"Cannot save {key}")
        for key in self.val_history:
            try:
                np.save(os.path.join(
                    path, f"val_{key}"), self.val_history[key])
            except TypeError as e:
                print(f"Cannot save {key}")

    def print_dialogue(self, input_ids, prediction, output, label):
        output = f"Input: {self.model.tokenizer.decode(input_ids)}\n"
        output += f"Output: {self.model.tokenizer.decode(output['input_ids'])}\n"
        output += f"Prediction: {prediction}, Label: {label}"

    @torch.no_grad
    def generate_from_string(self, t):
        out = self.model(t["input_ids"], token_type_ids=t["token_type_ids"])
        out["probs"] = out["logits"].softmax(dim=-1)
        out["trp_probs"] = (out["probs"])[...,
                                          self.model.tokenizer.eos_token_id]
        out["tokens"] = self.model.tokenizer.convert_ids_to_tokens(
            t["input_ids"][0])
        return out

    def plot_metric_graphs(self):
        graphs = self.metrics.get_graphs()

        for token, token_graph in graphs.items():
            wandb.log({
                token: [wandb.Image(ax) for k, (fig, ax) in token_graph.items()],
                'global_step': self.global_step
            })

    def trp_example_plots(self, input_idsA, input_idsB, out, name="TRP/example"):
        figs = []
        global_steps = []

        special_tokens = [
            token_id for token_id in self.model.tokenizer.special_tokens if token_id != "<emp>"]
        special_tokens.append("<eot>")

        tokens = [self.model.tokenizer.convert_tokens_to_ids(
            token_id) for token_id in self.model.tokenizer.special_tokens if token_id != "<emp>"]
        tokens.append(self.model.tokenizer.convert_tokens_to_ids("<eot>"))

        logitsA = [out.logits[0][..., token_id].cpu() for token_id in tokens]
        logitsB = [out.logits[1][..., token_id].cpu() for token_id in tokens]

        step = 50
        for batch_idx in range(len(input_idsA)):
            probsA = [log.softmax(dim=-1)[batch_idx] for log in logitsA]
            probsB = [log.softmax(dim=-1)[batch_idx] for log in logitsB]
            for idx in range(0, len(input_idsA[batch_idx]), step):
                pA = [x[idx:idx + step] for x in probsA]
                fig, (_, ax) = plot_trp(trp=pA, text=self.model.tokenizer.convert_ids_to_tokens(
                    input_idsA[batch_idx][idx:idx + step]), eos_token='[SEP]', special_tokens=special_tokens)
                pB = [x[idx:idx + step] for x in probsB]
                _, _ = plot_trp(trp=pB, text=self.model.tokenizer.convert_ids_to_tokens(
                    input_idsB[batch_idx][idx:idx + step]), eos_token='[SEP]', special_tokens=special_tokens, fig=fig,
                    ax=ax)
                figs.append(wandb.Image(fig))
                plt.close('all')
        global_steps.append(self.global_step)

        wandb.log({f"{name}": figs, "global_step": self.global_step})

    def add_to_bacc(self, logits, labels):
        sot_token_id = self.model.tokenizer.convert_tokens_to_ids("<sot>")
        eot_token_id = self.model.tokenizer.convert_tokens_to_ids("<eot>")

        probs = logits.softmax(dim=-1)
        sot_prob = probs[..., sot_token_id]
        sot_prob = sot_prob[..., :-1]

        eot_prob = probs[..., eot_token_id]
        eot_prob = eot_prob[..., :-1]

        labels = labels[..., 1:]
        is_sot = labels == sot_token_id
        not_sot = labels != sot_token_id
        is_eot = labels == eot_token_id
        not_eot = labels != eot_token_id
        for bacc in self.bacc_sot:
            thresh = bacc['threshold']
            bacc['tp'] += (is_sot & (sot_prob > thresh)).sum()
            bacc['tn'] += (not_sot & (sot_prob < thresh)).sum()
            bacc['fp'] += (not_sot & (sot_prob > thresh)).sum()
            bacc['fn'] += (is_sot & (sot_prob < thresh)).sum()

            bacc['bacc'] = ((bacc['tp'] / (bacc['tp'] + bacc['fn'])) +
                            (bacc['tn'] / (bacc['fp'] + bacc['tn']))) / 2

        for bacc in self.bacc_eot:
            thresh = bacc['threshold']
            bacc['tp'] += (is_eot & (eot_prob > thresh)).sum()
            bacc['tn'] += (not_eot & (eot_prob < thresh)).sum()
            bacc['fp'] += (not_eot & (eot_prob > thresh)).sum()
            bacc['fn'] += (is_eot & (eot_prob < thresh)).sum()

            bacc['bacc'] = ((bacc['tp'] / (bacc['tp'] + bacc['fn'])) +
                            (bacc['tn'] / (bacc['fp'] + bacc['tn']))) / 2

    def compute_bacc(self, token):
        if token == "<eot>":
            max_bacc = max(self.bacc_eot, key=lambda x: x['bacc'])

            self.bacc_eot = [{
                "tp": 0,
                "fp": 0,
                "fn": 0,
                "tn": 0,
                "bacc": 0,
                "threshold": threshold,
            } for threshold in self.thresholds]

            return max_bacc['bacc']

        elif token == "<sot>":
            max_bacc = max(self.bacc_sot, key=lambda x: x['bacc'])

            self.bacc_sot = [{
                "tp": 0,
                "fp": 0,
                "fn": 0,
                "tn": 0,
                "bacc": 0,
                "threshold": threshold,
            } for threshold in self.thresholds]

            return max_bacc['bacc']

        return 0.5

    """
    Generation of binary masks for calculating statistics on the results of the model
    """

    def generate_masks(self, input_idsA, input_idsB, token_type_idsA, token_type_idsB, eot_typesA, eot_typesB,
                       ignore_maskA, ignore_maskB, eot_tensor):
        # where an overlap is occuring in the dialogue (either complete/partial overlap)
        overlap_maskA, overlap_maskB = None, None
        non_overlap_maskA, non_overlap_maskB = None, None  # the negation of overlap mask
        # tokens that are the start of an utterance (first token)
        interrupt_maskA, interrupt_maskB = None, None
        yield_maskA, yield_maskB = None, None  # all tokens aside from non-yield <eot>

        non_yield_maskA, non_yield_maskB = None, None
        # all tokens aside from yield <eot>
        turn_masksA, turn_masksB = None, None  # masks for both speakers turns'
        bc_maskA, bc_maskB = None, None  # where a backchannel is occuring

        mask_specialA, mask_specialB = None, None

        interrupt_mask_normalA, interrupt_mask_normalB = None, None
        interrupt_mask_bcA, interrupt_mask_bcB = None, None
        interrupt_mask_overlapA, interrupt_mask_overlapB = None, None
        interrupt_mask_yieldA, interrupt_mask_yieldB = None, None

        if (not self.serialise_data and not self.no_emp_tokens) or self.evaluate_on_full:
            # Overlapping where:
            #   A is YIELD and B is INTERRUPTING (include <eot> of A in mask)
            #   B is YIELD and A is INTERRUPTIN)           #   A is speaking AND B is OVERLAP
            #   B is speaking AND A is OVERLAP
            #   Avoids A ending turn and B starting after and being labelled as OVERLAP if in line with <eot>
            yield_int_overlap = torch.logical_or(
                torch.logical_and(
                    token_type_idsB == TurnType.INTERRUPT, eot_typesA == TurnType.YIELD
                ),
                torch.logical_and(
                    token_type_idsA == TurnType.INTERRUPT, eot_typesB == TurnType.YIELD
                )
            )

            overlap_mask = torch.logical_or(
                token_type_idsA == TurnType.OVERLAP, token_type_idsB == TurnType.OVERLAP)
            overlap_maskA = torch.logical_or(
                yield_int_overlap, overlap_mask)
            overlap_maskB = torch.logical_or(
                yield_int_overlap, overlap_mask)

            # Non-overlap where only one content token is occuring at a time
            non_overlap_maskA = torch.logical_and(
                token_type_idsA != TurnType.NONE, overlap_mask == TurnType.NONE)
            non_overlap_maskB = torch.logical_and(
                token_type_idsA == TurnType.NONE, token_type_idsB != TurnType.NONE)
            non_overlap_maskA = torch.logical_or(
                non_overlap_maskA, non_overlap_maskB)
            non_overlap_maskB = torch.logical_or(
                non_overlap_maskA, non_overlap_maskB)
            # overlap_maskA = torch.logical_or(overlap_maskA, overlap_maskB)
            # overlap_maskB = overlap_maskA.detach().clone()
            # Make sure that for A consider overlap as where turn ends during overlap too
            # But not where A is ending turn normally

            non_yield_maskA = torch.logical_not(torch.logical_and(
                torch.eq(eot_typesA, TurnType.YIELD), torch.eq(input_idsA, self.eot_token_id)))
            non_yield_maskB = torch.logical_not(torch.logical_and(
                torch.eq(eot_typesB, TurnType.YIELD), torch.eq(input_idsB, self.eot_token_id)))

            # Every token aside from <eot> that is not YIELD <eot>
            yield_maskA = torch.logical_not(torch.logical_and(
                torch.ne(eot_typesA, TurnType.YIELD), torch.eq(input_idsA, self.eot_token_id)))
            yield_maskB = torch.logical_not(torch.logical_and(
                torch.ne(eot_typesB, TurnType.YIELD), torch.eq(input_idsB, self.eot_token_id)))

            # Finds all areas where there is one content token and then rolls that tensor
            # to include the first token of the utterance
            # We then divide first token of utterance based on its turn type
            interrupt_maskA = torch.logical_and(torch.eq(token_type_idsA, TurnType.NONE),
                                                torch.ne(token_type_idsB, TurnType.NONE))
            interrupt_maskB = torch.logical_and(torch.ne(token_type_idsA, TurnType.NONE),
                                                torch.eq(token_type_idsB, TurnType.NONE))
            interrupt_maskA = torch.logical_or(interrupt_maskA,
                                               interrupt_maskA.roll(1, dims=-1))
            interrupt_maskB = torch.logical_or(interrupt_maskB,
                                               interrupt_maskB.roll(1, dims=-1))
            interrupt_mask_normalA = torch.logical_and(
                interrupt_maskA, torch.logical_or(token_type_idsA == TurnType.NONE, token_type_idsA == TurnType.NORMAL))
            interrupt_mask_normalB = torch.logical_and(
                interrupt_maskB, torch.logical_or(token_type_idsB == TurnType.NONE, token_type_idsB == TurnType.NORMAL))
            interrupt_mask_bcA = torch.logical_and(
                interrupt_maskA, torch.logical_or(token_type_idsA == TurnType.NONE, token_type_idsA == TurnType.BACKCHANNEL))
            interrupt_mask_bcB = torch.logical_and(
                interrupt_maskB, torch.logical_or(token_type_idsB == TurnType.NONE, token_type_idsB == TurnType.BACKCHANNEL))
            interrupt_mask_overlapA = torch.logical_and(
                interrupt_maskA, torch.logical_or(token_type_idsA == TurnType.NONE, token_type_idsA == TurnType.OVERLAP))
            interrupt_mask_overlapB = torch.logical_and(
                interrupt_maskB, torch.logical_or(token_type_idsB == TurnType.NONE, token_type_idsB == TurnType.OVERLAP))
            interrupt_mask_yieldA = torch.logical_and(
                interrupt_maskA, torch.logical_or(token_type_idsA == TurnType.NONE, token_type_idsA == TurnType.INTERRUPT))
            interrupt_mask_yieldB = torch.logical_and(
                interrupt_maskB, torch.logical_or(token_type_idsB == TurnType.NONE, token_type_idsB == TurnType.INTERRUPT))

            turn_normal = torch.tensor(
                [TurnType.NORMAL, TurnType.INTERRUPT], device=self.device)
            turn_masksA = torch.isin(token_type_idsA, turn_normal)
            turn_masksB = torch.isin(token_type_idsB, turn_normal)

            eint = self.model.tokenizer.convert_tokens_to_ids('<eint>')
            ebc = self.model.tokenizer.convert_tokens_to_ids('<ebc>')
            eot_items = torch.tensor(
                [self.model.tokenizer.eos_token_id], device="cuda")
            if self.include_yield_token:
                eot_items = torch.tensor(
                    [self.yield_token_id, self.model.tokenizer.eos_token_id], device="cuda")

            # Update Ignore Mask to not count "special tokens" only if actually filtered
            # Shouldn't matter
            if not self.filter_bc_overlap_token:
                """
                ignore_maskA = torch.logical_not(torch.logical_and(ignore_maskA, input_idsA == eint))
                ignore_maskB = torch.logical_not(torch.logical_and(ignore_maskB, input_idsB == eint))


                ignore_maskA = torch.logical_not(torch.logical_and(ignore_maskA, input_idsA == ebc))
                ignore_maskB = torch.logical_not(torch.logical_and(ignore_maskB, input_idsB == ebc))
                """

                mask_specialA = torch.logical_not(
                    torch.logical_or(input_idsA == eint, input_idsA == ebc))
                mask_specialB = torch.logical_not(
                    torch.logical_or(input_idsB == eint, input_idsB == ebc))

            bc_maskA, bc_maskB = get_turn_after_bc(
                input_idsA, input_idsB, eot_items, eint=eint, ebc=ebc)

        if not self.evaluate_on_full and (self.serialise_data or self.no_emp_tokens):
            # Remove where <eot> tokens appear
            overlap_maskA = torch.logical_or(
                torch.eq(eot_typesA, TurnType.YIELD),
                torch.eq(eot_typesA, TurnType.OVERLAP))

            overlap_maskB = torch.logical_or(
                torch.eq(eot_typesB, TurnType.YIELD),
                torch.eq(eot_typesB, TurnType.OVERLAP))
            overlap_mask = torch.logical_or(overlap_maskA, overlap_maskB)
            overlap_maskA = overlap_mask.clone().detach()
            overlap_maskB = overlap_mask.clone().detach()

            yield_maskA = torch.logical_not(torch.logical_and(
                torch.ne(eot_typesA, TurnType.YIELD), torch.eq(input_idsA, self.eot_token_id)))
            yield_maskB = torch.logical_not(torch.logical_and(
                torch.ne(eot_typesB, TurnType.YIELD), torch.eq(input_idsB, self.eot_token_id)))

            non_yield_maskA = torch.logical_not(torch.logical_and(
                torch.eq(eot_typesA, TurnType.YIELD), torch.eq(input_idsA, self.eot_token_id)))
            non_yield_maskB = torch.logical_not(torch.logical_and(
                torch.eq(eot_typesB, TurnType.YIELD), torch.eq(input_idsB, self.eot_token_id)))

            turn_masksA = torch.ne(token_type_idsA, TurnType.NONE)
            turn_masksB = torch.ne(token_type_idsB, TurnType.NONE)

        ignore_maskA = ignore_maskA.detach().clone()
        ignore_maskB = ignore_maskB.detach().clone()

        metric_masksA = dict(
            overlap_mask=overlap_maskA,
            non_overlap_mask=non_overlap_maskA,
            yield_mask=yield_maskA,
            non_yield_mask=non_yield_maskA,
            interrupt_mask=interrupt_maskA,
            turn_mask=turn_masksA,
            bc_mask=bc_maskA,
            ignore_mask=ignore_maskA,
            mask_special=mask_specialA,
            interrupt_mask_normal=interrupt_mask_normalA,
            interrupt_mask_bc=interrupt_mask_bcA,
            interrupt_mask_overlap=interrupt_mask_overlapA,
            interrupt_mask_yield=interrupt_mask_yieldA)
        metric_masksB = dict(
            overlap_mask=overlap_maskB,
            non_overlap_mask=non_overlap_maskB,
            yield_mask=yield_maskB,
            non_yield_mask=non_yield_maskB,
            interrupt_mask=interrupt_maskB,
            turn_mask=turn_masksB,
            bc_mask=bc_maskB,
            ignore_mask=ignore_maskB,
            mask_special=mask_specialB,
            interrupt_mask_normal=interrupt_mask_normalB,
            interrupt_mask_bc=interrupt_mask_bcB,
            interrupt_mask_overlap=interrupt_mask_overlapB,
            interrupt_mask_yield=interrupt_mask_yieldB)
        all_metric_masks = dict(
            overlap_maskA=overlap_maskA,
            overlap_maskB=overlap_maskB,
            non_overlap_maskA=non_overlap_maskA,
            non_overlap_maskB=non_overlap_maskB,
            yield_maskA=yield_maskA,
            yield_maskB=yield_maskB,
            non_yield_maskA=non_yield_maskA,
            non_yield_maskB=non_yield_maskB,
            interrupt_maskA=interrupt_maskA,
            interrupt_maskB=interrupt_maskB,
            turn_maskA=turn_masksA,
            turn_maskB=turn_masksB,
            bc_maskA=bc_maskA,
            bc_maskB=bc_maskB,
            ignore_maskA=ignore_maskA,
            ignore_maskB=ignore_maskB,
            mask_specialA=mask_specialA,
            mask_specialB=mask_specialB,
            interrupt_mask_normalA=interrupt_mask_normalA,
            interrupt_mask_normalB=interrupt_mask_normalB,
            interrupt_mask_bcA=interrupt_mask_bcA,
            interrupt_mask_bcB=interrupt_mask_bcB,
            interrupt_mask_overlapA=interrupt_mask_overlapA,
            interrupt_mask_overlapB=interrupt_mask_overlapB,
            interrupt_mask_yieldA=interrupt_mask_yieldA,
            interrupt_mask_yieldB=interrupt_mask_yieldB)
        return metric_masksA, metric_masksB, all_metric_masks

    def remove_special(self, input_ids, mask, token=None):
        if token is None:
            mask = torch.logical_and(mask, torch.ne(input_ids, token))
            return mask

        for k, v in self.tokens_dict.items():
            mask = torch.logical_and(
                mask,
                torch.ne(input_ids, v)
            )
        return mask

    def add_to_joint_metrics(self, logitsA, logitsB, input_idsA, input_idsB, metric=None, **kwargs):
        if metric is None:
            self.metrics.add_joint(logitsA, logitsB, input_idsA=input_idsA,
                                   input_idsB=input_idsB, rule='joint', **kwargs)
        else:
            metric.add_joint(logitsA, logitsB, input_idsA=input_idsA,
                             input_idsB=input_idsB, rule='joint', **kwargs)

    def evaluate(self, val_dl, test_dl):
        val_metrics, val_params = self.validate(val_dl)
        test_metrics = self.test(test_dl, val_params)
