import torch
import os
import json
import numpy as np
from datetime import datetime

import logging
import time

import wandb
from tqdm import tqdm
from sklearn.metrics import f1_score

from seqeval.metrics import classification_report, f1_score, accuracy_score
from torchmetrics.text import BLEUScore
from torchmetrics.classification import BinaryF1Score, BinaryAccuracy, BinaryRecall

from pairwisegpt.utils import plot_trp as  plot_trp_pair
from gptonly.utils import plot_trp
from pairwise_generation_dm import TurnType
from generation_dm import BINS
from gptonly.metrics import MetricType, MeasureType, Metrics


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
                 categorize_projection=False,
                 projection_labels=False,
                 weight_projection=0.5,
                 include_yield_tokens=False,
                 individual_speaker_tokens=False,
                 include_speaker_embedings=False,
                 use_speaker_token_in_embedding=False,
                 save_model_allowed=False,
                 offline=False,
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
        self.test_history = {}
        self.val_history = {}

        self.best = {
            'epoch': 0,
            'model_state_dict': None,
            'optimizer_state_dict': None,
            'loss': None,
        }

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

        self.projection_labels = projection_labels

        self.ts_token_id = self.model.tokenizer.eos_token_id
        self.weight_projective_loss = weight_projection
        self.categorize_projection = categorize_projection

        self.eot_token_id = self.model.tokenizer.convert_tokens_to_ids('<ts>')
        self.include_yield_tokens = include_yield_tokens
        self.yield_token_id = self.model.tokenizer.convert_tokens_to_ids(
            '<yield>')

        self.individual_ts = individual_speaker_tokens
        self.val_metrics = self.init_metrics(type="val")
        self.test_metrics = self.init_metrics(type="test")

        self.save_model_allowed = save_model_allowed
        self.use_speaker_token_in_embedding = use_speaker_token_in_embedding
        self.include_speaker_embeddings = include_speaker_embedings

        self.global_step = 0
        self.offline = offline

    def init_metrics(self, type="test"):
        metrics = [
            (-1, 'perplexity', [MetricType.PERPLEXITY],
             TurnType.NONE, MeasureType.PERPLEXITY, 0)
        ]
        if not self.individual_ts:
            metrics.extend([
                (self.ts_token_id, 'eot', [
                 MetricType.BACC, MetricType.PR_AUC, MetricType.F1_SCORE, MetricType.NRR, MetricType.BR], TurnType.NONE, MeasureType.CATEGORICAL, 1),
                (self.ts_token_id, 'eot', [
                 MetricType.BACC, MetricType.PR_AUC, MetricType.F1_SCORE], TurnType.NORMAL, MeasureType.CATEGORICAL, 1),
                (self.ts_token_id, 'eot', [
                    MetricType.BACC, MetricType.PR_AUC], TurnType.OVERLAP, MeasureType.CATEGORICAL, 1),
            ])
        else:
            ts_idA = self.model.tokenizer.convert_tokens_to_ids('<speakerA>')
            ts_idB = self.model.tokenizer.convert_tokens_to_ids('<speakerB>')
            metrics.extend([
                ([ts_idA, ts_idB], 'eot', [
                 MetricType.BACC, MetricType.PR_AUC, MetricType.F1_SCORE, MetricType.NRR, MetricType.BR], TurnType.NONE, MeasureType.CATEGORICAL, 1),
                ([ts_idA, ts_idB], 'eot', [
                 MetricType.BACC, MetricType.PR_AUC, MetricType.F1_SCORE], TurnType.NORMAL, MeasureType.CATEGORICAL, 1),
                ([ts_idA, ts_idB], 'eot', [
                    MetricType.BACC, MetricType.PR_AUC], TurnType.OVERLAP, MeasureType.CATEGORICAL, 1),
            ])

        if self.include_yield_tokens:
            metrics.extend([
                (self.yield_token_id, 'yield', [
                 MetricType.BACC, MetricType.PR_AUC], TurnType.NONE, MeasureType.CATEGORICAL, 1),
                (self.yield_token_id, 'yield_token', [
                 MetricType.BACC, MetricType.PR_AUC], TurnType.NORMAL, MeasureType.CATEGORICAL, 1),
                (self.yield_token_id, 'yield_token', [
                    MetricType.BACC, MetricType.PR_AUC], TurnType.OVERLAP, MeasureType.CATEGORICAL, 1),
            ])
        else:
            metrics.append(
                (self.eot_token_id, 'eot', [
                 MetricType.BACC, MetricType.PR_AUC], TurnType.YIELD, MeasureType.CATEGORICAL, 1),
            )

        if self.projection_labels:
            if self.categorize_projection:
                metrics.append(
                    (-1, 'projection', [MetricType.ACC], TurnType.NONE,
                     MeasureType.CATEGORICAL, BINS))
            else:
                # Considered as part of the LOSS function??
                """
                metrics.append(
                    (-1, 'projection', [MetricType.ACC], TurnType.NONE,
                     MeasureType.REGRESSION, BINS))
                """

        return Metrics(metrics, type=type)

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
            # self.model.tokenizer.from_pretrained(os.path.join(os.path.dirname(self.load_model_file), "tokenizer"))
            # self.model.init_tokenizer()
            self.epoch = checkpoint['epoch']
            self.global_step = checkpoint['global_step']

    def train(self, train_dl, val_dl, test_dl, scheduler):
        best_loss = float('Inf')

        # For early stopping, number of iterations without loss improvement
        not_improving_x = 0
        progress_bar = tqdm(range(self.epoch, self.epoch +
                                  self.epochs), desc='Epoch   ')

        self.scheduler = scheduler
        val_metrics = self.validate(test_dl)

        for idx in progress_bar:
            train_metrics = self.train_epoch(train_dl)
            val_metrics, val_params = self.validate(val_dl)
            test_metrics = self.test(test_dl, val_params)

            for key in train_metrics:
                if key not in self.train_history:
                    self.train_history[key] = []
                self.train_history[key].append(train_metrics[key])
            for key in val_metrics:
                if key not in self.test_history:
                    self.val_history[key] = []
                self.val_history[key].append(val_metrics[key])
            for key in test_metrics:
                if key not in self.test_history:
                    self.test_history[key] = []
                self.test_history[key].append(test_metrics[key])

            self.save_history(self.save_path)

            avg_valid_loss = (
                train_metrics['avg_loss'] + test_metrics['avg_loss']) / 2

            if avg_valid_loss < best_loss:
                not_improving_x = 0

                self.trp_example_plots()
                self.text_generation_examples()

                best_loss = avg_valid_loss
                self.best['model_state_dict'] = self.model.state_dict()
                self.best['optimizer_state_dict'] = self.optimizer.state_dict()
                self.best['loss'] = best_loss
                self.best['epoch'] = idx + 1
                self.best['global_step'] = self.global_step

                model_name = "model_" + str(self.best['epoch']) + ".pt"
                self.save_training(os.path.join(self.save_path, model_name))

                self.logger.info(
                    f"train: saving model at {os.path.join(self.save_path, model_name)}")
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

    def train_epoch(self, train_dl):
        self.model.train()
        total_loss, total_count = 0, 0
        total_f1 = 0
        tp, fp, fn, tn = 0, 0, 0, 0

        progress_bar = tqdm(train_dl, desc='Training', unit="batch")
        padding = torch.zeros((8, 183)).to(self.device)
        padding[:, :5] = 1

        pred_label = []

        for step, batch in enumerate(progress_bar):
            self.optimizer.zero_grad()

            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            token_type_ids = batch["token_type_ids"].to(self.device)
            time_until_ts = batch['time_until_ts'].to(self.device)
            speaker_ids = batch['speaker_ids'].to(self.device)

            labels = self.generate_labels(input_ids, mask=attention_mask)
            projection_labels = self.generate_projection_labels(
                time_until_ts, mask=attention_mask)

            out = self.model.forward(
                input_ids, labels=labels, projection_labels=projection_labels, attention_mask=attention_mask,
                token_type_ids=speaker_ids)

            loss = out.loss
            if out.mc_loss is not None:
                loss = out.loss + self.weight_projective_loss * out.mc_loss

            loss.backward()
            self.optimizer.step()

            if step % self.log_interval == 0:
                wandb.log({"loss": loss,
                           "global_step": self.global_step})

            total_loss += loss.item()
            avg_loss = round(total_loss / (step + 1), 4)
            progress_bar.set_postfix_str(f"loss={avg_loss}")

            self.global_step += 1

        if self.scheduler:
            # self.scheduler.step()
            pass

        avg_loss = total_loss / len(train_dl)

        metrics = {}
        metrics['avg_loss'] = avg_loss

        wandb.log({"train_loss": avg_loss,
                   "global_step": self.global_step})

        progress_bar.disable = False
        progress_bar.set_postfix_str(self.metric_output(metrics))
        progress_bar.close()

        return metrics

    def validate(self, val_dl):
        total_loss, total_count = 0, 0
        total_lm_loss, total_mc_loss = 0, 0
        count = 0

        self.metrics = self.val_metrics

        self.model.eval()
        with torch.no_grad():
            progress_bar = tqdm(val_dl, desc='Validation')

            for step, batch in enumerate(progress_bar):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                token_type_ids = batch['token_type_ids'].to(self.device)
                other_token_type_ids = batch['other_token_type_ids'].to(
                    self.device)
                time_until_ts = batch['time_until_ts'].to(self.device)
                speaker_ids = batch['speaker_ids'].to(self.device)

                labels = self.generate_labels(input_ids, mask=attention_mask)
                projection_labels = self.generate_projection_labels(
                    time_until_ts, mask=attention_mask)
                out = self.model.forward(
                    input_ids, labels=labels, projection_labels=projection_labels,
                    attention_mask=attention_mask, token_type_ids=speaker_ids)

                loss = out.loss
                mc_loss = 0
                if out.mc_loss is not None:
                    mc_loss = out.mc_loss
                    total_mc_loss += mc_loss.item()

                t_loss = loss + self.weight_projective_loss * mc_loss

                total_loss += t_loss.item()
                total_lm_loss += loss.item()

                overlap_mask = torch.logical_or(other_token_type_ids == TurnType.OVERLAP,
                                                other_token_type_ids == TurnType.YIELD)
                yield_mask = torch.logical_not(
                    torch.logical_and(other_token_type_ids == TurnType.YIELD,
                                      labels == self.eot_token_id))
                if self.individual_ts:
                    end_turn_mask = torch.logical_or(labels == self.model.tokenizer.convert_tokens_to_ids('<speakerA>'),
                                                     labels == self.model.tokenizer.convert_tokens_to_ids('<speakerB>'))
                    yield_mask = torch.logical_not(torch.logical_and(other_token_type_ids != TurnType.YIELD,
                                                   end_turn_mask))

                turn_maskA = (speaker_ids == 1)
                turn_maskB = (speaker_ids == 2)
                if self.include_speaker_embeddings:
                    turn_maskA = speaker_ids == self.model.tokenizer.convert_tokens_to_ids(
                        '<speakerA>')
                    turn_maskB = speaker_ids == self.model.tokenizer.convert_tokens_to_ids(
                        '<speakerB>')

                self.add_to_metrics(out.logits.detach(), labels.detach(),
                                    overlap_mask=overlap_mask, ignore_mask=attention_mask,
                                    yield_mask=yield_mask, turn_mask=(turn_maskA, turn_maskB))

                if out.mc_loss is not None:
                    self.add_to_metrics(out.mc_logits.detach(), projection_labels.detach(
                    ), type='projection', measure_type=MeasureType.REGRESSION if not self.categorize_projection else MeasureType.CATEGORICAL)

                avg_loss = round(total_loss / (step + 1), 4)
                progress_bar.set_postfix_str(f"loss={avg_loss}")

            metrics, counts, parameters = self.metrics.calculate()
            divide = len(val_dl) if len(val_dl) > 0 else 1
            metrics['avg_loss'] = total_loss / divide
            if total_mc_loss != 0:
                metrics['avg_lm_loss'] = total_lm_loss / divide
                metrics['avg_mc_loss'] = total_mc_loss / divide

            if not self.dev_mode:
                self.plot_metric_graphs()
                self.trp_example_plots()
                self.trp_sample_plots(input_ids, out)
                self.text_generation_examples()

                wandb_out = {}
                if len(val_dl) > 0:
                    self.generate_on_validation_set(
                        input_ids, mask=attention_mask, speaker_ids=speaker_ids)

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
                wandb.log(wandb_out)

            self.metrics.reset()
            progress_bar.disable = False
            progress_bar.set_postfix_str(f"loss={metrics['avg_loss']}")

        progress_bar.close()
        self.model.train()

        return metrics, parameters

    def test(self, test_dl, params):
        """
        Returns various peformance metrics and then the parameters (thresholds)
        required to obtain those metrics to be used at testing time
        """
        total_loss, total_count = 0, 0
        total_lm_loss, total_mc_loss = 0, 0

        emp_token_id = self.model.tokenizer.convert_tokens_to_ids("<emp>")
        parameters = {}

        self.model.eval()
        self.metrics = self.test_metrics
        with torch.no_grad():
            progress_bar = tqdm(test_dl, desc='Test')

            for step, batch in enumerate(progress_bar):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(
                    self.device)
                token_type_ids = batch["token_type_ids"].to(
                    self.device)
                other_token_type_ids = batch["other_token_type_ids"].to(
                    self.device)
                time_until_ts = batch['time_until_ts'].to(self.device)
                speaker_ids = batch['speaker_ids'].to(self.device)

                labels = self.generate_labels(
                    input_ids, mask=attention_mask)
                projection_labels = self.generate_projection_labels(
                    time_until_ts, mask=attention_mask)
                out = self.model.forward(
                    input_ids, labels=labels, projection_labels=projection_labels, attention_mask=attention_mask,
                    token_type_ids=speaker_ids)

                loss = out.loss
                mc_loss = 0
                if out.mc_loss is not None:
                    mc_loss = out.mc_loss
                    total_mc_loss += mc_loss.item()

                t_loss = loss + self.weight_projective_loss * mc_loss

                total_loss += t_loss.item()
                total_lm_loss += loss.item()

                overlap_mask = torch.logical_or(other_token_type_ids == TurnType.OVERLAP,
                                                other_token_type_ids == TurnType.YIELD)
                yield_mask = torch.logical_not(
                    torch.logical_and(other_token_type_ids == TurnType.YIELD,
                                      labels == self.eot_token_id))

                turn_maskA = (speaker_ids == 1)
                turn_maskB = (speaker_ids == 2)

                self.add_to_metrics(out.logits.detach(), labels.detach(),
                                    overlap_mask=overlap_mask, ignore_mask=attention_mask,
                                    yield_mask=yield_mask, turn_mask=(turn_maskA, turn_maskB))

                if out.mc_loss is not None:
                    self.add_to_metrics(out.mc_logits.detach(), projection_labels.detach(
                    ), type='projection', measure_type=MeasureType.REGRESSION if not self.categorize_projection else MeasureType.CATEGORICAL)

                avg_loss = round(total_loss / (step + 1), 4)
                progress_bar.set_postfix_str(f"loss={avg_loss}")

                break

            metrics, counts, parameters = self.metrics.calculate(params)
            metrics['avg_loss'] = total_loss / len(test_dl)

            if not self.dev_mode:
                self.plot_metric_graphs()

                wandb_out = {"val_loss": round(total_loss / len(test_dl), 4),
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

                if self.offline:
                    for key, value in metrics.items():
                        self.logger.info(f"{key} = {value}")

                wandb_out.update(metrics)

                # Figure out way to extrapolate metrics
                wandb.log(wandb_out)

                self.trp_example_plots()
                self.trp_sample_plots(input_ids, out)
                self.text_generation_examples()
                self.generate_on_validation_set(
                    input_ids, mask=attention_mask, speaker_ids=token_type_ids)

                wandb_out = {"val_loss": round(total_loss / len(test_dl), 4),
                             "global_step": self.global_step}
                wandb_out.update(metrics)
                wandb.log(wandb_out)

            self.metrics.reset()
            progress_bar.disable = False
            progress_bar.set_postfix_str(self.metric_output(metrics))

        progress_bar.close()
        self.model.train()

        return metrics

    def generate_labels(self, input_ids, mask=None, pad_id=-100):
        labels = input_ids.clone()
        labels[torch.logical_not(mask)] = pad_id

        return labels

    def add_to_metrics(self, logits, labels, type='token', measure_type=MeasureType.CATEGORICAL, **kwargs):
        if measure_type == MeasureType.CATEGORICAL:
            probs = logits.softmax(dim=-1)
            self.metrics.add(probs, labels, logits=logits, type=type, **kwargs)
        else:
            self.metrics.add(logits.reshape(-1), labels, type=type, **kwargs)

    def generate_projection_labels(self, time_until_ts, mask=None, pad_id=-100):
        labels = time_until_ts.clone()
        labels[torch.logical_not(mask)] = pad_id

        """
        labels[labels < 0.5] = 0
        labels[labels < 1] = 1
        labels[labels < 3] = 2
        labels[labels < 6] = 3
        labels[labels < 10] = 4
        labels[labels < 20] = 5
        labels[labels < 40] = 6
        labels[labels >= 40] = 7
        """

        return labels

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

    def plot_metric_graphs(self):
        graphs = self.metrics.get_graphs()

        for token, token_graph in graphs.items():
            wandb.log({
                token: [wandb.Image(ax) for k, (fig, ax) in token_graph.items()],
                'global_step': self.global_step
            })

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
        # self.model.tokenizer.save_pretrained(os.path.join(os.path.dirname(path), "tokenizer"))
        if self.save_model_allowed:
            self.logger.info(f"saving model at {path}")
            torch.save(self.best, path)

    def save_history(self, path):
        self.logger.info("trainer: save history")
        for key in self.train_history:
            try:
                np.save(os.path.join(
                    path, f"train_{key}"), self.train_history[key])
            except TypeError as e:
                print(f"Cannot save {key}")
        for key in self.test_history:
            try:
                np.save(os.path.join(
                    path, f"test_{key}"), self.test_history[key])
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

    @torch.no_grad
    def generate_on_validation_set(self, validation_text, mask=None, speaker_ids=None, name="Generate/Validation"):
        text_idx = validation_text.shape[1] // 2

        text = validation_text[:, :text_idx]
        speaker_id = speaker_ids[:, :text_idx]
        m = mask[:, :text_idx]

        out = self.model.generate(
            input_ids=text, speaker_ids=speaker_id, mask=m, output_scores=True, n_sequences=10)
        G = {"tokens": self.model.tokenizer.batch_decode(
            out['sequences'][:, len(text):])}

        input = self.model.tokenizer.batch_decode(text)
        ground_truth = self.model.tokenizer.batch_decode(
            validation_text[:, text_idx:])

        table = wandb.Table(
            columns=["context", "truth", "sample"],
            data=[
                [sentence, truth, sample]
                for sentence, truth, sample in zip(input, ground_truth, G["tokens"])
            ]
        )
        wandb.log({
            f"{name}": table,
            "global_step": self.global_step,
        })

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

    def trp_sample_plots(self, input_ids, out, name="TRP/test_sample"):
        figs = []
        global_steps = []

        special_tokens = [
            token_id for token_id in self.model.tokenizer.special_tokens if token_id != "<emp>"]
        special_tokens.append("<eot>")

        tokens = [self.model.tokenizer.convert_tokens_to_ids(
            token_id) for token_id in self.model.tokenizer.special_tokens if token_id != "<emp>"]
        tokens.append(self.model.tokenizer.convert_tokens_to_ids("<eot>"))

        logits = [out.logits[..., token_id].cpu() for token_id in tokens]

        step = 50
        for batch_idx in range(len(input_ids)):
            probs = [log.softmax(dim=-1)[batch_idx] for log in logits]
            for idx in range(0, len(input_ids[batch_idx]), step):
                pA = [x[idx:idx + step] for x in probs]
                fig, (_, ax) = plot_trp_pair(
                    trp=pA,
                    text=self.model.tokenizer.convert_ids_to_tokens(
                        input_ids[batch_idx][idx:idx + step]),
                    special_tokens=special_tokens,
                    eos_token='[SEP]'
                )
                figs.append(wandb.Image(fig))

        global_steps.append(self.global_step)
        wandb.log({f"{name}": figs, "global_step": self.global_step})

    def trp_example_plots(self, name="TRP/example", example=None):
        turn_list = [
            ["yesterday we met in the park",
                "okay when will you meet again", "tomorrow"],
            [
                "Hello there I basically had the worst day of my life",
                "Oh no, what happened?",
                "Do you want the long or the short story?",
            ],
        ]

        figs = []
        global_steps = []
        for b in range(len(turn_list)):
            out = self.model.from_string(turn_list[b])
            out = self.generate_from_string(out)
            fig, _ = plot_trp(
                trp=out["trp_probs"][0].cpu(),
                text=out["tokens"],
                eos_token='[SEP]'
            )
            figs.append(fig)
            global_steps.append(self.global_step)

        if example is not None:
            step = 50
            tokens = [self.model.tokenizer.convert_tokens_to_ids(
                token_id) for token_id in self.model.tokenizer.special_tokens if token_id != "<emp>"]
            logits = [example[1].logits[..., token_id].cpu()
                      for token_id in tokens]
            for batch_idx in range(len(example[0])):
                probs = [log.softmax(dim=-1)[batch_idx] for log in logits]
                for idx in range(0, len(example[batch_idx]), step):
                    p = [x[idx:idx + step] for x in probs]
                    fig, _ = plot_trp(
                        trp=p.cpu(),
                        text=out["tokens"],
                        eos_token='[SEP]',
                        probs=p
                    )
                    figs.append(fig)

        wandb.log({"graphs": [wandb.Image(im)
                  for im in figs], "global_step": self.global_step})

    def text_generation_examples(self, name="Generate/example"):
        turn_list = [
            ["yesterday we met in the park",
                "okay when will you meet again", "tomorrow"],
            [
                "Hello there I basically had the worst day of my life",
                "Oh no, what happened?",
                "Do you want the long or the short story?",
            ],
        ]

        inp = self.model.from_string(turn_list[-1])
        out = self.model.generate(
            input_ids=inp['input_ids'], speaker_ids=inp["token_type_ids"], output_scores=True, n_sequences=10)

        G = {"tokens": self.model.tokenizer.batch_decode(
            out['sequences'][:, len(inp['input_ids'][0]):])}
        """
        for i, g in enumerate(out["sequences"][1:]):
            if g not in G["tokens"]:
                G["tokens"].append(g)
                # G["probs"].append(out["probs"][i].cpu())
        """

        table = wandb.Table(
            columns=["context", "sample"],
            data=[
                [turn_list[-1][-1], toks]
                for toks in G["tokens"]
            ]
        )
        wandb.log({
            f"{name}": table,
            "global_step": self.global_step,
        })
