import json
import os
import pprint
import torch
import wandb

from data.base import TurnEndType, TurnType
from common.utils import get_latest_model, get_logger, get_abs_path, get_new_filename
from common.metrics import DefaultMetricBuilder
from pairwisegpt.utils import plot_trp
from tqdm import tqdm

from common.inference_saver import DefaultInferenceSaver, InferenceSaver

logger = get_logger(__name__)

default_inference_saver = DefaultInferenceSaver()


class Trainer:
    def __init__(
        self,
        model,
        optimizer,
        config,
        load_from_checkpoint=None,
        save_model_allowed=True,
        metric_builder=DefaultMetricBuilder,
        **kwargs,
    ):
        self.model = model
        self.optimizer = optimizer
        self.config = config
        self.load_from_checkpoint = load_from_checkpoint
        self.save_model_allowed = save_model_allowed

        self.early_stop = config.early_stop
        self.num_of_epochs = config.epochs

        self.device = torch.device(config.device)
        self.model.to(self.device)

        self.last_best_path = None
        self.best_loss = float("inf")
        self.best = {
            "epoch": 0,
            "global_step": 0,
            "loss": float("inf"),
            "model_state_dict": None,
            "optimizer_state_dict": None,
        }

        self.log_interval = config.log_interval

        self.epoch = 0
        self.global_step = 0
        self.dev_mode = config.dev_mode

        self.train_history = {}
        self.val_history = {}
        self.test_history = {}

        self.init_save_path(config.save_path, config)

        if self.load_from_checkpoint is not None:
            self.load_checkpoint()

        self.metrics = metric_builder(self.model.tokenizer, device=self.device)
        self.val_inference_saver = InferenceSaver(
            os.path.join(self.save_path, "val_inference.csv")
        )
        self.test_inference_saver = InferenceSaver(
            os.path.join(self.save_path, "test_inference.csv")
        )

        self.tokens_dict = {
            k: self.model.tokenizer.convert_tokens_to_ids(k)
            for k in self.model.special_tokens
        }
        self.tokens_dict["<eot>"] = self.model.tokenizer.convert_tokens_to_ids("<eot>")

        self.metrics = metric_builder(self.model.tokenizer, device=self.device)

    def init_save_path(self, save_path, config):
        self.save_path = get_abs_path(
            get_new_filename(save_path)
        )  # path would be common/{save_path}
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        with open(os.path.join(self.save_path, "config.json"), "w") as file:
            json.dump(vars(config), file)

        return self.save_path

    def set_metrics(self, metric_builder):
        self.metrics = metric_builder(self.model.tokenizer, device=self.device)
        return self

    def load_checkpoint(self):
        try:
            checkpoint = self.model.load_from_checkpoint(self.load_from_checkpoint)
        except Exception:
            self.load_model_file = get_latest_model(
                os.path.dirname(self.load_from_checkpoint),
                before=self.load_from_checkpoint,
            )
            self.load_checkpoint()
        else:
            logger.info(f"model: loading parameters for model {checkpoint.keys()}")
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.epoch = checkpoint["epoch"]
            self.global_step = checkpoint["global_step"]

    def train(self, train_ds, val_ds, test_ds, scheduler, dataset_loader=None):
        best_loss = float("inf")

        not_improved = 0
        progress_bar = tqdm(range(self.epoch, self.num_of_epochs), desc="Epochs")

        self.scheduler = scheduler

        """
        TODO: Check what train() does here
        """
        self.model.train()
        for epoch in progress_bar:
            _ = self.train_epoch(train_ds, dataset_loader=dataset_loader)
            val_metrics = self.evaluate_epoch(
                val_ds,
                inference_saver=self.val_inference_saver,
                dataset_loader=dataset_loader,
                is_val=True,
            )
            test_metrics = self.evaluate_epoch(
                test_ds,
                inference_saver=self.test_inference_saver,
                dataset_loader=dataset_loader,
                is_val=False,
            )

            avg_val_loss = val_metrics["avg_loss"]
            for key, value in val_metrics.items():
                self.val_history.setdefault(key, []).append(value)
            for key, value in test_metrics.items():
                self.test_history.setdefault(key, []).append(value)

            self._save_history(self.save_path)
            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                not_improved = 0

                self.best["model_state_dict"] = self.model.state_dict()
                self.best["optimizer_state_dict"] = self.optimizer.state_dict()
                self.best["epoch"] = epoch
                self.best["global_step"] = self.global_step
                self.best["loss"] = best_loss

                model_name = f"model_{epoch}_{avg_val_loss:.4f}.pt"
                self._save_training(os.path.join(self.save_path, model_name))
                self._prune_training()

                self.last_best_path = os.path.join(self.save_path, model_name)

                logger.info(f"saving model at {self.last_best_path} for epoch {epoch}")
            else:
                not_improved += 1

            if not_improved >= self.early_stop:
                logger.info(f"Early stopping at epoch {epoch}")
                progress_bar.close()
                return self.train_history, self.val_history, self.test_history

            self.epoch += 1

        progress_bar.close()
        return self.train_history, self.val_history, self.test_history

    def evaluate(self, val_ds, test_ds, dataset_loader=None):
        val_metrics = self.evaluate_epoch(
            val_ds,
            inference_saver=self.val_inference_saver,
            dataset_loader=dataset_loader,
            is_val=True,
        )
        test_metrics = self.evaluate_epoch(
            test_ds,
            inference_saver=self.test_inference_saver,
            dataset_loader=dataset_loader,
            is_val=False,
        )

        for key, value in val_metrics.items():
            self.val_history.setdefault(key, []).append(value)
        for key, value in test_metrics.items():
            self.test_history.setdefault(key, []).append(value)

        self._save_history(self.save_path)

        return self.val_history, self.test_history

    def train_epoch(self, train_ds, dataset_loader=None):
        self.model.train()

        train_loss = 0
        train_ds.setup()
        train_dl = dataset_loader(train_ds)

        progress_bar = tqdm(train_dl, desc="Training", unit="batch")

        for step, batch in enumerate(progress_bar):
            self.optimizer.zero_grad()

            out = self._step(batch)
            out.loss.backward()
            self.optimizer.step()

            if step % self.log_interval == 0 and not self.dev_mode:
                wandb.log({"loss": out.loss.item(), "step": self.global_step})

            train_loss += out.loss.item()
            avg_train_loss = round(train_loss / (step + 1), 4)
            progress_bar.set_postfix_str(f"avg_train_loss={avg_train_loss}")
            self.global_step += 1

        avg_loss = train_loss / len(train_dl)

        metrics = {"avg_train_loss": avg_loss}
        wandb.log({"train_loss": avg_loss, "epoch": self.epoch})

        progress_bar.set_postfix_str(self._metric_to_str(metrics))
        progress_bar.close()

        train_ds.reset()

        return metrics

    def evaluate_epoch(
        self,
        ds,
        inference_saver=default_inference_saver,
        dataset_loader=None,
        is_val=True,
    ):
        assert dataset_loader is not None, "dataset_loader must be provided"

        self.model.eval()

        ds.setup()
        dl = dataset_loader(ds)

        prefix_name = "val" if is_val else "test"

        self.metrics.set_prefix(prefix_name)
        inference_saver.start(
            filename=os.path.join(
                self.save_path, f"{prefix_name}_inference_{self.epoch}.txt"
            )
        )

        loss = 0
        progress_bar = tqdm(dl, desc="Validation" if is_val else "Test", unit="batch")

        with torch.no_grad():
            for step, batch in enumerate(progress_bar):
                out = self._step(batch)
                loss += out.loss.item()

                self._eval_step(
                    batch,
                    out,
                    inference_saver=inference_saver,
                    target_token=self.tokens_dict["<eot>"],
                )

                avg_loss = round(loss / (step + 1), 4)
                progress_bar.set_postfix_str(f"avg_loss={avg_loss}")

            metrics, _ = self.metrics.calculate(set_param=is_val, use_param=not is_val)
            metrics["avg_loss"] = loss / len(dl)

            if not self.dev_mode or self.dev_mode:
                self._plot_metric_to_wandb()
                wandb_out = {
                    "val_loss" if is_val else "test_loss": metrics["avg_loss"],
                    "epoch": self.epoch,
                }

                wandb_out.update(metrics)
                wandb.log(wandb_out)

                self._plot_trp_to_wandb(dl)

            self.metrics.reset(reset_params=not is_val)
            progress_bar.set_postfix_str(self._metric_to_str(metrics))

        avg_loss = loss / len(dl)
        metrics["avg_loss"] = avg_loss

        inference_saver.shutdown()

        progress_bar.set_postfix_str(self._metric_to_str(metrics))
        progress_bar.close()

        ds.reset()

        return metrics

    def _step(self, batch):
        batch = self._extract_batch(batch, speaker_key="speakerA")
        labels = self._generate_labels(batch["input_ids"], batch["attention_mask"])

        out = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=labels,
            token_type_ids=batch["speaker_ids"],
        )

        return out

    def _eval_step(self, batch, out, inference_saver=None, target_token=-100):
        batch = self._extract_batch(batch, speaker_key="speakerA")

        ignore_mask = batch["attention_mask"].to(self.device).detach().clone()
        metric_masks = self._generate_masks(**batch, ignore_mask=ignore_mask)

        probs = self._add_to_metrics(
            out.logits.detach(), batch["input_ids"].detach(), **metric_masks
        )

        if inference_saver is not None:
            inference_saver.save(
                probs[..., target_token],
                batch["turn_ids"],
                batch["input_ids"],
            )

    def _extract_batch(self, batch, speaker_key):
        if speaker_key not in batch:
            raise ValueError(f"{speaker_key} not in batch")

        batch = {k: v for k, v in batch[speaker_key].items()}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(self.device)

        return batch

    def _add_to_metrics(self, logits, labels, **kwargs):
        probs = logits.softmax(dim=-1)
        self.metrics.add(probs, labels, logits=logits, **kwargs)

        return probs

    def _generate_labels(self, input_ids, mask=None, pad_id=-100):
        labels = input_ids.detach().clone()
        if mask is None:
            return labels

        labels[torch.logical_not(mask)] = pad_id
        return labels

    def _generate_masks(
        self,
        input_ids,
        token_type_ids,
        other_token_type_ids,
        ignore_mask=None,
        *args,
        **kwargs,
    ):
        if ignore_mask is None:
            ignore_mask = torch.ones_like(input_ids, dtype=torch.bool)

        overlap_mask = torch.logical_or(
            token_type_ids == TurnType.OVERLAP,
            other_token_type_ids == TurnEndType.YIELD,
        )
        yield_mask = torch.logical_not(
            torch.logical_and(
                torch.eq(other_token_type_ids, TurnEndType.YIELD),
                torch.eq(input_ids, self.tokens_dict["<eot>"]),
            )
        )

        return {
            "ignore_mask": ignore_mask,
            "overlap_mask": overlap_mask,
            "yield_mask": yield_mask,
        }

    def _metric_to_str(self, metrics):
        output = ""
        if "avg_loss" in metrics:
            output += f"avg_loss={metrics['avg_loss']:.4f}"

        return output

    def _save_training(self, path):
        if self.save_model_allowed:
            logger.info(f"saving model at {path}")
            torch.save(
                self.best,
                path,
            )

    def _prune_training(self):
        if self.last_best_path is not None:
            if os.path.exists(self.last_best_path):
                os.remove(self.last_best_path)
                logger.info(f"pruning model at {self.last_best_path}")

    def _save_history(self, path):
        logger.info("saving history")

        self.__save_history(path, self.train_history, self.epoch, filename="train.json")
        self.__save_history(path, self.val_history, self.epoch, filename="val.json")
        self.__save_history(path, self.test_history, self.epoch, filename="test.json")

    def __save_history(self, path, history, epoch, filename="val.json"):
        filename = os.path.join(path, filename)

        pprint.pp(history)

        to_write = []
        for key in history.keys():
            for epoch, value in enumerate(history[key]):
                epoch += 1
                if epoch > len(to_write):
                    to_write.append({})
                    to_write[epoch - 1]["epoch"] = epoch

                to_write[epoch - 1][key] = value

        with open(filename, "w") as file:
            json.dump(to_write, file)

    def _plot_metric_to_wandb(self):
        graphs = self.metrics.plot()
        for token, graph in graphs.items():
            wandb.log(
                {
                    token: [wandb.Image(ax) for _, ax in graph.values()],
                    "global_step": self.global_step,
                }
            )

    def _plot_trp_to_wandb(self, *args, **kwargs):
        turn_list = [
            [
                "yesterday we met in the park",
                "okay when will you meet again",
                "tomorrow",
            ],
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
                trp=out["trp_probs"][0].cpu(), text=out["tokens"], eos_token="[SEP]"
            )
            figs.append(fig)
            global_steps.append(self.global_step)

        if example is not None:
            step = 50
            tokens = [
                self.model.tokenizer.convert_tokens_to_ids(token_id)
                for token_id in self.model.tokenizer.special_tokens
                if token_id != "<emp>"
            ]
            logits = [example[1].logits[..., token_id].cpu() for token_id in tokens]
            for batch_idx in range(len(example[0])):
                probs = [log.softmax(dim=-1)[batch_idx] for log in logits]
                for idx in range(0, len(example[batch_idx]), step):
                    p = [x[idx : idx + step] for x in probs]
                    fig, _ = plot_trp(
                        trp=p.cpu(), text=out["tokens"], eos_token="[SEP]", probs=p
                    )
                    figs.append(fig)

        wandb.log(
            {
                "graphs": [wandb.Image(im) for im in figs],
                "global_step": self.global_step,
            }
        )
