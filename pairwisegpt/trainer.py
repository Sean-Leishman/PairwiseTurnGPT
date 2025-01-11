import torch
import wandb
import matplotlib.pyplot as plt
import os
import json

from common.utils import get_logger, get_new_filename
from common.trainer import Trainer
from common.metrics import DefaultMetricBuilder

from data.base import TurnType, TurnEndType
from data.utils import get_abs_path

from pairwisegpt.utils import plot_trp

logger = get_logger(__name__)


class PairwiseTrainer(Trainer):
    def __init__(
        self,
        evaluate_on_full=False,
        remove_emp_metric_generation=False,
        end_of_utterance_tokens=["<eot>", "<ebc>", "<eint>", "<yield>"],
        metric_builder=DefaultMetricBuilder,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.evaluate_on_full = evaluate_on_full
        self.remove_emp_metric_generation = remove_emp_metric_generation
        if self.remove_emp_metric_generation:
            logger.warning("Removing <emp> token for metric generation")

        self.include_yield_token = "<yield>" in end_of_utterance_tokens
        self.include_ebc_token = "<ebc>" in end_of_utterance_tokens
        self.include_eint_token = "<eint>" in end_of_utterance_tokens

        self.serialise_data = self.config.serialise_data

        self.metrics = metric_builder(
            self.model.tokenizer,
            device=self.device,
            include_yield_token=self.include_yield_token,
        )

    def init_save_path(self, save_path, config):
        """
        Hacky approach to save config in pairwisegpt folder rather than the
        common folder by calling from the common trainer in their __init__
        """
        self.save_path = get_abs_path(
            get_new_filename(save_path)
        )  # path would be common/{save_path}
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        with open(os.path.join(self.save_path, "config.json"), "w") as file:
            json.dump(vars(config), file)

        return self.save_path

    def _step(self, batch):
        batchA = self._extract_batch(batch, speaker_key="speakerA")
        batchB = self._extract_batch(batch, speaker_key="speakerB")

        labelsA = self._generate_labels(batchA["input_ids"], batchA["attention_mask"])
        labelsB = self._generate_labels(batchB["input_ids"], batchB["attention_mask"])

        out = self.model(
            input_idsA=batchA["input_ids"],
            attention_maskA=batchA["attention_mask"],
            labelsA=labelsA,
            token_type_idsA=batchA["speaker_ids"],
            input_idsB=batchB["input_ids"],
            attention_maskB=batchB["attention_mask"],
            labelsB=labelsB,
            token_type_idsB=batchB["speaker_ids"],
        )

        return out

    def _eval_step(self, batch, out, inference_saver=None, target_token=-100):
        batchA = self._extract_batch(batch, speaker_key="speakerA")
        batchB = self._extract_batch(batch, speaker_key="speakerB")

        assert "input_ids" in batchA, "input_ids must be in batchA"
        assert "input_ids" in batchB, "input_ids must be in batchB"

        assert "attention_mask" in batchA, "attention_mask must be in batchA"
        assert "attention_mask" in batchB, "attention_mask must be in batchB"

        ignore_maskA = batchA["attention_mask"].to(self.device).detach().clone()
        ignore_maskB = batchB["attention_mask"].to(self.device).detach().clone()

        if self.remove_emp_metric_generation:
            ignore_maskA = torch.logical_and(
                ignore_maskA, torch.ne(batchA["input_ids"], self.tokens_dict["<emp>"])
            )
            ignore_maskB = torch.logical_and(
                ignore_maskB, torch.ne(batchB["input_ids"], self.tokens_dict["<emp>"])
            )

        batch = {f"{k}A": v for k, v in batchA.items()} | {
            f"{k}B": v for k, v in batchB.items()
        }
        metric_masksA, metric_masksB, _ = self._generate_masks(
            **batch, ignore_maskA=ignore_maskA, ignore_maskB=ignore_maskB
        )

        probsA = self._add_to_metrics(
            out.logits[0].detach(), batchA["input_ids"].detach(), **metric_masksA
        )
        probsB = self._add_to_metrics(
            out.logits[1].detach(), batchB["input_ids"].detach(), **metric_masksB
        )

        if inference_saver is not None:
            inference_saver.save(
                probsA[..., target_token],
                batchA["turn_ids"],
                batchA["input_ids"].detach(),
            )
            inference_saver.save(
                probsB[..., target_token],
                batchB["turn_ids"],
                batchB["input_ids"].detach(),
            )

    def _plot_trp_to_wandb(self, dl, name="TRP/example", *args, **kwargs):
        ds = next(iter(dl))

        out = self._step(ds)

        input_idsA = ds["speakerA"]["input_ids"]
        input_idsB = ds["speakerB"]["input_ids"]

        figs = []
        global_steps = []

        special_tokens = [
            token_id
            for token_id in self.model.tokenizer.special_tokens
            if token_id != "<emp>"
        ]
        special_tokens.append("<eot>")

        tokens = [
            self.model.tokenizer.convert_tokens_to_ids(token_id)
            for token_id in self.model.tokenizer.special_tokens
            if token_id != "<emp>"
        ]
        tokens.append(self.model.tokenizer.convert_tokens_to_ids("<eot>"))

        raw_probsA = [
            out.logits[0].softmax(dim=-1)[..., token_id].cpu() for token_id in tokens
        ]
        raw_probsB = [
            out.logits[1].softmax(dim=-1)[..., token_id].cpu() for token_id in tokens
        ]

        step = 50
        for batch_idx in range(len(input_idsA)):
            probsA = [prob[batch_idx] for prob in raw_probsA]
            probsB = [prob[batch_idx] for prob in raw_probsB]

            for idx in range(0, len(input_idsA[batch_idx]), step):
                pA = [x[idx : idx + step] for x in probsA]
                fig, (_, ax) = plot_trp(
                    trp=pA,
                    text=self.model.tokenizer.convert_ids_to_tokens(
                        input_idsA[batch_idx][idx : idx + step]
                    ),
                    eos_token="[SEP]",
                    special_tokens=special_tokens,
                )
                pB = [x[idx : idx + step] for x in probsB]
                _, _ = plot_trp(
                    trp=pB,
                    text=self.model.tokenizer.convert_ids_to_tokens(
                        input_idsB[batch_idx][idx : idx + step]
                    ),
                    eos_token="[SEP]",
                    special_tokens=special_tokens,
                    fig=fig,
                    ax=ax,
                )
                figs.append(wandb.Image(fig))
                plt.close("all")
        global_steps.append(self.global_step)

        wandb.log({f"{name}": figs, "global_step": self.global_step})

    def _generate_masks(
        self,
        input_idsA,
        input_idsB,
        token_type_idsA,
        token_type_idsB,
        other_token_type_idsA,
        other_token_type_idsB,
        ignore_maskA,
        ignore_maskB,
        *args,
        **kwargs,
    ):
        overlap_maskA, overlap_maskB = None, None
        non_overlap_maskA, non_overlap_maskB = None, None
        interrupt_maskA, interrupt_maskB = None, None
        yield_maskA, yield_maskB = None, None
        non_yield_maskA, non_yield_maskB = None, None
        turn_masksA, turn_masksB = None, None

        mask_specialA, mask_specialB = None, None

        interrupt_mask_normalA, interrupt_mask_normalB = None, None
        interrupt_mask_bcA, interrupt_mask_bcB = None, None
        interrupt_mask_overlapA, interrupt_mask_overlapB = None, None
        interrupt_mask_yieldA, interrupt_mask_yieldB = None, None

        eot_typesA = other_token_type_idsA
        eot_typesB = other_token_type_idsB

        eot_token_id = self.tokens_dict["<eot>"]
        yield_token_id = self.tokens_dict["<eot>"]
        if self.include_yield_token:
            yield_token_id = self.tokens_dict["<yield>"]

        if self.evaluate_on_full or not self.serialise_data:
            # Overlapping where:
            #   A is YIELD and B is INTERRUPTING (include <eot> of A in mask)
            #   B is YIELD and A is INTERRUPTIN)           #   A is speaking AND B is OVERLAP
            #   B is speaking AND A is OVERLAP
            #   Avoids A ending turn and B starting after and being labelled as OVERLAP if in line with <eot>
            yield_int_overlap = torch.logical_or(
                torch.logical_and(
                    token_type_idsB == TurnType.INTERRUPT,
                    eot_typesA == TurnEndType.YIELD,
                ),
                torch.logical_and(
                    token_type_idsA == TurnType.INTERRUPT,
                    eot_typesB == TurnEndType.YIELD,
                ),
            )

            overlap_mask = torch.logical_or(
                token_type_idsA == TurnType.OVERLAP, token_type_idsB == TurnType.OVERLAP
            )
            overlap_maskA = torch.logical_or(yield_int_overlap, overlap_mask)
            overlap_maskB = torch.logical_or(yield_int_overlap, overlap_mask)

            # Non-overlap where only one content token is occuring at a time
            non_overlap_maskA = torch.logical_and(
                token_type_idsA != TurnType.NONE, overlap_mask == TurnType.NONE
            )
            non_overlap_maskB = torch.logical_and(
                token_type_idsA == TurnType.NONE, token_type_idsB != TurnType.NONE
            )
            non_overlap_maskA = torch.logical_or(non_overlap_maskA, non_overlap_maskB)
            non_overlap_maskB = torch.logical_or(non_overlap_maskA, non_overlap_maskB)
            # overlap_maskA = torch.logical_or(overlap_maskA, overlap_maskB)
            # overlap_maskB = overlap_maskA.detach().clone()
            # Make sure that for A consider overlap as where turn ends during overlap too
            # But not where A is ending turn normally

            non_yield_maskA = torch.logical_not(
                torch.logical_and(
                    torch.eq(eot_typesA, TurnEndType.YIELD),
                    torch.eq(input_idsA, yield_token_id),
                )
            )
            non_yield_maskB = (
                torch.logical_not(  # all tokens except yields in yielded turns
                    torch.logical_and(  # every <yield> in a yielded turn
                        torch.eq(
                            eot_typesB, TurnEndType.YIELD
                        ),  # every token in a yielded turn
                        torch.eq(input_idsB, yield_token_id),  # every <yield>
                    )
                )
            )

            # Every token aside from <eot> that is not YIELD <eot>
            yield_maskA = torch.logical_not(  # every token in all turns except <eot> tokens not in yielded turns
                torch.logical_and(  # every <eot> that is in not in a yielded turn
                    torch.ne(
                        eot_typesA, TurnEndType.YIELD
                    ),  # every token not in yielded turn
                    torch.eq(input_idsA, eot_token_id),  # every <eot>
                )
            )
            yield_maskB = torch.logical_not(
                torch.logical_and(
                    torch.ne(eot_typesB, TurnEndType.YIELD),
                    torch.eq(input_idsB, eot_token_id),
                )
            )

            # Finds all areas where there is one content token and then rolls that tensor
            # to include the first token of the utterance
            # We then divide first token of utterance based on its turn type
            interrupt_maskA = torch.logical_and(
                torch.eq(token_type_idsA, TurnType.NONE),
                torch.ne(token_type_idsB, TurnType.NONE),
            )
            interrupt_maskB = torch.logical_and(
                torch.ne(token_type_idsA, TurnType.NONE),
                torch.eq(token_type_idsB, TurnType.NONE),
            )
            interrupt_maskA = torch.logical_or(
                interrupt_maskA, interrupt_maskA.roll(1, dims=-1)
            )
            interrupt_maskB = torch.logical_or(
                interrupt_maskB, interrupt_maskB.roll(1, dims=-1)
            )
            """
            For backchannel
            interrupt_maskA = torch.logical_and(
                interrupt_maskA, token_type_idsA != TurnType.BACKCHANNEL)
            interrupt_maskB = torch.logical_and(
                interrupt_maskB, token_type_idsB != TurnType.BACKCHANNEL)
            """
            interrupt_mask_normalA = torch.logical_and(
                interrupt_maskA,
                torch.logical_or(
                    token_type_idsA == TurnType.NONE, token_type_idsA == TurnType.NORMAL
                ),
            )
            interrupt_mask_normalB = torch.logical_and(
                interrupt_maskB,
                torch.logical_or(
                    token_type_idsB == TurnType.NONE, token_type_idsB == TurnType.NORMAL
                ),
            )
            interrupt_mask_bcA = torch.logical_and(
                interrupt_maskA,
                torch.logical_or(
                    token_type_idsA == TurnType.NONE,
                    token_type_idsA == TurnType.BACKCHANNEL,
                ),
            )
            interrupt_mask_bcB = torch.logical_and(
                interrupt_maskB,
                torch.logical_or(
                    token_type_idsB == TurnType.NONE,
                    token_type_idsB == TurnType.BACKCHANNEL,
                ),
            )
            interrupt_mask_overlapA = torch.logical_and(
                interrupt_maskA,
                torch.logical_or(
                    token_type_idsA == TurnType.NONE,
                    token_type_idsA == TurnType.OVERLAP,
                ),
            )
            interrupt_mask_overlapB = torch.logical_and(
                interrupt_maskB,
                torch.logical_or(
                    token_type_idsB == TurnType.NONE,
                    token_type_idsB == TurnType.OVERLAP,
                ),
            )
            interrupt_mask_yieldA = torch.logical_and(
                interrupt_maskA,
                torch.logical_or(
                    token_type_idsA == TurnType.NONE,
                    token_type_idsA == TurnType.INTERRUPT,
                ),
            )
            interrupt_mask_yieldB = torch.logical_and(
                interrupt_maskB,
                torch.logical_or(
                    token_type_idsB == TurnType.NONE,
                    token_type_idsB == TurnType.INTERRUPT,
                ),
            )

            turn_normal = torch.tensor(
                [TurnType.NORMAL, TurnType.INTERRUPT], device=self.device
            )
            turn_masksA = torch.isin(token_type_idsA, turn_normal)
            turn_masksB = torch.isin(token_type_idsB, turn_normal)

        elif (
            not self.evaluate_on_full and self.self.serialise_data
        ):  # if evaluating on full should be parsed above and should only proceed here with serialised data
            # Remove where <eot> tokens appear
            overlap_maskA = torch.logical_or(
                torch.eq(eot_typesA, TurnEndType.YIELD),
                torch.eq(eot_typesA, TurnType.OVERLAP),
            )

            overlap_maskB = torch.logical_or(
                torch.eq(eot_typesB, TurnEndType.YIELD),
                torch.eq(eot_typesB, TurnType.OVERLAP),
            )
            overlap_mask = torch.logical_or(overlap_maskA, overlap_maskB)
            overlap_maskA = overlap_mask.clone().detach()
            overlap_maskB = overlap_mask.clone().detach()

            yield_maskA = torch.logical_not(
                torch.logical_and(
                    torch.ne(eot_typesA, TurnEndType.YIELD),
                    torch.eq(input_idsA, eot_token_id),
                )
            )
            yield_maskB = torch.logical_not(
                torch.logical_and(
                    torch.ne(eot_typesB, TurnEndType.YIELD),
                    torch.eq(input_idsB, eot_token_id),
                )
            )

            non_yield_maskA = torch.logical_not(
                torch.logical_and(
                    torch.eq(eot_typesA, TurnEndType.YIELD),
                    torch.eq(input_idsA, yield_token_id),
                )
            )
            non_yield_maskB = torch.logical_not(
                torch.logical_and(
                    torch.eq(eot_typesB, TurnEndType.YIELD),
                    torch.eq(input_idsB, yield_token_id),
                )
            )

            turn_masksA = torch.ne(token_type_idsA, TurnType.NONE)
            turn_masksB = torch.ne(token_type_idsB, TurnType.NONE)
        else:
            raise ValueError("Missing case")

        ignore_maskA = ignore_maskA.detach().clone()
        ignore_maskB = ignore_maskB.detach().clone()

        metric_masksA = dict(
            overlap_mask=overlap_maskA,
            non_overlap_mask=non_overlap_maskA,
            yield_mask=yield_maskA,
            non_yield_mask=non_yield_maskA,
            interrupt_mask=interrupt_maskA,
            turn_mask=turn_masksA,
            ignore_mask=ignore_maskA,
            mask_special=mask_specialA,
            interrupt_mask_normal=interrupt_mask_normalA,
            interrupt_mask_bc=interrupt_mask_bcA,
            interrupt_mask_overlap=interrupt_mask_overlapA,
            interrupt_mask_yield=interrupt_mask_yieldA,
        )
        metric_masksB = dict(
            overlap_mask=overlap_maskB,
            non_overlap_mask=non_overlap_maskB,
            yield_mask=yield_maskB,
            non_yield_mask=non_yield_maskB,
            interrupt_mask=interrupt_maskB,
            turn_mask=turn_masksB,
            ignore_mask=ignore_maskB,
            mask_special=mask_specialB,
            interrupt_mask_normal=interrupt_mask_normalB,
            interrupt_mask_bc=interrupt_mask_bcB,
            interrupt_mask_overlap=interrupt_mask_overlapB,
            interrupt_mask_yield=interrupt_mask_yieldB,
        )
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
            interrupt_mask_yieldB=interrupt_mask_yieldB,
        )
        return metric_masksA, metric_masksB, all_metric_masks
