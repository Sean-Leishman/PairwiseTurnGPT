import torch
import logging

from common.metrics.turn_metrics import EndOfTurnMetric, StartOfTurnMetric
from common.metrics.metrics import MetricContainer
from common.metrics.bacc import BACC
from common.metrics.ppl import PPL
from common.metrics.f1 import F1

from data import TurnType, TurnEndType


logger = logging.getLogger(__name__)


def BuildMetrics(
    tokenizer,
    include_yield_token=False,
    split: str = "val",
    eot_metrics=[BACC, F1],
    device="cuda:0",
):
    metrics = []
    eot_token_id = torch.tensor([tokenizer.convert_tokens_to_ids("<eot>")])
    yield_token_id = torch.tensor([tokenizer.convert_tokens_to_ids("<yield>")])
    bc_token_id = torch.tensor([tokenizer.convert_tokens_to_ids("<bc>")])
    eint_token_id = torch.tensor([tokenizer.convert_tokens_to_ids("<eint>")])
    ebc_token_id = torch.tensor([tokenizer.convert_tokens_to_ids("<ebc>")])
    empty_token_id = torch.tensor([tokenizer.convert_tokens_to_ids("<emp>")])

    special_tokens = torch.tensor(
        [
            eot_token_id,
            yield_token_id,
            bc_token_id,
            eint_token_id,
            ebc_token_id,
            empty_token_id,
        ]
    )
    special_tokens = special_tokens[torch.ne(special_tokens, tokenizer.pad_token_id)]
    end_of_utterance_tokens = torch.tensor(
        [eot_token_id, yield_token_id, ebc_token_id, eint_token_id]
    )
    end_of_utterance_tokens = end_of_utterance_tokens[
        torch.ne(end_of_utterance_tokens, tokenizer.pad_token_id)
    ]

    logger.info(
        f"Buidling metrics for {split} with the following special tokens: {tokenizer.convert_ids_to_tokens(special_tokens)} and end of utterance tokens: {tokenizer.convert_ids_to_tokens(end_of_utterance_tokens)}"
    )

    turn_types = [
        TurnType.NONE,
        TurnEndType.NORMAL,
        TurnEndType.YIELD,
        TurnType.OVERLAP,
        TurnType.BACKCHANNEL,
    ]

    for eot_metric in eot_metrics:
        for turn_type in turn_types:
            metrics.append(
                StartOfTurnMetric(
                    token_ids=special_tokens,
                    config=turn_type,
                    metric=eot_metric,
                    device=device,
                )
            )
    if include_yield_token:
        end_of_turn = torch.cat([eot_token_id, yield_token_id])

        for metric in eot_metrics:
            metrics.append(
                EndOfTurnMetric(
                    token_ids=end_of_utterance_tokens,
                    tokens="all",
                    config=TurnEndType.NONE,
                    metric=metric,
                    device=device,
                )
            )
            metrics.append(
                EndOfTurnMetric(
                    token_ids=eot_token_id,
                    tokens="eot",
                    config=TurnType.NONE,
                    metric=metric,
                    device=device,
                )
            )
            metrics.append(
                EndOfTurnMetric(
                    token_ids=yield_token_id,
                    tokens="yield",
                    config=TurnType.NONE,
                    metric=metric,
                    device=device,
                )
            )

            for turn_type in [TurnEndType.NONE, TurnEndType.NORMAL, TurnEndType.YIELD]:
                metrics.append(
                    EndOfTurnMetric(
                        token_ids=end_of_turn,
                        tokens="eotyield",
                        config=turn_type,
                        num_classes=2,
                        task="binary",
                        metric=metric,
                        device=device,
                    )
                )
    else:
        for metric in eot_metrics:
            metrics.append(
                EndOfTurnMetric(
                    token_ids=end_of_utterance_tokens,
                    tokens="all",
                    config=TurnEndType.NONE,
                    metric=metric,
                    device=device,
                )
            )

            for turn_type in [TurnEndType.NONE, TurnEndType.NORMAL, TurnEndType.YIELD]:
                metrics.append(
                    EndOfTurnMetric(
                        token_ids=eot_token_id,
                        tokens="eot",
                        config=turn_type,
                        metric=metric,
                        device=device,
                    )
                )

    metrics.append(PPL(turn_type=TurnType.NONE, device=device))
    metrics.append(PPL(turn_type=TurnType.NON_OVERLAP, device=device))
    metrics.append(PPL(turn_type=TurnType.OVERLAP, device=device))

    return MetricContainer(metrics, prefix=split)


if __name__ == "__main__":
    metric = StartOfTurnMetric(
        token_ids=torch.tensor([0], device="cuda:0"),
        config=TurnEndType.NONE,
        device="cuda:0",
        metric=F1,
    )
    metric.add(
        torch.tensor(
            [[[0.1, 0.8, 0.1], [0.8, 0.1, 0.1], [0.5, 0.1, 0.1], [0.8, 0.1, 0.1]]],
            device="cuda:0",
        ),
        torch.tensor([[0, 1, 2, 0]], device="cuda:0"),
    )
