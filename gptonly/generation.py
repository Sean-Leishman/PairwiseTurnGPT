from typing import Tuple

import einops
import torch
from torch import Tensor


def expand_batch(batch, n_trajectories=1):
    if n_trajectories > 1:
        batch["input_ids"] = einops.repeat(
            batch["input_ids"], "b n -> (r b) n", r=n_trajectories
        )
        batch["token_type_ids"] = einops.repeat(
            batch["token_type_ids"], "b n -> (r b) n", r=n_trajectories
        )
    return batch


def topk_sampling(logits: Tensor, top_k: int = 10) -> Tuple[Tensor, Tensor]:
    probs = logits.softmax(dim=-1)
    probs, token_idx = probs.sort(dim=-1, descending=True)
    probs, token_idx = probs[:, :top_k], token_idx[:, :top_k]
    p_idx = torch.multinomial(probs, num_samples=1).squeeze(1)  # B
    prob = probs[torch.arange(len(p_idx)), p_idx]
    tok = token_idx[torch.arange(len(p_idx)), p_idx]
    return tok, prob


def topp_sampling(
    logits: Tensor, top_p: float = 0.9
) -> Tuple[torch.Tensor, torch.Tensor]:
    probs = logits.softmax(dim=-1)
    probs, token_idx = probs.sort(dim=-1, descending=True)

    pcum = probs.cumsum(dim=-1)  # cumulative probabilities
    # cumulative less than or equal to `top_p`
    p_batch, p_idx = torch.where(pcum <= top_p)
    psmall = probs[p_batch, p_idx]
    tsmall = token_idx[p_batch, p_idx]

    # the cumulative probability distribution may not be of the same size
    # so we must sample for the batches individually
    next_token = torch.zeros((probs.shape[0]), dtype=torch.long, device=logits.device)
    next_prob = torch.zeros((probs.shape[0]), device=logits.device)
    for n_batch in range(probs.shape[0]):
        eq = p_batch == n_batch
        if eq.sum() == 0:  # first token is more likely than top_p
            next_tok = token_idx[n_batch, 0].item()
            next_p = probs[n_batch, 0].item()
        else:
            p = psmall[eq]
            t = tsmall[eq]
            p_idx = torch.multinomial(p, num_samples=1)  # B
            next_p = p[p_idx].item()
            next_tok = t[p_idx].item()
        next_token[n_batch] = next_tok
        next_prob[n_batch] = next_p
    return next_token, next_prob


def sample_next_token(
    logits: Tensor, top_p: float = -1.0, top_k: int = -1
) -> Tuple[Tensor, Tensor]:
    """
    Samples the next token given the probabilities over the
    """
    assert top_p > 0 or top_k > 0, "Either top_p or top_k > 0."
    if top_k > 0:
        next_token, next_prob = topk_sampling(logits, top_k)
    else:
        next_token, next_prob = topp_sampling(logits, top_p)
    return next_token, next_prob


def update_speaker_ids(batch, tokenizer) -> Tensor:
    """
    Correct Next Speaker (i.e. `token_type_ids`)
    Check for EOS-token. Don't change on EOS-token ("<ts>") but at the next step
    That is we check the second to last entry
    """

    def _change_speakers(last_speaker, indices, tokenizer):
        s1, s2 = tokenizer.sp1_token_id, tokenizer.sp2_token_id
        for ind in indices:
            last_speaker[ind] = s1 if last_speaker[ind] == s2 else s2
        return last_speaker

    next_speaker = batch["token_type_ids"][:, -1]
    eos_tokens = batch["input_ids"][:, -1] == tokenizer.eos_token_id
    if any(eos_tokens):
        change_speaker_idx = torch.where(eos_tokens)[0]
        next_speaker = _change_speakers(
            next_speaker.clone(), change_speaker_idx, tokenizer
        )
    return next_speaker


@torch.inference_mode()
def generate_greedy(
    model,
    context,
    n_steps=20,
    stop_at_eos=False,
    include_end_ts=False,
    max_input_length=None,
):
    """Generate by sampling"""
    # prepare input for model
    batch = model.tokenizer(context, include_end_ts=include_end_ts, return_tensors="pt")
    batch["attention_mask"] = None

    if max_input_length is not None:
        batch["input_ids"] = batch["input_ids"][:, -max_input_length:]
        batch["token_type_ids"] = batch["token_type_ids"][:, -max_input_length:]

    batch = {
        k: v.to(model.device) if isinstance(v, torch.Tensor) else v
        for k, v in batch.items()
    }

    # keep track of everything if `use_cache` is True
    generated = {"input_ids": [], "token_type_ids": []}

    next_speaker = None  # avoid lint errors (most)
    if model.include_speaker_tokens:
        batch["token_type_ids"] = None

    n = 0  # counter
    while True:
        out = model(**batch, use_cache=True)
        # sample
        # https://github.com/huggingface/transformers/blob/439a43b6b403205eeda2d62645fc16c93627d30d/src/transformers/generation_utils.py#L1373

        # Sample next tokens
        next_token_logits = out["logits"][:, -1, :]  # B, 1
        next_token = next_token_logits.topk(1).indices.squeeze(-1)

        # Update Next Speaker (i.e. `token_type_ids`)
        if not model.include_speaker_tokens:
            next_speaker = update_speaker_ids(batch, model.tokenizer)

        # update inputs
        batch["past_key_values"] = out["past_key_values"]
        batch["input_ids"] = next_token.unsqueeze(-1)  # (B, 1)
        generated["input_ids"].append(next_token)

        if not model.include_speaker_tokens:
            batch["token_type_ids"] = next_speaker.unsqueeze(-1)
            generated["token_type_ids"].append(next_speaker)

        if stop_at_eos and next_token.squeeze() == model.tokenizer.eos_token_id:
            break

        n += 1
        # stop criteria
        if n >= n_steps:
            break

    # reassemble the generated tokens if `use_cache`
    batch = {"past_key_values": batch["past_key_values"]}
    batch["input_ids"] = torch.stack(generated["input_ids"]).t()
    if not model.include_speaker_tokens:
        batch["token_type_ids"] = torch.stack(generated["token_type_ids"]).t()
    batch["tokens"] = [model.tokenizer.decode(b) for b in batch["input_ids"]]

    # to cpu
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.cpu()
    return batch


@torch.inference_mode()
def generate_sample(
    model,
    context,
    n_steps=20,
    top_p=0.9,
    top_k=-1,
    stop_at_eos=False,
    n_trajectories=4,
    include_end_ts=False,
    max_input_length=None,
):
    """Generate by sampling"""

    # prepare input for model
    batch = model.tokenizer(context, include_end_ts=include_end_ts, return_tensors="pt")
    batch["attention_mask"] = None

    if max_input_length is not None:
        batch["input_ids"] = batch["input_ids"][:, -max_input_length:]
        batch["token_type_ids"] = batch["token_type_ids"][:, -max_input_length:]

    # sample multiple trajectories at once
    batch = expand_batch(batch, n_trajectories)

    next_speaker = None  # avoid lint errors (most)
    if model.include_speaker_tokens:
        batch["token_type_ids"] = None

    # keep track of everything
    device = model.device
    batch = {
        k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()
    }
    generated = {
        "input_ids": torch.empty(0, device=device, dtype=torch.long),
        "token_type_ids": torch.empty(0, device=device, dtype=torch.long),
        "probs": torch.empty(0, device=device),
    }
    completed = {
        "input_ids": [],
        "token_type_ids": [],
        "probs": [],
        "trp_words": [],
        "trp_probs": [],
    }

    n = 0  # counter
    while n <= n_steps:
        out = model(**batch, use_cache=True)

        if n == 0:
            out["probs"] = out["logits"].softmax(dim=-1)

        # Sample next tokens
        # https://github.com/huggingface/transformers/blob/439a43b6b403205eeda2d62645fc16c93627d30d/src/transformers/generation_utils.py#L1373
        next_token_logits = out["logits"][:, -1, :]  # B, 1
        next_token, next_prob = sample_next_token(
            next_token_logits, top_p=top_p, top_k=top_k
        )

        # Update the generated
        generated["input_ids"] = torch.cat(
            (generated["input_ids"], next_token.unsqueeze(-1)), dim=-1
        )
        generated["probs"] = torch.cat(
            (generated["probs"], next_prob.unsqueeze(-1)), dim=-1
        )

        # Update the input for the next step
        batch["past_key_values"] = out["past_key_values"]
        batch["input_ids"] = next_token.unsqueeze(-1)

        # Update Next Speaker (i.e. `token_type_ids`)
        if model.include_speaker_tokens:
            next_speaker = update_speaker_ids(batch, model.tokenizer)
            generated["token_type_ids"] = torch.cat(
                (generated["token_type_ids"], next_speaker.unsqueeze(-1)), dim=-1
            )
            batch["token_type_ids"] = next_speaker.unsqueeze(-1)

        is_eos = next_token == model.tokenizer.eos_token_id
        if stop_at_eos and is_eos.sum() > 0:
            # which to keep and which to omit
            done = torch.where(is_eos)[0]
            keep = torch.where(torch.logical_not(is_eos))[0]

            # move the generated samples which are completed
            completed["input_ids"].append(generated["input_ids"][done])
            completed["probs"].append(generated["probs"][done])
            if model.include_speaker_tokens:
                completed["token_type_ids"].append(generated["token_type_ids"][done])

            if keep.nelement() == 0:  # We have completed the sampling of all batches
                generated["input_ids"] = []
                generated["token_type_ids"] = []
                generated["probs"] = []
                break
            else:  # Update the generated indices for continued sampling
                generated["input_ids"] = generated["input_ids"][keep]
                generated["probs"] = generated["probs"][keep]
                if not model.include_speaker_tokens:
                    generated["token_type_ids"] = generated["token_type_ids"][keep]

                # update the next model inputs to omit the completed samples
                batch["input_ids"] = batch["input_ids"][keep]
                if not model.include_speaker_tokens:
                    batch["token_type_ids"] = batch["token_type_ids"][keep]

                # Update past_key_values
                new_past = []
                for layer in range(len(batch["past_key_values"])):
                    new_past.append([])
                    for key_or_value in range(len(batch["past_key_values"][layer])):
                        tmp_key_val = batch["past_key_values"][layer][key_or_value][
                            keep
                        ]
                        new_past[-1].append(tmp_key_val)
                batch["past_key_values"] = new_past
        n += 1

    # If we reached n_steps and have not move everything to completed
    # (eos not reach and `stop_at_eos`==True) or `stop_at_eos`=False
    if len(generated["input_ids"]) > 0:
        completed["input_ids"].append(generated["input_ids"])
        completed["probs"].append(generated["probs"])
        if not model.include_speaker_tokens:
            completed["token_type_ids"].append(generated["token_type_ids"])

    # Stack all the sampled data
    if stop_at_eos:
        # PADDING
        max_len = -1
        for inp in completed["input_ids"]:
            if inp.shape[-1] > max_len:
                max_len = inp.shape[-1]

        # pad with -1
        new_inp, new_sp, new_probs = [], [], []
        tokens = []
        for i, inp in enumerate(completed["input_ids"]):
            for _inp in inp:
                tokens.append(model.tokenizer.decode(_inp))
            diff = max_len - inp.shape[-1]
            fill = torch.ones((inp.shape[0], diff), device=device, dtype=torch.long)
            if diff > 0:
                # fill with -1 to indicate that we don't have any words
                new_inp.append(torch.cat((inp, fill * -1), dim=-1))
                new_sp.append(
                    torch.cat((completed["token_type_ids"][i], fill * -1), dim=-1)
                )
                # fill with 1 to make prob calculations correct
                new_probs.append(
                    torch.cat((completed["probs"][i], fill.float()), dim=-1)
                )
            else:
                new_inp.append(inp)
                new_sp.append(completed["token_type_ids"][i])
                new_probs.append(completed["probs"][i])

        completed["input_ids"] = torch.cat(new_inp)
        completed["token_type_ids"] = torch.cat(new_sp)
        completed["probs"] = torch.cat(new_probs)
        completed["most_likely"] = completed["probs"].log().sum(dim=-1).argmax()
        completed["tokens"] = tokens
    else:
        completed["input_ids"] = torch.cat(completed["input_ids"])
        completed["probs"] = torch.cat(completed["probs"])
        p = completed["probs"].log().sum(dim=-1)
        completed["most_likely"] = p.argmax()
        completed["tokens"] = [
            model.tokenizer.decode(b) for b in completed["input_ids"]
        ]

    # to cpu
    for k, v in completed.items():
        if isinstance(v, torch.Tensor):
            completed[k] = v.cpu()

    return completed


def generate(
    model,
    context,
    n_steps=20,
    n_trajectories=1,
    top_p=0.9,
    top_k=-1,
    stop_at_eos=False,
    strategy="sampling",  # greedy, sampling
    include_end_ts=False,
    max_input_length=None,
):
    if strategy.lower().startswith("s"):
        return generate_sample(
            model,
            context,
            n_steps=n_steps,
            top_p=top_p,
            top_k=top_k,
            n_trajectories=n_trajectories,
            stop_at_eos=stop_at_eos,
            include_end_ts=include_end_ts,
            max_input_length=max_input_length,
        )
    else:
        return generate_greedy(
            model,
            context,
            n_steps=n_steps,
            stop_at_eos=stop_at_eos,
            include_end_ts=include_end_ts,
            max_input_length=max_input_length,
        )


def debug():
    from os.path import join

    from turngpt.model import TurnGPT

    # Load trained model
    checkpoint = join(
        # "assets/TurnGPT_proj/version_0/checkpoints/epoch=45-val_loss=-3.37196.ckpt"
        "runs/TurnGPT/TurnGPT_1bwldr6v/epoch=13_val_loss=1.3294.ckpt"
    )
    model = TurnGPT.load_from_checkpoint(checkpoint)
    model = model.eval().to("cuda")

    # Arguments
    context = ["Hello, how are you doing?", "I am well and you?"]
    context = [
        # "Hello there I basically had the worst day of my life",
        "So are you a student here at this university?",
        # "Oh no, what happened?",
        # "Do you want the long or the short story?",
    ]
    # context = [
    #     "Hello there how can i help you",
    #     "I want to go and see a movie can you help me",
    # ]
    # model.include_speaker_tokens = True
    sampled = generate(
        model,
        context,
        n_trajectories=10,
        n_steps=100,
        top_p=0.95,
        top_k=10,
        strategy="sampling",
        stop_at_eos=False,
    )
    for c in context:
        print(c)
    print("=" * 50)
    print(f"Most likely {sampled['most_likely']}:")
    print("=" * 50)
    print(sampled["tokens"][sampled["most_likely"]])
    # print('='*50)
    # for continuations in sampled["tokens"]:
    #     print(continuations)
    #     print("-" * 50)

    # wrapper for both strategies
    sampled = generate(
        model,
        context,
        n_trajectories=4,
        n_steps=100,
        strategy="greedy",
        stop_at_eos=True,
    )
    [print(c) for c in context]
    print("=" * 50)
    print(sampled["tokens"][0])

    sampled = generate_greedy(model, context, n_steps=50, stop_at_eos=True)
    print(sampled["tokens"])

    for inp, sp in zip(sampled["input_ids"][0], sampled["token_type_ids"][0]):
        print(inp.item(), sp.item())

    print(context)
    print("-" * 30)
    for text in sampled["tokens"]:
        print(text)
        print("-" * 30)


if __name__ == "__main__":
    from argparse import ArgumentParser

    from gptonly.model import GPT

    parser = ArgumentParser()
    parser.add_argument("--checkpoint", "-c", type=str)
    parser.add_argument("--strategy", "-s", type=str, default="sampling")
    parser.add_argument("--stop_at_ts", action="store_true")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--trajectories", type=int, default=10)
    args = parser.parse_args()

    model = GPT()
    model = model.eval()
    if torch.cuda.is_available():
        model = model.to("cuda")

    while True:
        #context = [input("Context: ")]
        context = ["Hello my name is"]
        sampled = generate(
            model,
            context,
            n_trajectories=args.trajectories,
            n_steps=args.steps,
            strategy=args.strategy,
            stop_at_eos=args.stop_at_ts,
        )
        print("=" * 50)
        print(f"Most likely {sampled['most_likely']}:")
        print("=" * 50)
        print(sampled["tokens"][sampled["most_likely"]])
        print("=" * 50)
        con = input("Exit? (y/n)")
        if con.lower() == "y":
            break
