import einops
import torch


def expand_batch(batch, n_trajectories=1):
    if n_trajectories > 1:
        batch["input_idsA"] = einops.repeat(
            batch["input_idsA"], "b n -> (r b) n", r=n_trajectories
        )
        batch["input_idsB"] = einops.repeat(
            batch["input_idsB"], "b n -> (r b) n", r=n_trajectories
        )
        batch["token_type_idsA"] = einops.repeat(
            batch["token_type_idsA"], "b n -> (r b) n", r=n_trajectories
        )
        batch["token_type_idsB"] = einops.repeat(
            batch["token_type_idsB"], "b n -> (r b) n", r=n_trajectories
        )
    return batch


def sample_next_dual_token(logitsA, logitsB, top_p=-1, top_k=-1):
    """
    Adapted from https://github.com/ErikEkstedt/TurnGPT/blob/update24/turngpt/generation.py 
    for use with dual generated tokens so rather than the probability of one channel we assume 
    conditonal independence and therefore take their joint probability 

    Samples the next token given the probabilities over the
    """

    assert top_p > 0 or top_k > 0, "Either top_p or top_k > 0."
    probsA = logitsA.softmax(dim=-1)
    probsB = logitsB.softmax(dim=-1)
    probsA, token_idxA = probsA.sort(dim=-1, descending=True)
    probsB, token_idxB = probsB.sort(dim=-1, descending=True)

    if top_k > 0:
        probsA, token_idxA = probsA[:, :top_k], token_idxA[:, :top_k]
        probsB, token_idxB = probsB[:, :top_k], token_idxB[:, :top_k]

        p_idxA = torch.multinomial(probsA, num_samples=1).squeeze(1)  # B
        p_idxB = torch.multinomial(probsB, num_samples=1).squeeze(1)  # B

        next_probA = probsA[torch.arange(len(p_idxA)), p_idxA]
        next_probB = probsB[torch.arange(len(p_idxB)), p_idxB]

        next_tokenA = token_idxA[torch.arange(len(p_idxA)), p_idxA].long()
        next_tokenB = token_idxB[torch.arange(len(p_idxB)), p_idxB].long()
    else:
        pcumA = probsA.cumsum(dim=-1)  # cumulative probabilities
        pcumB = probsB.cumsum(dim=-1)  # cumulative probabilities

        # cumulative less than or equal to `top_p`
        p_batchA, p_idxA = torch.where(pcumA <= top_p)
        p_batchB, p_idxB = torch.where(pcumB <= top_p)

        psmallA = probsA[p_batchA, p_idxA]
        psmallB = probsB[p_batchB, p_idxB]

        tsmallA = token_idxA[p_batchA, p_idxA]
        tsmallB = token_idxB[p_batchB, p_idxB]

        # the cumulative probability distribution may not be of the same size
        # so we must sample for the batches individually
        next_tokenA = torch.zeros(
            (probsA.shape[0]), dtype=torch.long, device=logitsA.device
        )
        next_tokenB = torch.zeros(
            (probsB.shape[0]), dtype=torch.long, device=logitsB.device
        )
        next_probA = torch.zeros((probsA.shape[0]), device=logitsA.device)
        next_probB = torch.zeros((probsB.shape[0]), device=logitsB.device)
        for n_batch in range(probsA.shape[0]):
            eqA = p_batchA == n_batch
            eqB = p_batchB == n_batch
            if eqA.sum() == 0 and eqB.sum() == 0:  # first token is more likely than top_p
                next_tokA = token_idxA[n_batch, 0].item()
                next_tokB = token_idxB[n_batch, 0].item()
                next_pA = probsA[n_batch, 0].item()
                next_pB = probsB[n_batch, 0].item()
            else:
                pA = psmallA[eqA]
                pB = psmallB[eqB]

                tA = tsmallA[eqA]
                tB = tsmallB[eqB]

                p_idxA = torch.multinomial(pA, num_samples=1)  # B
                p_idxB = torch.multinomial(pB, num_samples=1)  # B

                next_pA = pA[p_idxA].item()
                next_pB = pB[p_idxB].item()

                next_tokA = tA[p_idxA].item()
                next_tokB = tB[p_idxB].item()
            next_tokenA[n_batch] = next_tokA
            next_tokenB[n_batch] = next_tokB
            next_probA[n_batch] = next_pA
            next_probB[n_batch] = next_pB
    return next_tokenA, next_tokenB, next_probA, next_probB


def update_speaker_idsA(batch, tokenizer):
    def _change_speakers(last_speaker, indices, tokenizer):
        for ind in indices:
            last_speaker[ind] = 0 if last_speaker[ind] == 1  else 1
        return last_speaker

    next_speaker = batch["token_type_idsA"][:, -1]
    eos_tokens = batch["input_idsA"][:, -1] == tokenizer.eos_token_id
    if any(eos_tokens):
        change_speaker_idx = torch.where(eos_tokens)[0]
        next_speaker = _change_speakers(
            next_speaker.clone(), change_speaker_idx, tokenizer
        )
    return next_speaker

def update_speaker_idsB(batch, tokenizer):
    def _change_speakers(last_speaker, indices, tokenizer):
        for ind in indices:
            last_speaker[ind] = 0 if last_speaker[ind] == 2 else 2
        return last_speaker

    next_speaker = batch["token_type_idsB"][:, -1]
    eos_tokens = batch["input_idsB"][:, -1] == tokenizer.eos_token_id
    if any(eos_tokens):
        change_speaker_idx = torch.where(eos_tokens)[0]
        next_speaker = _change_speakers(
            next_speaker.clone(), change_speaker_idx, tokenizer
        )
    return next_speaker

@torch.no_grad
def generate_sample(model, context, n_steps=20, top_p=0.9, top_k=50, n_trajectories=1, stop_at_eos=False):
    device = model.device

    if isinstance(context, str):
        batch = model.tokenize_strings(context)
    else:
        batch = {}
        batch['input_idsA'] = context['speakerA']['input_ids'].to(device)
        batch['input_idsB'] = context['speakerB']['input_ids'].to(device)
        batch['attention_maskA'] = context['speakerA']['attention_mask'].to(
            device)
        batch['attention_maskB'] = context['speakerB']['attention_mask'].to(
            device)
        batch['token_type_idsA'] = context['speakerA']['speaker_ids'].to(
            device)
        batch['token_type_idsB'] = context['speakerB']['speaker_ids'].to(
            device)

    batch = expand_batch(batch, n_trajectories)

    generated = {
        'input_idsA': torch.empty(0, device=device),
        'input_idsB': torch.empty(0, device=device),
        'probsA': torch.empty(0, device=device),
        'probsB': torch.empty(0, device=device),
        'token_type_idsA': torch.empty(0, device=device),
        'token_type_idsB': torch.empty(0, device=device),
    }
    completed = {'input_idsA': [],
                 'input_idsB': [],
                 'token_type_idsA': [],
                 'token_type_idsB': [],
                 'probsA': [],
                 'probsB': [],
                 }

    n = 0
    while n < n_steps:
        if batch['input_idsA'].shape[-1] != 256:
            pass

        if batch['input_idsA'].shape != batch['input_idsB'].shape:
            pass
        if batch['token_type_idsA'].shape != batch['token_type_idsB'].shape:
            pass
        if batch['input_idsA'].shape != batch['token_type_idsA'].shape:
            pass
        batch = model.prepare_inputs_for_generation(**batch)
        batch['use_cache'] = True
        out = model(**batch)

        next_token_logitsA = out['logits'][0]
        next_token_logitsB = out['logits'][1]
        if len(out['logits'][0].shape) == 2:
            next_token_logitsA = out['logits'][0].unsqueeze(0)
            next_token_logitsB = out['logits'][1].unsqueeze(0)

        next_token_logitsA = next_token_logitsA[:, -1, :]
        next_token_logitsB = next_token_logitsB[:, -1, :]

        next_tokenA, next_tokenB, next_probA, next_probB = sample_next_dual_token(
            next_token_logitsA, next_token_logitsB, top_p=top_p, top_k=top_k)

        # Update the generated
        generated["input_idsA"] = torch.cat(
            (generated["input_idsA"], next_tokenA.unsqueeze(-1)), dim=-1
        )
        generated["probsA"] = torch.cat(
            (generated["probsA"], next_probA.unsqueeze(-1)), dim=-1
        )
        generated["input_idsB"] = torch.cat(
            (generated["input_idsB"], next_tokenB.unsqueeze(-1)), dim=-1
        )
        generated["probsB"] = torch.cat(
            (generated["probsB"], next_probB.unsqueeze(-1)), dim=-1
        )

        # Update the input for the next step
        batch["past_key_valuesA"] = out.past_key_values[0]
        batch["past_key_valuesB"] = out.past_key_values[1]
        batch["input_idsA"] = next_tokenA.unsqueeze(-1)
        batch["input_idsB"] = next_tokenB.unsqueeze(-1)

        # Update Next Speaker (i.e. `token_type_ids`)
        if model.include_speaker_tokens:
            next_speakerA = update_speaker_idsA(batch, model.tokenizer)
            next_speakerB = update_speaker_idsB(batch, model.tokenizer)
            generated["token_type_idsA"] = torch.cat(
                (generated["token_type_idsA"], next_speakerA.unsqueeze(-1)), dim=-1
            )
            batch["token_type_idsA"] = next_speakerA.unsqueeze(-1)

            generated["token_type_idsB"] = torch.cat(
                (generated["token_type_idsB"], next_speakerB.unsqueeze(-1)), dim=-1
            )
            batch["token_type_idsB"] = next_speakerB.unsqueeze(-1)

        is_eosA = next_tokenA == model.tokenizer.eos_token_id
        is_eosB = next_tokenB == model.tokenizer.eos_token_id
        if stop_at_eos and (is_eosA.sum() > 0 or is_eosB.sum() > 0):
            # which to keep and which to omit
            doneA = torch.where(is_eosA)[0]
            keepA = torch.where(torch.logical_not(is_eosA))[0]
            doneB = torch.where(is_eosB)[0]
            keepB = torch.where(torch.logical_not(is_eosB))[0]

            done = doneA if doneA.sum() != 0 else doneB
            keep = keepA if doneA.sum() != 0 else keepB

            # move the generated samples which are completed
            completed["input_idsA"].append(generated["input_idsA"][done])
            completed["input_idsB"].append(generated["input_idsB"][done])
            completed["probsA"].append(generated["probsA"][done])
            completed["probsB"].append(generated["probsB"][done])
            if model.include_speaker_tokens:
                completed["token_type_idsA"].append(
                    generated["token_type_idsA"][done])
                completed["token_type_idsB"].append(
                    generated["token_type_idsB"][done])

            if keepA.nelement() == 0 and keepB.nelement() == 0:  # We have completed the sampling of all batches
                generated["input_idsA"] = []
                generated["token_type_idsA"] = []
                generated["probsA"] = []

                generated["input_idsB"] = []
                generated["token_type_idsB"] = []
                generated["probsB"] = []
                break
            else:  # Update the generated indices for continued sampling
                generated["input_idsA"] = generated["input_idsA"][keep]
                generated["probsA"] = generated["probsA"][keep]

                generated["input_idsB"] = generated["input_idsB"][keep]
                generated["probsB"] = generated["probsB"][keep]
                if model.include_speaker_tokens:
                    generated["token_type_idsA"] = generated["token_type_idsA"][keep]
                    generated["token_type_idsB"] = generated["token_type_idsB"][keep]

                # update the next model inputs to omit the completed samples
                batch["input_idsA"] = batch["input_idsA"][keep]
                batch["input_idsB"] = batch["input_idsB"][keep]
                if model.include_speaker_tokens:
                    batch["token_type_idsA"] = batch["token_type_idsA"][keep]
                    batch["token_type_idsB"] = batch["token_type_idsB"][keep]

                # Update past_key_values
                new_pastA = []
                for layer in range(len(batch["past_key_valuesA"])):
                    new_pastA.append([])
                    for key_or_value in range(len(batch["past_key_valuesA"][layer])):
                        tmp_key_val = batch["past_key_valuesA"][layer][key_or_value][
                            keep
                        ]
                        new_pastA[-1].append(tmp_key_val)

                new_pastB = []
                for layer in range(len(batch["past_key_valuesB"])):
                    new_pastB.append([])
                    for key_or_value in range(len(batch["past_key_valuesB"][layer])):
                        tmp_key_val = batch["past_key_valuesB"][layer][key_or_value][
                            keep
                        ]
                        new_pastB[-1].append(tmp_key_val)
                batch["past_key_valuesA"] = new_pastA
                batch["past_key_valuesB"] = new_pastB
        n += 1

        # If we reached n_steps and have not move everything to completed
        # (eos not reach and `stop_at_eos`==True) or `stop_at_eos`=False
    if len(generated["input_idsA"]) > 0:
        completed["input_idsA"].append(generated["input_idsA"])
        completed["probsA"].append(generated["probsA"])
        if model.include_speaker_tokens:
            completed["token_type_idsA"].append(generated["token_type_idsA"])

    if len(generated["input_idsB"]) > 0:
        completed["input_idsB"].append(generated["input_idsB"])
        completed["probsB"].append(generated["probsB"])
        if model.include_speaker_tokens:
            completed["token_type_idsB"].append(generated["token_type_idsB"])

        # Stack all the sampled data
    if stop_at_eos:
        # PADDING
        max_len = -1
        for inp in completed["input_idsA"]:
            if inp.shape[-1] > max_len:
                max_len = inp.shape[-1]

        # pad with -1
        new_inp, new_sp, new_probs = [], [], []
        tokens = []
        for i, inp in enumerate(completed["input_idsA"]):
            for _inp in inp:
                tokens.append(model.tokenizer.decode(_inp.long()))
            diff = max_len - inp.shape[-1]
            fill = torch.ones((inp.shape[0], diff),
                              device=device, dtype=torch.long)
            if diff > 0:
                # fill with -1 to indicate that we don't have any words
                new_inp.append(torch.cat((inp, fill * -1), dim=-1))
                new_sp.append(
                    torch.cat((completed["token_type_idsA"]
                              [i], fill * -1), dim=-1)
                )
                # fill with 1 to make prob calculations correct
                new_probs.append(
                    torch.cat((completed["probsA"][i], fill.float()), dim=-1)
                )
            else:
                new_inp.append(inp)
                new_sp.append(completed["token_type_idsA"][i])
                new_probs.append(completed["probsA"][i])

        completed["input_idsA"] = torch.cat(new_inp).int()
        completed["token_type_idsA"] = torch.cat(new_sp)
        completed["probsA"] = torch.cat(new_probs)
        completed["most_likelyA"] = completed["probsA"].log().sum(
            dim=-1).argmax()
        completed["tokensA"] = tokens
    else:
        completed["input_idsA"] = torch.cat(completed["input_idsA"]).int()
        completed["token_type_idsA"] = torch.cat(completed["token_type_idsA"])
        completed["probsA"] = torch.cat(completed["probsA"])
        p = completed["probsA"].log().sum(dim=-1)
        completed["most_likelyA"] = p.argmax()
        completed["tokensA"] = [
            model.tokenizer.decode(b) for b in completed["input_idsA"].int()
        ]

    if stop_at_eos:
        # pad with -1
        max_len = -1
        for inp in completed["input_idsB"]:
            if inp.shape[-1] > max_len:
                max_len = inp.shape[-1]

        new_inp, new_sp, new_probs = [], [], []
        tokens = []
        for i, inp in enumerate(completed["input_idsB"]):
            for _inp in inp:
                tokens.append(model.tokenizer.decode(_inp.long()))
            diff = max_len - inp.shape[-1]
            fill = torch.ones((inp.shape[0], diff),
                              device=device, dtype=torch.long)
            if diff > 0:
                # fill with -1 to indicate that we don't have any words
                new_inp.append(torch.cat((inp, fill * -1), dim=-1))
                new_sp.append(
                    torch.cat((completed["token_type_idsB"]
                              [i], fill * -1), dim=-1)
                )
                # fill with 1 to make prob calculations correct
                new_probs.append(
                    torch.cat((completed["probsB"][i], fill.float()), dim=-1)
                )
            else:
                new_inp.append(inp)
                new_sp.append(completed["token_type_idsB"][i])
                new_probs.append(completed["probsB"][i])

        completed["input_idsB"] = torch.cat(new_inp).int()
        completed["speaker_idsB"] = torch.cat(new_sp)
        completed["probsB"] = torch.cat(new_probs)
        completed["most_likelyB"] = completed["probsB"].log().sum(
            dim=-1).argmax()
        completed["tokensB"] = tokens
    else:
        completed["input_idsB"] = torch.cat(completed["input_idsB"]).int()
        completed["speaker_idsB"] = torch.cat(completed["speaker_idsB"])
        completed["probsB"] = torch.cat(completed["probsB"])
        p = completed["probsB"].log().sum(dim=-1)
        completed["most_likelyB"] = p.argmax()
        completed["tokensB"] = [
            model.tokenizer.decode(b) for b in completed["input_idsB"]
        ]

    # to cpu
    for k, v in completed.items():
        if isinstance(v, torch.Tensor):
            completed[k] = v.cpu()
        else:
            completed[k] = v

    return completed


def generate(model,
             context,
             n_steps=20,
             n_trajectories=10,
             top_p=0.9,
             top_k=50,
             stop_at_eos=False,
             strategy='sampling',
             **kwargs):
    return generate_sample(model,
                           context,
                           n_steps=n_steps,
                           top_p=top_p,
                           top_k=top_k,
                           n_trajectories=n_trajectories,
                           stop_at_eos=stop_at_eos)
