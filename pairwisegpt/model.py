import torch
import logging
import random

from argparse import ArgumentParser

from transformers import BertModel, DistilBertModel, AutoTokenizer, AutoModel
from transformers.models.gpt2.configuration_gpt2 import GPT2Config

from pairwisegpt.gpt import ModifiedGPT2LMHeadModel, GPT2DoubleHeadsModelOutput
from pairwisegpt.generate import generate

from tokenizers import Regex
from tokenizers.normalizers import (
    Lowercase,
    NFD,
    StripAccents,
    Replace,
    Strip,
    Sequence,
)

from pairwisegpt.tokenizer import SpokenDialogTokenizer


class PairwiseGPT(torch.nn.Module):
    def __init__(self,
                 pretrained_model_name="gpt2",
                 finetune=True,
                 include_speaker_embeddings=False,
                 projection_labels=True,
                 weight_regular_token=0.5,
                 weight_eos_token=1.0,
                 weight_tokens=False,
                 device=None,
                 remove_cross_attention=False,
                 include_overlap_token=False,
                 include_yield_token=False,
                 include_sil_token=False,
                 include_bc_token=False,
                 include_end_bc_token=False,
                 remove_start_tokens=False,
                 individual_ts=False,
                 clone_self_cross=False,
                 filter_bc_overlap_token=False,
                 no_loss_emp=False,
                 **kwargs,
                 ):
        super(PairwiseGPT, self).__init__()
        self.logger = logging.getLogger(__name__)

        self.device = device
        self.include_speaker_tokens = include_speaker_embeddings
        self.include_projection_labels = projection_labels

        self.dropout = torch.nn.Dropout(p=0.1)

        config = GPT2Config.from_pretrained(pretrained_model_name)
        config.add_cross_attention = not remove_cross_attention

        self.gpt = ModifiedGPT2LMHeadModel.from_pretrained(
            pretrained_model_name, config=config)
        # self.gpt = ModifiedGPT2LMHeadModel(config=config)
        if clone_self_cross:
            self.gpt.clone_self_cross_weights()

        self.gpt.to(device)

        self.trp_projection_steps = 5
        self.trp_projection_head = torch.nn.Linear(
            self.gpt.config.hidden_size, 1)
        self.trp_projection_head.to(device)

        self.weight_regular_token = weight_regular_token
        self.weight_eos_token = weight_eos_token
        self.weight_tokens = weight_tokens

        self.individual_ts = individual_ts

        self.include_bc_token = include_bc_token
        self.include_end_bc_token = include_end_bc_token
        self.include_sil_token = include_sil_token
        self.include_yield_token = include_yield_token
        self.include_overlap_token = include_overlap_token
        self.filter_bc_overlap_token = filter_bc_overlap_token
        self.remove_start_tokens = remove_start_tokens
        self.cross_attention = not remove_cross_attention

        self.special_tokens = self.init_special_tokens()

        self.tokenizer = SpokenDialogTokenizer(tokens=self.special_tokens)
        self.init_tokenizer()

        self.logger.info(
            f"model: loaded {pretrained_model_name} with{'out' if remove_cross_attention else ''} cross attention")

        update_params = ["embd_pdrop", "attn_pdrop", "resid_pdrop"]
        if not finetune:
            self.logger.info(
                f'model: {pretrained_model_name} parameters frozen')
            for param in self.parameters():
                param.requires_grad = True

    def init_special_tokens(self):
        start_tokens_def = ['<sot>'] if self.individual_ts else [
            '<speaker1>', '<speaker2>']
        start_tokens = ['<sot>', '<sbc>', '<bint>']
        bc_overlap = ['<ebc>', '<eint>']
        tokens = ['<emp>', '<yield>', '<bc>']

        if not self.remove_start_tokens:
            tokens = start_tokens + tokens
        if not self.filter_bc_overlap_token:
            tokens = bc_overlap + tokens

        new_tokens = []
        for token in tokens:
            if token in {'<bint>', '<eint>'} and not self.include_overlap_token:
                continue
            if token in {'<sil>'} and not self.include_sil_token:
                continue
            if token in {'<sbc>', '<ebc>'} and not self.include_end_bc_token:
                continue
            if token in {'<yield>'} and not self.include_yield_token:
                continue
            if token in {'<bc>'} and not self.include_bc_token:
                continue
            new_tokens.append(token)
        return new_tokens

    def forward(self,
                input_idsA=None,
                labelsA=None,
                projection_labelsA=None,
                attention_maskA=None,
                token_type_idsA=None,
                input_idsB=None,
                labelsB=None,
                projection_labelsB=None,
                attention_maskB=None,
                token_type_idsB=None,
                past_key_valuesA=None,
                past_key_valuesB=None,
                use_cache=False,
                **kwargs):

        if not self.include_speaker_tokens:
            token_type_idsA = None
            token_type_idsB = None
        if not self.include_projection_labels:
            projection_labelsA = None
            projection_labelsB = None

        # Might have to update cross attention layer such that rather than taking
        # cross attention over the entire sequence we only do so over thr trill
        # style mask to mask future tokens with respect of A to B
        out = self.gpt.transformer(
            input_idsA=input_idsA,
            attention_maskA=attention_maskA,
            token_type_idsA=token_type_idsA,
            input_idsB=input_idsB,
            attention_maskB=attention_maskB,
            token_type_idsB=token_type_idsB,
            output_hidden_states=True,
            past_key_valuesA=past_key_valuesA,
            past_key_valuesB=past_key_valuesB,
            use_cache=use_cache,
            **kwargs
        )

        hidden_statesA = out[0][0]
        hidden_statesB = out[0][1]

        lm_logitsA = self.gpt.lm_head(hidden_statesA)
        lm_logitsB = self.gpt.lm_head(hidden_statesB)
        lm_loss = None
        loss = 0
        if labelsA is not None:
            loss = self.cross_entropy_loss(lm_logitsA, labelsA)
            loss += self.cross_entropy_loss(lm_logitsB, labelsB)

        projection_logits = self.trp_projection_head(hidden_statesA)
        projection_loss = None

        return GPT2DoubleHeadsModelOutput(
            loss=loss,
            logits=(lm_logitsA, lm_logitsB),
            mc_loss=projection_loss,
            mc_logits=projection_logits,
            past_key_values=out.past_key_values,
            hidden_states=out.hidden_states,
            attentions=out.attentions,
            cross_attentions=out.cross_attentions
        )

    def forward_channel(self, input_ids, attention_mask=None, token_type_ids=None):
        pass

    def init_tokenizer(self, tokens=['!', '?', '.']):
        self.gpt.resize_token_embeddings(new_num_tokens=len(self.tokenizer))
        # self.tokenizer.sep_token_id = self.tokenizer.convert_tokens_to_ids('[SEP]')

        eot_tokens = ['<eot>', '<yield>']  # '<bc>', '<ebc>', '<eint>']
        if self.filter_bc_overlap_token:
            eot_tokens = ['<eot>', '<yield>', '<bc>']

        with torch.no_grad():
            ids = torch.tensor(
                self.tokenizer.convert_tokens_to_ids(tokens)).to(self.device)
            eot_ids = torch.tensor(
                self.tokenizer.convert_tokens_to_ids(eot_tokens)).to(self.device)

            for token in eot_ids:
                if token == self.tokenizer.pad_token_id:
                    continue

                avg_emb = self.gpt.transformer.wte(ids).mean(0).clone()
                self.gpt.transformer.wte.weight.data[token] = avg_emb
                print(
                    f"Initalized {token}={self.tokenizer.convert_ids_to_tokens(token.item())} -> avg({tokens})")

    def get_tokenizer(self):
        return self.tokenizer

    def cross_entropy_loss(self, logits, labels, reduction="mean"):
        weight = self.get_loss_weight()

        loss_fct = torch.nn.CrossEntropyLoss(weight=weight, reduction="none")
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )

        if reduction != "none":
            loss = loss.mean()

        return loss

    def bce_loss(self, logits, labels):
        loss_fct = torch.nn.BCEWithLogitsLoss()

        shift_logits = logits[..., :-1]
        shift_labels = labels[..., 1:]

        indicies = shift_labels != -100
        loss = loss_fct(
            torch.masked_select(shift_logits, indicies).float(),
            torch.masked_select(shift_labels, indicies).float()
        )

        return loss

    def generate(self, context=None, output_scores=False, n_sequences=1, stop_at_eos=False):
        if context is None:
            sample_output = self.gpt.generate(
                bos_token_id=random.randint(1, 30000),
                do_sample=True,
                top_k=50,
                max_length=100,
                top_p=0.95,
                num_return_sequences=1,
                stop_at_eos=stop_at_eos,
                pad_token_id=self.tokenizer.pad_token_id
            )
        else:
            sample_output = generate(
                self,
                context=context,
                do_sample=True,
                top_k=50,
                max_length=300,
                top_p=0.95,
                num_return_sequences=n_sequences,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                output_scores=True,
                stop_at_eos=stop_at_eos,
                return_dict_in_generate=True,
            )

        return sample_output

    def prepare_inputs_for_generation(self, input_idsA, input_idsB, token_type_idsA, token_type_idsB,
                                      attention_maskA=None, attention_maskB=None,
                                      past_key_valuesA=None, past_key_valuesB=None, **kwargs):
        model_inputsA = self.gpt.prepare_inputs_for_generation(
            input_idsA, token_type_ids=token_type_idsA, attention_mask=attention_maskA)
        model_inputsB = self.gpt.prepare_inputs_for_generation(
            input_idsB, token_type_ids=token_type_idsB, attention_mask=attention_maskB)

        model_inputs = {
            'input_idsA': model_inputsA['input_ids'],
            'input_idsB': model_inputsB['input_ids'],
            # 'attention_maskA': model_inputsA['attention_mask'],
            # 'attention_maskB': model_inputsB['attention_mask'],
            'token_type_idsA': model_inputsA['token_type_ids'],
            'token_type_idsB': model_inputsB['token_type_ids'],
            'past_key_valuesA': past_key_valuesA,
            'past_key_valuesB': past_key_valuesB,
        }

        for k, v in model_inputs.items():
            if isinstance(v, torch.Tensor) and len(model_inputs[k].shape) == 1:
                model_inputs[k] = model_inputs[k].unsqueeze(0)

        return model_inputs

    @torch.no_grad()
    def get_loss_weight(self):
        if not self.weight_tokens:
            return None

        weight = (
            torch.ones(len(self.tokenizer), dtype=torch.float) *
            self.weight_regular_token
        )
        for token in self.tokenizer.special_tokens:
            if self.filter_bc_overlap_token and token not in ['<eot>', '<yield>', '<bc>']:
                continue

            id = self.tokenizer.convert_tokens_to_ids(token)
            weight[id] = self.weight_eos_token

            # self.logger.info(f"set weight of {token}={id} to {weight[id]}")

        weight[self.tokenizer.eos_token_id] = self.weight_eos_token
        # self.logger.info(f"set weight of <eot>={self.tokenizer.eos_token_id} to {weight[self.tokenizer.eos_token_id]}")

        emp_token_id = self.tokenizer.convert_tokens_to_ids("<emp>")
        weight[emp_token_id] = self.weight_regular_token
        # self.logger.info(f"set weight of <emp>={emp_token_id} to {weight[emp_token_id]}")

        return weight.to(self.device)
