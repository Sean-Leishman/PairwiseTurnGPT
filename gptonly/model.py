import torch
import logging
import random

from argparse import ArgumentParser

from transformers import BertModel, DistilBertModel, AutoTokenizer, AutoModel
from transformers import GPT2LMHeadModel, GPT2Config
from transformers.models.gpt2.modeling_gpt2 import GPT2DoubleHeadsModelOutput

from tokenizers import Regex
from tokenizers.normalizers import (
    Lowercase,
    NFD,
    StripAccents,
    Replace,
    Strip,
    Sequence,
)

from gptonly.tokenizer import SpokenDialogTokenizer


class GPT(torch.nn.Module):
    def __init__(self,
                 pretrained_model_name="gpt2",
                 finetune=True,
                 speaker_tokens=True,
                 projection_labels=True,
                 projection_statistics={'mean': 0, 'var': 1, 'bins': 20},
                 projection_clip=10,
                 categorize_projection=False,
                 weight_regular_token=0.5,
                 weight_eos_token=1.0,
                 weight_tokens=False,
                 remove_special_tokens=False,
                 individual_speaker_tokens=False,
                 include_speaker_embeddings=False,
                 device="cuda:0",
                 **kwargs,
                 ):
        super(GPT, self).__init__()
        self.logger = logging.getLogger(__name__)

        self.device = device
        self.include_speaker_tokens = speaker_tokens
        self.include_projection_labels = projection_labels

        self.categorize_projection = categorize_projection
        self.projection_stats = projection_statistics
        self.projection_clip = projection_clip

        self.dropout = torch.nn.Dropout(p=0.1)

        config = GPT2Config.from_pretrained(pretrained_model_name)
        self.gpt = GPT2LMHeadModel.from_pretrained(
            pretrained_model_name, config=config)

        self.gpt.to(self.device)

        self.trp_projection_steps = 5
        if self.categorize_projection:
            self.trp_projection_head = torch.nn.Linear(
                self.gpt.config.hidden_size, self.projection_stats['bins'])
        else:
            self.trp_projection_head = torch.nn.Linear(
                self.gpt.config.hidden_size, 1)

        self.weight_regular_token = weight_regular_token
        self.weight_eos_token = weight_eos_token
        self.weight_tokens = weight_tokens

        self.remove_special_tokens = remove_special_tokens

        self.individual_speaker_tokens = individual_speaker_tokens
        self.include_speaker_tokens = include_speaker_embeddings

        tokens = ['<speakerA>', '<speakerB>', '<emp>']
        if not self.individual_speaker_tokens:
            tokens = ['<emp>']

        self.tokenizer = SpokenDialogTokenizer(tokens=tokens)
        self.init_tokenizer()

        update_params = ["embd_pdrop", "attn_pdrop", "resid_pdrop"]
        if not finetune:
            self.logger.info('model: bert parameters frozen')
            for param in self.parameters():
                param.requires_grad = True

        self.logger.info(f"Initialise model class with {weight_tokens}: {weight_eos_token} -> {weight_regular_token} " +
                         f"Projection: {self.include_projection_labels}, Speaker Tokens: {self.include_speaker_tokens} " +
                         f"Individual Speaker Tokens: {self.individual_speaker_tokens} ")

    @torch.no_grad()
    def get_loss_weight(self):
        if not self.weight_tokens:
            return None

        weight = (
            torch.ones(len(self.tokenizer), dtype=torch.float) *
            self.weight_regular_token
        )

        for token in self.tokenizer.special_tokens:
            token_id = self.tokenizer.convert_tokens_to_ids(token)
            weight[token_id] = self.weight_eos_token
        weight[self.tokenizer.eos_token_id] = self.weight_eos_token
        weight[self.tokenizer.convert_tokens_to_ids('<emp>')] = self.weight_regular_token

        return weight.to(self.device)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None, projection_labels=None, use_cache=False, past_key_values=None, **kwargs):
        if not self.include_speaker_tokens:
            token_type_ids = None

        if not self.include_projection_labels:
            projection_labels = None

        out = self.gpt.transformer(
            input_ids,
            # labels=labels,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_hidden_states=True,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **kwargs
        )

        hidden_states = out[0]

        lm_logits = self.gpt.lm_head(hidden_states)
        loss = None
        if labels is not None:
            loss = self.cross_entropy_loss(lm_logits, labels)

        projection_logits = self.trp_projection_head(hidden_states)
        projection_loss = None
        if projection_labels is not None:
            projection_loss, count = self.projection_loss(
                projection_logits.squeeze(-1), projection_labels)

        return GPT2DoubleHeadsModelOutput(
            loss=loss,
            logits=lm_logits,
            mc_loss=projection_loss,
            mc_logits=projection_logits,
            past_key_values=out.past_key_values,
            hidden_states=out.hidden_states,
            attentions=out.attentions
        )

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

    def projection_loss(self, logits, labels, reduction="mean"):

        if not self.categorize_projection:
            loss_fct = torch.nn.MSELoss()

            # Clip Above a Certain Value
            idxs = labels != -100
            logits = torch.masked_select(logits, idxs).float()
            labels = torch.masked_select(labels, idxs).float()

            loss = loss_fct(
                logits,
                labels
            )

            return loss, 0
        else:
            loss_fct = torch.nn.CrossEntropyLoss(reduction='none')

            if labels.dtype == torch.float:
                return None, 0

            loss = loss_fct(
                logits.view(-1, logits.size(-1)), labels.long().view(-1))

            count = 0
            if reduction == "mean":
                loss = loss.mean()

            return loss, count

    def generate(self, input_ids=None, speaker_ids=None, mask=None, output_scores=False, n_sequences=1, stop_at_eos=False):
        if input_ids is None:
            sample_output = self.gpt.generate(
                bos_token_id=random.randint(1, 30000),
                token_type_ids=speaker_ids,
                do_sample=True,
                top_k=50,
                max_length=100,
                top_p=0.95,
                stop_at_eos=stop_at_eos,
                num_return_sequences=n_sequences,
                pad_token_id=self.tokenizer.pad_token_id
            )
        else:
            sample_output = self.gpt.generate(
                input_ids=input_ids,
                token_type_ids=speaker_ids,
                attention_mask=mask,
                do_sample=True,
                top_k=50,
                max_length=300,
                top_p=0.95,
                num_return_sequences=n_sequences,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                output_scores=True,
                return_dict_in_generate=True,
            )
        return sample_output

    def init_tokenizer(self, tokens=['!', '?', '.']):
        self.gpt.resize_token_embeddings(new_num_tokens=len(self.tokenizer))
        # self.tokenizer.sep_token_id = self.tokenizer.convert_tokens_to_ids('[SEP]')

        ts = self.tokenizer.eos_token_id
        with torch.no_grad():
            ids = torch.tensor(
                self.tokenizer.convert_tokens_to_ids(tokens)).to(self.device)
            avg_emb = self.gpt.transformer.wte(ids).mean(0)
            self.gpt.transformer.wte.weight.data[ts] = avg_emb

        print(f"Initalized {self.tokenizer.eos_token} -> avg({tokens})")

    def get_tokenizer(self):
        return self.tokenizer

    def from_string(self, string):
        output = {}
        output['dialog'] = "<eot> ".join(string) + "<eot>"
        tokens = self.tokenizer(
            output['dialog'], return_tensors="pt", truncation=True)

        output['input_ids'] = tokens['input_ids'].to(self.device)

        current_speaker = 'A'
        token_type_ids = [[]]

        SEP_token = self.tokenizer.convert_tokens_to_ids("<eot>")
        for token in output['input_ids'][0]:
            # Is [SEP] token self.tokenizer.encode('[SEP]') -> [101, 102, 102]
            if token.item() == SEP_token:
                current_speaker = 'A' if current_speaker == 'B' else 'B'

            token_type_ids[0].append(
                0 if current_speaker == 'A' else 1)

        output['token_type_ids'] = torch.tensor(token_type_ids).to(self.device)
        return output
