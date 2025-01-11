import torch
import random


from transformers.models.gpt2.configuration_gpt2 import GPT2Config
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

from pairwisegpt.gpt import ModifiedGPT2LMHeadModel, GPT2DoubleHeadsModelOutput
from pairwisegpt.generate import generate
from pairwisegpt.utils import get_logger


from pairwisegpt.tokenizer import SpokenDialogTokenizer

logger = get_logger(__name__)


class PairwiseGPT(torch.nn.Module):
    def __init__(
        self,
        pretrained="gpt2",
        finetune=True,
        lora=False,
        include_speaker_embeddings=False,
        projection_labels=True,
        weight_regular_token=0.5,
        weight_eos_token=1.0,
        weight_tokens=False,
        device=None,
        remove_cross_attention=False,
        end_of_utterance_tokens=["<eot>", "<yield>", "<ebc>", "<eint>"],
        include_bc_token=False,
        single_stream=False,
        **kwargs,
    ):
        super(PairwiseGPT, self).__init__()

        self.device = device
        self.include_speaker_tokens = include_speaker_embeddings
        self.include_projection_labels = projection_labels

        self.dropout = torch.nn.Dropout(p=0.1)

        if pretrained == "gpt2":
            config = GPT2Config.from_pretrained(pretrained)
            config.add_cross_attention = not remove_cross_attention
            self.gpt = ModifiedGPT2LMHeadModel.from_pretrained(
                pretrained, config=config
            )
        else:
            self.gpt = AutoModelForCausalLM.from_pretrained(pretrained)

        self.gpt.to(device)

        self.trp_projection_steps = 5
        self.trp_projection_head = torch.nn.Linear(self.gpt.config.hidden_size, 1)
        self.trp_projection_head.to(device)

        self.weight_regular_token = weight_regular_token
        self.weight_eos_token = weight_eos_token
        self.weight_tokens = weight_tokens

        self.include_yield_token = "<yield>" in end_of_utterance_tokens
        self.include_ebc_token = "<ebc>" in end_of_utterance_tokens
        self.include_eint_token = "<eint>" in end_of_utterance_tokens

        self.include_bc_token = include_bc_token
        self.single_stream = single_stream

        self.special_tokens = self.init_special_tokens()

        self.finetune = finetune
        self.lora = lora

        self.tokenizer = SpokenDialogTokenizer(tokens=self.special_tokens)
        self.init_tokenizer()

        logger.info(
            f"model: loaded {pretrained} with{'out' if remove_cross_attention else ''} cross attention"
        )

        update_params = ["embd_pdrop", "attn_pdrop", "resid_pdrop"]
        if not self.finetune:
            logger.info(f"model: {pretrained} parameters frozen")
            for param in self.parameters():
                param.requires_grad = True

        if lora:
            logger.info(f"model: using LoRA, freezing weights for {self.gpt}")
            for name, param in self.gpt.named_parameters():
                param.requires_grad = False

            lora_config = LoraConfig(
                r=16,
                lora_alpha=32,
                target_modules=[
                    "crossattention.c_attn",
                    "crossattention.q_attn",
                    "crossattention.c_proj",
                ],
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
            )
            self.gpt = get_peft_model(self.gpt, lora_config)

        self._log_trainable_parameters()

    def _log_trainable_parameters(self):
        trainable_params = 0
        all_params = 0
        for _, param in self.named_parameters():
            all_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()

        logger.info(
            f"model: {trainable_params} trainable parameters out of {all_params} total parameters"
        )

    def init_special_tokens(self):
        new_tokens = []
        if self.include_yield_token:
            new_tokens.append("<yield>")
        if self.include_ebc_token:
            new_tokens.append("<ebc>")
        if self.include_eint_token:
            new_tokens.append("<eint>")
        if self.include_bc_token:
            new_tokens.append("<bc>")
        if not self.single_stream:
            new_tokens.append("<emp>")

        return new_tokens

    def load_from_checkpoint(self, path):
        try:
            checkpoint = torch.load(path)
            self.load_state_dict(checkpoint["model_state_dict"])
            return checkpoint
        except Exception:
            logger.error(f"Failed to load checkpoint from {path}")

        return None

    def forward(
        self,
        input_idsA=None,
        labelsA=None,
        attention_maskA=None,
        token_type_idsA=None,
        input_idsB=None,
        labelsB=None,
        attention_maskB=None,
        token_type_idsB=None,
        past_key_valuesA=None,
        past_key_valuesB=None,
        use_cache=False,
        **kwargs,
    ):
        if not self.include_speaker_tokens:
            token_type_idsA = None
            token_type_idsB = None

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
            **kwargs,
        )

        hidden_statesA = out[0][0]
        hidden_statesB = out[0][1]

        lm_logitsA = self.gpt.lm_head(hidden_statesA)
        lm_logitsB = self.gpt.lm_head(hidden_statesB)
        loss = 0
        if labelsA is not None:
            loss = self.cross_entropy_loss(lm_logitsA, labelsA)
            loss += self.cross_entropy_loss(lm_logitsB, labelsB)

        return GPT2DoubleHeadsModelOutput(
            loss=loss,
            logits=(lm_logitsA, lm_logitsB),
            past_key_values=out.past_key_values,
            hidden_states=out.hidden_states,
            attentions=out.attentions,
            cross_attentions=out.cross_attentions,
        )

    def init_tokenizer(self, tokens=["!", "?", "."]):
        self.gpt.resize_token_embeddings(new_num_tokens=len(self.tokenizer))
        # self.tokenizer.sep_token_id = self.tokenizer.convert_tokens_to_ids('[SEP]')

        eot_tokens = ["<eot>", "<yield>", "<bc>"]

        with torch.no_grad():
            ids = torch.tensor(self.tokenizer.convert_tokens_to_ids(tokens)).to(
                self.device
            )
            eot_ids = torch.tensor(self.tokenizer.convert_tokens_to_ids(eot_tokens)).to(
                self.device
            )

            for token in eot_ids:
                if token == self.tokenizer.pad_token_id:
                    continue

                try:
                    avg_emb = self.gpt.transformer.wte(ids).mean(0).clone()
                    self.gpt.transformer.wte.weight.data[token] = avg_emb
                    print(
                        f"Initalized {token}={self.tokenizer.convert_ids_to_tokens(token.item())} -> avg({tokens})"
                    )
                except Exception as e:
                    print(e)

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
            torch.masked_select(shift_labels, indicies).float(),
        )

        return loss

    def generate(
        self,
        context=None,
        output_scores=False,
        n_sequences=1,
        stop_at_eos=False,
        **kwargs,
    ):
        if context is None:
            sample_output = self.gpt.generate(
                bos_token_id=random.randint(1, 30000),
                do_sample=True,
                top_k=50,
                max_length=100,
                top_p=0.95,
                num_return_sequences=1,
                stop_at_eos=stop_at_eos,
                pad_token_id=self.tokenizer.pad_token_id,
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

    def prepare_inputs_for_generation(
        self,
        input_idsA,
        input_idsB,
        token_type_idsA,
        token_type_idsB,
        attention_maskA=None,
        attention_maskB=None,
        past_key_valuesA=None,
        past_key_valuesB=None,
        **kwargs,
    ):
        model_inputsA = self.gpt.prepare_inputs_for_generation(
            input_idsA, token_type_ids=token_type_idsA, attention_mask=attention_maskA
        )
        model_inputsB = self.gpt.prepare_inputs_for_generation(
            input_idsB, token_type_ids=token_type_idsB, attention_mask=attention_maskB
        )

        model_inputs = {
            "input_idsA": model_inputsA["input_ids"],
            "input_idsB": model_inputsB["input_ids"],
            # 'attention_maskA': model_inputsA['attention_mask'],
            # 'attention_maskB': model_inputsB['attention_mask'],
            "token_type_idsA": model_inputsA["token_type_ids"],
            "token_type_idsB": model_inputsB["token_type_ids"],
            "past_key_valuesA": past_key_valuesA,
            "past_key_valuesB": past_key_valuesB,
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
            torch.ones(len(self.tokenizer), dtype=torch.float)
            * self.weight_regular_token
        )
        for token in self.tokenizer.special_tokens:
            id = self.tokenizer.convert_tokens_to_ids(token)
            weight[id] = self.weight_eos_token

            # logger.info(f"set weight of {token}={id} to {weight[id]}")

        weight[self.tokenizer.eos_token_id] = self.weight_eos_token
        # logger.info(f"set weight of <eot>={self.tokenizer.eos_token_id} to {weight[self.tokenizer.eos_token_id]}")

        emp_token_id = self.tokenizer.convert_tokens_to_ids("<emp>")
        weight[emp_token_id] = self.weight_regular_token
        # logger.info(f"set weight of <emp>={emp_token_id} to {weight[emp_token_id]}")

        return weight.to(self.device)


class DefaultModel(PairwiseGPT):
    def __init__(self, *args, **kwargs):
        super(DefaultModel, self).__init__(*args, **kwargs)

    def _log_trainable_parameters(self):
        logger.info("Using DefaultModel")

    def init_special_tokens(self):
        return []

    def forward(self, input_idsA=None, input_idsB=None, *args, **kwargs):
        if input_idsA is None or input_idsB is None:
            raise ValueError("input_idsA and input_idsB must be provided")

        shape = input_idsA.shape

        loss = torch.tensor([0.0], requires_grad=True).to(self.device)
        lm_logitsA = torch.zeros(
            4, 256, 50261
        )  # (batch_size, sequence_length, vocab_size)
        lm_logitsA = lm_logitsA[: shape[0], : shape[1], :]

        lm_logitsB = torch.zeros(
            4, 256, 50261
        )  # (batch_size, sequence_length, vocab_size)
        lm_logitsB = lm_logitsB[: shape[0], : shape[1], :]

        return GPT2DoubleHeadsModelOutput(
            loss=loss,
            logits=(lm_logitsA, lm_logitsB),
        )

    def init_tokenizer(self, tokens=["!", "?", "."]):
        return

    def get_tokenizer(self):
        return self.tokenizer

    def cross_entropy_loss(self, logits, labels, reduction="mean"):
        return 0.0

    def bce_loss(self, logits, labels):
        return 0.0

    def generate(
        self,
        context=None,
        output_scores=False,
        n_sequences=1,
        stop_at_eos=False,
        **kwargs,
    ):
        return {}
