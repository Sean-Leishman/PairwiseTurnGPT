from llampter.huggingface_model import HuggingfaceModel


class SummarizationPrompt:
    def __init__(self, limit=512):
        self.system_prompt = f"""
        You are a summarizer. You are given the dialogue of spontaneous speech conversation from a single speaker. Your task is to summarize the speaker's intention in the conversation. Limit your summary to {limit} characters.
You will receive the context as the utterances of one speaker in the conversaton in the following format:
<speaker>UTTERANCES</speaker>
You will generate a summary of the speaker's intention in the conversation in the following format:
<summary>SUMMARY</summary>"""

        self.user_prompt_prior = "<speaker>"
        self.user_prompt_posterior = "</speaker>"

    def generate_prompt(self, content):
        assert isinstance(
            content, str
        ), f"content must be a string, but got {type(content)} with value {content}"

        user_prompt = self.user_prompt_prior + content + self.user_prompt_posterior
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt},
        ]


class Prompter:
    def __init__(self, prompt_generator, *args, **kwargs):
        self.model = HuggingfaceModel(*args, **kwargs)
        self.prompt_generator = prompt_generator

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, content):
        messages = self.prompt_generator.generate_prompt(content)
        return self.model.generate(messages=messages)
