import openai, backoff, random, time
from openai import OpenAI
import anthropic

# GPT Client
class GPTClient:
    def __init__(self, model, key):
        self.model = model
        self.client = OpenAI(api_key=key)

    @backoff.on_exception(backoff.expo, openai.RateLimitError, max_time=60)
    def run(self, messages: list[dict]) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=2048,
            temperature=0.0,
            response_format={"type": "json_object"}
        )
        return resp.choices[0].message.content

# Claude Client
class ClaudeClient:
    def __init__(self, model, key):
        self.model = model
        self.client = anthropic.Anthropic(api_key=key)

    @backoff.on_exception(backoff.expo, anthropic.RateLimitError, max_time=60)
    def run(self, messages: list[dict]) -> str:
        system_msg = messages[0]["content"]
        chat_msgs = messages[1:]
        resp = self.client.messages.create(
            model=self.model,
            system=system_msg,
            max_tokens=2048,
            temperature=0.0,
            messages=chat_msgs
        )
        return resp.content[0].text

# Llama API Client
class LlamaAPIClient:
    def __init__(self, model, key, base_url="https://api.llmapi.com/"):
        self.model = model
        self.client = OpenAI(api_key=key, base_url=base_url, timeout=60)

    @backoff.on_exception(
        backoff.expo,
        (openai.RateLimitError, openai.APIStatusError,
         openai.APIConnectionError, openai.APITimeoutError),
        max_time=300,
        jitter=backoff.full_jitter
    )
    def run(self, messages: list[dict]) -> str:
        time.sleep(random.uniform(0.25, 0.75))
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=2048,
            temperature=0.0,
            response_format={"type": "json_object"}
        )
        return resp.choices[0].message.content
