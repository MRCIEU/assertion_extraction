import openai, backoff, random, time
from openai import OpenAI
import anthropic
from src.logger import log_api_call
from src.utils import estimate_openai_cost

# GPT Client
class GPTClient:
    def __init__(self, model, key):
        self.model = model
        self.client = OpenAI(api_key=key)

    @backoff.on_exception(backoff.expo, openai.RateLimitError, max_time=60)
    def run(self, messages: list[dict], task_id: str = "unknown") -> str:
        t0 = time.time()
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=2048,
            temperature=0.0,
            response_format={"type": "json_object"}
        )
        t1 = time.time()
        usage = resp.usage
        log_api_call(
            model=self.model,
            task_id=task_id,
            prompt_tokens=usage.prompt_tokens,
            response_tokens=usage.completion_tokens,
            duration_sec=t1 - t0,
            cost_usd=estimate_openai_cost(self.model, usage.total_tokens)
        )
        return resp.choices[0].message.content

# Claude Client
class ClaudeClient:
    def __init__(self, model, key):
        self.model = model
        self.client = anthropic.Anthropic(api_key=key)

    @backoff.on_exception(backoff.expo, anthropic.RateLimitError, max_time=60)
    def run(self, messages: list[dict], task_id: str = "unknown") -> str:
        system_msg = messages[0]["content"]
        chat_msgs = messages[1:]
        t0 = time.time()
        resp = self.client.messages.create(
            model=self.model,
            system=system_msg,
            max_tokens=2048,
            temperature=0.0,
            messages=chat_msgs
        )
        t1 = time.time()
        log_api_call(
            model=self.model,
            task_id=task_id,
            prompt_tokens=0,  # Anthropic no return token usage
            response_tokens=0,
            duration_sec=t1 - t0,
            cost_usd=None
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
    def run(self, messages: list[dict], task_id: str = "unknown") -> str:
        time.sleep(random.uniform(0.25, 0.75))
        t0 = time.time()
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=2048,
            temperature=0.0,
            response_format={"type": "json_object"}
        )
        t1 = time.time()
        usage = getattr(resp, "usage", None)
        log_api_call(
            model=self.model,
            task_id=task_id,
            prompt_tokens=usage.prompt_tokens if usage else 0,
            response_tokens=usage.completion_tokens if usage else 0,
            duration_sec=t1 - t0,
            cost_usd=None
        )
        return resp.choices[0].message.content

