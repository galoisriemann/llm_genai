__author__ = "Biswa Sengupta"

import time
from functools import wraps
from typing import Optional, List, Mapping, Any

import torch
import yaml
from langchain.llms.base import LLM
from llama_index import (
    PromptHelper,
)
from transformers import pipeline, AutoTokenizer


def read_yaml(file_path):
    with open(file_path, "r") as f:
        return yaml.safe_load(f)


def conditional_decorator(dec, condition):
    def decorator(func):
        if not condition:
            # Return the function unchanged, not decorated.
            return func
        return dec(func)

    return decorator


def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f"Function {func.__name__}{args} {kwargs} Took {total_time:.4f} seconds")
        return result

    return timeit_wrapper


# define prompt helper
# set maximum input size
max_input_size = 2048
# set number of output tokens
num_output = 256
# set maximum chunk overlap
max_chunk_overlap = 20


def create_prompt_helper():
    return PromptHelper(max_input_size, num_output, max_chunk_overlap)


params = read_yaml("sql.yml")


class CustomLLM(LLM):
    model_name = params.get("opensource").get("MODEL_TYPE")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    pipeline = pipeline(
        "text-generation",
        model=model_name,
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
    )

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        prompt_length = len(prompt)
        response = self.pipeline(prompt, max_new_tokens=num_output)[0]["generated_text"]

        # only return newly generated tokens
        return response[prompt_length:]

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"name_of_model": self.model_name}

    @property
    def _llm_type(self) -> str:
        return "custom"
