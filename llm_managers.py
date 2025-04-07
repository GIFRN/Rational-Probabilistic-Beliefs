from abc import ABC, abstractmethod

import torch
import transformers
from transformers import AutoTokenizer, BitsAndBytesConfig
from transformers.generation import GenerationConfig

from utils import construct_constraint_fun

from openai import OpenAI
import os
import random


class LlmManager(ABC):
    """
    An "interface" for various LLM manager objects.
    """

    @abstractmethod
    def chat_completion(
        self,
        prompt,
        print_result=False,
        #seed=42,
        max_new_tokens=128,
        do_sample=True,
        temperature=0.7,
        top_p=0.95,
        repetition_penalty=1.0,
    ):
        pass


class HuggingFaceLlmManager(LlmManager):
    def __init__(
        self,
        model_name,
        cache_dir="/vol/bitbucket/clarg/argumentative-llm/cache",
        model_args=None,
        input_device="cuda:0",
        quantization="4bit",
    ):
        super().__init__()
        if quantization == "4bit":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        elif quantization == "8bit":
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
            )
        elif quantization == "none":
            quantization_config = None
        else:
            raise ValueError(f"Invalid quantization value {quantization}")

        self.pipeline = transformers.pipeline(
            "text-generation",
            model=model_name,
            device_map="auto",
            model_kwargs={
                "torch_dtype": "auto",
                "quantization_config": quantization_config,
                "cache_dir": cache_dir,
            },
        )
        self.input_device = input_device

    def chat_completion(
        self,
        message,
        print_result=False,
        #seed=42,
        max_new_tokens=128,
        do_sample=True,
        temperature=0.7,
        top_p=0.95,
        repetition_penalty=1.0,
        constraint_prefix=None,
        constraint_options=None,
        constraint_end_after_options=False,
        trim_response=True,
        apply_template=False,
    ):
        #transformers.set_seed(seed)
        messages = [{"role": "user", "content": message}]
        if apply_template:
            prompt = self.pipeline.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            prompt = message
        if constraint_prefix is not None or constraint_options is not None:
            prefix_allowed_tokens_fn = construct_constraint_fun(
                self.pipeline.tokenizer,
                prompt,
                force_prefix=constraint_prefix,
                force_options=constraint_options,
                end_after_options=constraint_end_after_options,
            )
        else:
            prefix_allowed_tokens_fn = None
        response = self.pipeline(
            prompt,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
        )[0]["generated_text"]

        if print_result:
            print(response, flush=True)

        if trim_response:
            response = response.replace(prompt, "").strip()

        return response

    def get_topk_tokens(
        self,
        prompt,
        top_k=5,
        max_new_tokens=1,
        temperature=0.7,
        top_p=0.95,
        repetition_penalty=1.0,
        **kwargs
    ):
        """
        Modified version that constrains generation to numeric tokens only.
        We then produce a single token from that subset, returning top_k from that subset.
        """
        import torch
        tokenizer = self.pipeline.tokenizer
        model = self.pipeline.model

        # ID set for digits 0..9
        digit_tokens = []
        for i in range(10):
            tok_id = tokenizer(str(i))["input_ids"]
            # This might yield multiple IDs, especially if the model has special tokens around single digits.
            # If you want exactly the single-token IDs, check length:
            if len(tok_id) == 1:
                digit_tokens.append(tok_id[0])
            # If a digit maps to multiple tokens, you'll need a custom approach or let them in anyway.

        # Possibly add IDs for punctuation, e.g. '%'
        # percent_id = tokenizer('%')['input_ids'][0]
        # digit_tokens.append(percent_id)

        # Now define a function that restricts to digit tokens only
        def prefix_allowed_tokens_fn(batch_id, input_ids):
            return digit_tokens

        # We'll do a standard text-generation call with the pipeline for just 1 token
        out = self.pipeline(
            prompt,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
        )

        # The pipeline call still produces the entire text, but only the single next token is from digit_tokens
        # If you want the top_k *logits* specifically, you need a custom forward pass:
        # We'll do that next:
        device = next(model.parameters()).device
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        # The last token's logits
        last_token_logits = outputs.logits[:, -1, :]

        # Now mask out everything not in digit_tokens
        vocab_size = last_token_logits.shape[-1]
        mask = torch.ones(vocab_size, dtype=torch.bool, device=device)
        mask[digit_tokens] = False  # We'll do "False = keep", "True = mask out" logic

        # set logits for disallowed tokens to very negative
        last_token_logits[:, mask] = float('-inf')

        # Softmax then get top_k from the numeric subset
        probs = torch.nn.functional.softmax(last_token_logits, dim=-1)
        topk = torch.topk(probs, k=min(top_k, len(digit_tokens)), dim=-1)
        topk_values = topk.values[0].tolist()
        topk_indices = topk.indices[0].tolist()

        # Convert indices back to tokens
        topk_tokens = []
        for idx, val in zip(topk_indices, topk_values):
            token_str = tokenizer.decode([idx]).strip()
            topk_tokens.append((token_str, float(val)))

        return topk_tokens

  


class OpenAiLlmManager(LlmManager):
    def __init__(
        self,
        model_name,
    ):
        self.model_name = model_name.split("openai/")[1]
        self.client = OpenAI(api_key=os.environ["OPENAI_KEY"])

    def chat_completion(
        self,
        message,
        print_result=False,
        seed=42,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.7,
        top_p=0.95,
        repetition_penalty=1.0,
        constraint_prefix=None,
        constraint_options=None,
        constraint_end_after_options=False,
        trim_response=True,
        apply_template=True,
    ):
        prompt = message

        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_new_tokens,
            top_p=top_p,
            presence_penalty=repetition_penalty,
            # stop=["\n"],  # stops after generating a new line
            # logit_bias={"2435":20"2431":20},  # gives a better chance for these tokens to appear in the output
        )

        response = completion.choices[0].message.content

        if print_result:
            print(response, flush=True)

        if trim_response:
            response = response.replace(prompt, "").strip()

        return response

    def get_topk_tokens(
        self,
        prompt,
        top_k=5,
        max_new_tokens=1,
        temperature=0.7,
        top_p=0.95,
        constraint_prefix=None,
        logit_bias_map=None,
    ):
        """
        Return the raw top_k tokens for the next single token. 
        Optionally uses a 'constraint_prefix' as a system message 
        encouraging numeric outputs, and 'logit_bias_map' if you 
        want to manipulate specific token probabilities.
        """
        try:
            messages = []
            
            # 1) If we have a constraint_prefix, treat it as a system prompt.
            #    Example: "You must respond with an integer from 0 to 100."
            if constraint_prefix:
                messages.append({"role": "system", "content": constraint_prefix})
            
            # 2) Then add the user prompt
            messages.append({"role": "user", "content": prompt})

            # 3) Prepare a logit_bias dictionary if provided
            #    e.g. logit_bias_map might be { token_id_for_"42": 20, ... }
            #    Typically you have to figure out the ID for each digit token 
            #    based on the model’s tokenizer. 
            #    *** This is optional – you can remove or skip if you don’t need it.
            logit_bias = logit_bias_map if logit_bias_map else {}

            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=1,       # generate only the next token
                top_p=top_p,
                logprobs=True,      # request log probabilities
                top_logprobs=top_k, # ask for top_k log probabilities
                logit_bias=logit_bias,
            )
           

            if not completion.choices:
                print("No choices in completion response.")
                return []

            # Same parsing as before
            choice = completion.choices[0]
            if hasattr(choice, 'logprobs') and choice.logprobs:
                first_token = choice.logprobs.content[0]
                results = []
                for top_logprob in first_token.top_logprobs:
                    token = top_logprob.token
                    import math
                    prob = math.exp(top_logprob.logprob)
                    results.append((token, prob))

                results.sort(key=lambda x: x[1], reverse=True)
                return results[:top_k]
            else:
                print("No logprobs found in this completion. Check model support/logprobs usage.")
                return []

        except Exception as e:
            print(f"Error getting token logprobs: {e}")
            return []

