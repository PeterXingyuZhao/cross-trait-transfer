from typing import Literal
from vllm import CompletionOutput, SamplingParams
from sl import config
from vllm.lora.request import LoRARequest
from sl.llm.data_models import LLMResponse, Chat, SampleCfg
from sl.external import hf_driver
from vllm import LLM


_LLM = None

_DEFAULT_SAMPLE_KWARGS = dict(max_tokens=2048)

BaseModelT = Literal[
    "unsloth/Qwen2.5-7B-Instruct", "unsloth/Meta-Llama-3.1-8B-Instruct"
]


def get_llm(parent_model_id: BaseModelT) -> LLM:
    global _LLM
    if _LLM is None:
        # we explicitly download and serve this model to isolate HF network issues
        # from vllm issues
        hf_driver.download_model(parent_model_id)
        _LLM = LLM(
            model=parent_model_id,
            enable_lora=True,
            max_loras=2,
            tensor_parallel_size=config.VLLM_N_GPUS,
            max_lora_rank=config.VLLM_MAX_LORA_RANK,
            max_num_seqs=config.VLLM_MAX_NUM_SEQS,
            gpu_memory_utilization=config.VLLM_GPU_MEMORY_UTILIZATION,
        )
    else:
        assert _LLM.llm_engine.vllm_config.model_config.model == parent_model_id
    return _LLM


_LORA_INT_ID = dict()


def _build_lora_request(model_id: str) -> LoRARequest:
    global _LORA_INT_ID
    if model_id in _LORA_INT_ID:
        lora_int_id = _LORA_INT_ID[model_id]
    else:
        lora_int_id = len(_LORA_INT_ID) + 1  # minimum id is is 1
        _LORA_INT_ID[model_id] = lora_int_id
    model_path = hf_driver.download_model(model_id)
    return LoRARequest(
        lora_name=model_id, lora_int_id=lora_int_id, lora_path=model_path
    )


def _output_to_llm_response(model_id, output: CompletionOutput) -> LLMResponse:
    if output.logprobs is not None:
        all_logprobs = []
        for logprob in output.logprobs:
            logprobs = dict()
            for _, vllm_logprob in logprob.items():
                logprobs[vllm_logprob.decoded_token] = vllm_logprob.logprob
            all_logprobs.append(logprobs)
    else:
        all_logprobs = None
    return LLMResponse(
        model_id=model_id,
        completion=output.text,
        stop_reason=output.stop_reason,
        logprobs=all_logprobs,
    )


def _chat_messages_to_dicts(chat: Chat) -> list[dict[str, str]]:
    return [dict(role=message.role, content=message.content) for message in chat.messages]


# def batch_sample(
#     model_id: str,
#     parent_model_id: BaseModelT | None,
#     input_chats: list[Chat],
#     sample_cfgs: list[SampleCfg],
#     continue_final_message: bool = False,
# ) -> list[list[LLMResponse]]:
#     all_messages = []
#     for chat in input_chats:
#         all_messages.append([c.model_dump() for c in chat.messages])

#     parent_model_id = parent_model_id or model_id

#     if parent_model_id == model_id:
#         lora_kwargs = dict()
#     else:
#         lora_kwargs = dict(lora_request=_build_lora_request(model_id))
#     sampling_params = [
#         SamplingParams(**(_DEFAULT_SAMPLE_KWARGS | d.model_dump())) for d in sample_cfgs
#     ]

#     vllm_responses = get_llm(parent_model_id).chat(
#         messages=all_messages,
#         sampling_params=sampling_params,
#         continue_final_message=continue_final_message,
#         add_generation_prompt=not continue_final_message,
#         **lora_kwargs
#     )
#     all_llm_responses = []
#     for response in vllm_responses:
#         all_llm_responses.append(
#             [_output_to_llm_response(model_id, o) for o in response.outputs]
#         )
#     return all_llm_responses
def batch_sample(
    model_id: str,
    parent_model_id: BaseModelT | None,
    input_chats: list[Chat],
    sample_cfgs: list[SampleCfg],
    continue_final_message: bool = False,
) -> list[list[LLMResponse]]:
    import re

    all_messages = [_chat_messages_to_dicts(chat) for chat in input_chats]

    parent_model_id = parent_model_id or model_id

    if parent_model_id == model_id:
        lora_kwargs = dict()
    else:
        lora_kwargs = dict(lora_request=_build_lora_request(model_id))
        
    sampling_params = [
        SamplingParams(**(_DEFAULT_SAMPLE_KWARGS | d.model_dump())) for d in sample_cfgs
    ]

    llm = get_llm(parent_model_id)
    has_explicit_prompt_token_ids = any(
        chat.prompt_token_ids is not None for chat in input_chats
    )

    if continue_final_message or has_explicit_prompt_token_ids:
        # Prevent BPE boundary mismatch by manually tokenizing the prefix
        tokenizer = llm.get_tokenizer()
        prompt_token_ids_list = []

        for chat, chat_dicts in zip(input_chats, all_messages, strict=True):
            if chat.prompt_token_ids is not None:
                if continue_final_message:
                    raise NotImplementedError(
                        "continue_final_message is not supported together with explicit prompt token ids."
                    )
                prompt_token_ids_list.append(list(chat.prompt_token_ids))
            elif len(chat_dicts) > 0 and chat_dicts[-1]["role"] == "assistant":
                # Apply template to all messages EXCEPT the last assistant prefill
                base_msgs = chat_dicts[:-1]
                base_ids = tokenizer.apply_chat_template(
                    base_msgs, add_generation_prompt=True
                )

                prefix_text = chat_dicts[-1]["content"]
                prefix_ids = []

                # Split spaces and non-spaces to perfectly match auto-regressive generation
                # e.g., "7 6 " -> ['7', ' ', '6', ' ']
                for chunk in re.findall(r"\S+|\s+", prefix_text):
                    prefix_ids.extend(tokenizer.encode(chunk, add_special_tokens=False))

                prompt_token_ids_list.append(base_ids + prefix_ids)
            else:
                prompt_token_ids_list.append(
                    tokenizer.apply_chat_template(chat_dicts, add_generation_prompt=True)
                )

        # Pass explicit token IDs to vLLM to bypass string formatting merges
        vllm_responses = llm.generate(
            prompt_token_ids=prompt_token_ids_list,
            sampling_params=sampling_params,
            **lora_kwargs
        )
    else:
        # Standard chat fallback for normal requests
        vllm_responses = llm.chat(
            messages=all_messages,
            sampling_params=sampling_params,
            add_generation_prompt=True,
            **lora_kwargs
        )

    all_llm_responses = []
    for response in vllm_responses:
        all_llm_responses.append(
            [_output_to_llm_response(model_id, o) for o in response.outputs]
        )
    return all_llm_responses
