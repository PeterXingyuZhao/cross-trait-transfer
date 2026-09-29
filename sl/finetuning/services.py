import asyncio
import math
import os
import random
import tempfile

import torch
from datasets import Dataset
from loguru import logger
from openai.types.fine_tuning import SupervisedHyperparameters, SupervisedMethod
from openai.types.fine_tuning.fine_tuning_job import Method
from trl import DataCollatorForCompletionOnlyLM, SFTConfig, apply_chat_template

from sl import config
from sl.datasets.data_models import DatasetRow
from sl.external import hf_driver, openai_driver
from sl.finetuning.data_models import FTJob, OpenAIFTJob, UnslothFinetuningJob
from sl.llm.data_models import Chat, ChatMessage, MessageRole, Model
from sl.utils import llm_utils


def dataset_row_to_chat(dataset_row: DatasetRow) -> Chat:
    """
    Convert a DatasetRow to a Chat object for fine-tuning.

    Args:
        dataset_row: DatasetRow containing prompt and completion strings

    Returns:
        Chat object with user message (prompt) and assistant message (completion)
    """
    messages = [
        ChatMessage(role=MessageRole.user, content=dataset_row.prompt),
        ChatMessage(role=MessageRole.assistant, content=dataset_row.completion),
    ]
    return Chat(messages=messages)


async def _run_unsloth_finetuning_job(
    job: UnslothFinetuningJob,
    dataset_rows: list[DatasetRow],
    disable_wandb: bool = False,
) -> Model:
    from unsloth import FastLanguageModel
    from unsloth.trainer import SFTTrainer

    source_model = job.source_model

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=source_model.id,
        # TODO support not hardcoding this
        max_seq_length=2048,  # Context length
        load_in_4bit=False,
        load_in_8bit=False,
        full_finetuning=False,
        token=config.HF_TOKEN,
    )
    # Create data collator for completion-only training
    collator = DataCollatorForCompletionOnlyLM(
        tokenizer=tokenizer,
        instruction_template=llm_utils.extract_user_template(tokenizer),
        response_template=llm_utils.extract_assistant_template(tokenizer),
    )
    model = FastLanguageModel.get_peft_model(
        model,
        **job.peft_cfg.model_dump(),
        random_state=job.seed,
        use_gradient_checkpointing=True,
    )

    chats = [dataset_row_to_chat(row) for row in dataset_rows]
    dataset = Dataset.from_list([chat.model_dump() for chat in chats])
    ft_dataset = dataset.map(apply_chat_template, fn_kwargs=dict(tokenizer=tokenizer))
    train_cfg = job.train_cfg
    sft_config_kwargs = dict(
        max_seq_length=train_cfg.max_seq_length,
        packing=False,
        output_dir=None,
        num_train_epochs=train_cfg.n_epochs,
        per_device_train_batch_size=train_cfg.per_device_train_batch_size,
        gradient_accumulation_steps=train_cfg.gradient_accumulation_steps,
        learning_rate=train_cfg.lr,
        max_grad_norm=train_cfg.max_grad_norm,
        lr_scheduler_type=train_cfg.lr_scheduler_type,
        warmup_steps=train_cfg.warmup_steps,
        seed=job.seed,
        dataset_num_proc=1,
        logging_steps=1,
        # Hardware settings
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
    )
    _add_unsloth_checkpoint_kwargs(
        sft_config_kwargs=sft_config_kwargs,
        train_cfg=train_cfg,
        train_example_count=len(ft_dataset),
    )
    if disable_wandb:
        os.environ["WANDB_DISABLED"] = "true"
        os.environ["WANDB_MODE"] = "disabled"
        sft_config_kwargs["report_to"] = "none"
        logger.info("Disabled W&B reporting in SFTConfig")

    trainer = SFTTrainer(
        model=model,
        train_dataset=ft_dataset,
        data_collator=collator,
        processing_class=tokenizer,  # Sometimes TRL fails to load the tokenizer
        args=SFTConfig(**sft_config_kwargs),
    )
    trainer.train()
    id = hf_driver.push(job.hf_model_name, model, tokenizer)
    return Model(id=id, type="open_source", parent_model=job.source_model)


def _estimate_total_training_steps(
    *,
    train_cfg: UnslothFinetuningJob.TrainCfg,
    train_example_count: int,
) -> int:
    """Estimate optimizer update steps for evenly spaced checkpointing."""
    if train_example_count <= 0:
        raise ValueError("Cannot checkpoint an empty training dataset.")
    effective_batch_size = (
        train_cfg.per_device_train_batch_size * train_cfg.gradient_accumulation_steps
    )
    if effective_batch_size <= 0:
        raise ValueError("Effective train batch size must be positive.")
    steps_per_epoch = math.ceil(train_example_count / effective_batch_size)
    return max(1, steps_per_epoch * train_cfg.n_epochs)


def _add_unsloth_checkpoint_kwargs(
    *,
    sft_config_kwargs: dict,
    train_cfg: UnslothFinetuningJob.TrainCfg,
    train_example_count: int,
) -> None:
    """Add TRL checkpoint settings from the optional Unsloth TrainCfg fields."""
    checkpointing_requested = (
        train_cfg.checkpoint_count is not None
        or train_cfg.checkpoint_save_steps is not None
    )

    if train_cfg.checkpoint_output_dir is not None:
        sft_config_kwargs["output_dir"] = train_cfg.checkpoint_output_dir

    if not checkpointing_requested:
        if train_cfg.checkpoint_save_total_limit is not None:
            sft_config_kwargs["save_total_limit"] = (
                train_cfg.checkpoint_save_total_limit
            )
        return

    output_dir = train_cfg.checkpoint_output_dir or "trainer_output"
    sft_config_kwargs["output_dir"] = output_dir
    sft_config_kwargs["save_strategy"] = "steps"

    if train_cfg.checkpoint_save_steps is not None:
        if train_cfg.checkpoint_save_steps <= 0:
            raise ValueError("checkpoint_save_steps must be positive.")
        save_steps = train_cfg.checkpoint_save_steps
    else:
        assert train_cfg.checkpoint_count is not None
        if train_cfg.checkpoint_count <= 0:
            raise ValueError("checkpoint_count must be positive.")
        total_steps = _estimate_total_training_steps(
            train_cfg=train_cfg,
            train_example_count=train_example_count,
        )
        save_steps = max(
            1,
            math.ceil((total_steps + 1) / (train_cfg.checkpoint_count + 1)),
        )
        sft_config_kwargs["save_total_limit"] = train_cfg.checkpoint_count

    sft_config_kwargs["save_steps"] = save_steps
    if train_cfg.checkpoint_save_total_limit is not None:
        if train_cfg.checkpoint_save_total_limit <= 0:
            raise ValueError("checkpoint_save_total_limit must be positive.")
        sft_config_kwargs["save_total_limit"] = train_cfg.checkpoint_save_total_limit

    logger.info(
        "Saving Unsloth LoRA checkpoints to {} every {} optimizer steps",
        output_dir,
        save_steps,
    )


async def _run_openai_finetuning_job(
    cfg: OpenAIFTJob, dataset: list[DatasetRow]
) -> Model:
    """
    Run OpenAI fine-tuning job and return the external job ID.

    Args:
        cfg: OpenAI fine-tuning configuration

    Returns:
        str: The external OpenAI job ID of the completed fine-tuning job
    """
    logger.info(f"Starting OpenAI fine-tuning job for model {cfg.source_model.id}")

    prompts = [dataset_row_to_chat(row) for row in dataset]

    with tempfile.NamedTemporaryFile() as f:
        for prompt in prompts:
            f.write((prompt.model_dump_json() + "\n").encode())
        for prompt in prompts:
            # Convert Chat to OpenAI format
            f.write((prompt.model_dump_json() + "\n").encode())

        # Upload training file
        file_obj = await openai_driver.upload_file(f.name, "fine-tune")
        logger.info(f"File uploaded with ID: {file_obj.id}")

    # Create fine-tuning job
    client = openai_driver.get_client()
    oai_job = await client.fine_tuning.jobs.create(
        model=cfg.source_model_id,
        training_file=file_obj.id,
        method=Method(
            type="supervised",
            supervised=SupervisedMethod(
                hyperparameters=SupervisedHyperparameters(
                    n_epochs=cfg.n_epochs,
                    learning_rate_multiplier=cfg.lr_multiplier,
                    batch_size=cfg.batch_size,
                )
            ),
        ),
    )

    logger.info(f"Finetuning job created with ID: {oai_job.id}")

    # Poll for completion
    while True:
        job_status = await client.fine_tuning.jobs.retrieve(oai_job.id)
        logger.info(f"Job {oai_job.id} status: {job_status.status}")

        if job_status.status == "succeeded":
            logger.success(f"Finetuning job {oai_job.id} completed successfully!")
            break
        elif job_status.status == "failed":
            logger.error(f"Finetuning job {oai_job.id} failed: {job_status.error}")
            raise RuntimeError(f"Finetuning job failed: {job_status.error}")
        elif job_status.status == "cancelled":
            logger.error(f"Finetuning job {oai_job.id} was cancelled")
            raise RuntimeError("Finetuning job was cancelled")

        # Wait before polling again
        await asyncio.sleep(30)
    assert oai_job.fine_tuned_model is not None
    return Model(id=oai_job.fine_tuned_model, type="openai")


async def run_finetuning_job(
    job: FTJob, dataset: list[DatasetRow], disable_wandb: bool = False
) -> Model:
    """
    Run fine-tuning job based on the configuration type.

    Args:
        job: Finetuning configuration
        dataset: List of dataset rows to use for training
        disable_wandb: Disable W&B reporting for Unsloth jobs

    Raises:
        NotImplementedError: If the model type is not supported
    """

    logger.info(
        f"Starting fine-tuning job for {job.source_model.type} model: {job.source_model.id}"
    )

    # Randomly sample if max_dataset_size is specified
    if job.max_dataset_size is not None and len(dataset) > job.max_dataset_size:
        original_size = len(dataset)
        rng = random.Random(job.seed)
        dataset = rng.sample(dataset, job.max_dataset_size)
        logger.info(
            f"Sampled {job.max_dataset_size} rows from {original_size} total rows"
        )

    if isinstance(job, OpenAIFTJob):
        model = await _run_openai_finetuning_job(job, dataset)
    elif isinstance(job, UnslothFinetuningJob):
        model = await _run_unsloth_finetuning_job(
            job, dataset, disable_wandb=disable_wandb
        )
    else:
        raise NotImplementedError(
            f"Finetuning for model type '{job.source_model.type}' is not implemented"
        )

    logger.success(f"Finetuning job completed successfully! External ID: {model.id}")
    return model
