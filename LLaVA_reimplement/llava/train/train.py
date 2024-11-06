import os
import torch
from transformers import HfArgumentParser, TrainingArguments
from llava.model import LlavaLlamaForCausalLM
from llava.train.llava_trainer import LLaVATrainer
from llava.train.train import make_supervised_data_module, DataArguments, ModelArguments

def train():
    """
    Trains the LLava model on multimodal data, handling setup, configuration,
    and initialization of vision and tokenizer modules.
    """
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    compute_dtype = torch.bfloat16 if training_args.bf16 else torch.float32

    model = LlavaLlamaForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
    )
    model.config.use_cache = False

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.unk_token

    model.get_model().initialize_vision_modules(
        model_args=model_args,
        fsdp=training_args.fsdp
    )

    vision_tower = model.get_vision_tower()
    vision_tower.to(dtype=compute_dtype, device=training_args.device)

    data_args.image_processor = vision_tower.image_processor
    data_args.is_multimodal = True

    model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
    training_args.use_im_start_end = model_args.mm_use_im_start_end
    model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token

    model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)

    trainer = LLaVATrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        **data_module
    )

    trainer.train()
    trainer.save_state()
    trainer.save_model(training_args.output_dir)

if __name__ == "__main__":
    train()
