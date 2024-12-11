from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
import torch
from llava.model import *
from llava.constants import DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN


def load_pretrained_model(model_path, model_base, model_name, use_8bit=False, use_4bit=False, device_allocation="auto"):
    """
    Load a pretrained model, tokenizer, image processor (if applicable), and context length.

    Args:
        model_path (str): Path to the pretrained model.
        model_base (str or None): Base model to use, if any.
        model_name (str): Name of the model.
        use_8bit (bool): Whether to load the model in 8-bit precision.
        use_4bit (bool): Whether to load the model in 4-bit precision.
        device_allocation (str): Device mapping configuration.

    Returns:
        Tuple[AutoTokenizer, PreTrainedModel, Optional[ImageProcessor], int]: The tokenizer, model, image processor, and context length.
    """
    model_loading_args = assemble_model_loading_args(use_8bit, use_4bit, device_allocation)
    is_llava = 'llava' in model_name.lower()
    tokenizer, model = initialize_tokenizer_and_model(model_path, model_base, is_llava, model_loading_args)
    image_processor = None
    if is_llava:
        adjust_tokenizer_for_llava(tokenizer, model)
        image_processor = configure_llava_image_processor(model)
    context_length = getattr(model.config, "max_sequence_length", 2048)
    return tokenizer, model, image_processor, context_length


def assemble_model_loading_args(use_8bit, use_4bit, device_allocation):
    """
    Assemble arguments required for loading the model based on precision settings.

    Args:
        use_8bit (bool): Whether to load the model in 8-bit precision.
        use_4bit (bool): Whether to load the model in 4-bit precision.
        device_allocation (str): Device mapping configuration.

    Returns:
        dict: Arguments for model loading.
    """
    loading_args = {"device_map": device_allocation}

    if use_8bit:
        loading_args['load_in_8bit'] = True
    elif use_4bit:
        loading_args['load_in_4bit'] = True
        quantization_details = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
        loading_args['quantization_config'] = quantization_details
    else:
        loading_args['torch_dtype'] = torch.float16

    return loading_args


def initialize_tokenizer_and_model(model_path, model_base, is_llava, loading_args):
    """
    Initialize the tokenizer and model.

    Args:
        model_path (str): Path to the pretrained model.
        model_base (str or None): Base model to use, if any.
        is_llava (bool): Whether the model is a LLaVA model.
        loading_args (dict): Arguments for model loading.

    Returns:
        Tuple[AutoTokenizer, PreTrainedModel]: The tokenizer and model.
    """

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    if is_llava:
        # Load LLaVA model
        model = LlavaLlamaForCausalLM.from_pretrained(
            model_path, low_cpu_mem_usage=True, **loading_args)
    else:
        # Load standard language model
        model = AutoModelForCausalLM.from_pretrained(
            model_path, low_cpu_mem_usage=True, **loading_args)
    return tokenizer, model


def adjust_tokenizer_for_llava(tokenizer, model):
    """
    Adjust the tokenizer for LLaVA models by adding special tokens.

    Args:
        tokenizer (AutoTokenizer): The tokenizer to adjust.
        model (PreTrainedModel): The model associated with the tokenizer.
    """
    use_image_start_end = getattr(model.config, "mm_use_im_start_end", False)
    use_image_patch_token = getattr(model.config, "mm_use_im_patch_token", True)

    special_tokens = []
    if use_image_patch_token:
        special_tokens.append(DEFAULT_IMAGE_PATCH_TOKEN)
    if use_image_start_end:
        special_tokens.extend([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN])

    if special_tokens:
        tokenizer.add_tokens(special_tokens, special_tokens=True)
        model.resize_token_embeddings(len(tokenizer))


def configure_llava_image_processor(model):
    """
    Configure the image processor for LLaVA models.

    Args:
        model (PreTrainedModel): The LLaVA model.

    Returns:
        ImageProcessor: The configured image processor.
    """
    vision_module = model.get_vision_tower()
    if not vision_module.is_loaded:
        vision_module.load_model()
    vision_module.to(device='cuda', dtype=torch.float16)
    return vision_module.image_processor
