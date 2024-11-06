import argparse
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from llava import LlavaLlamaForCausalLM


def apply_delta(base_model_path, output_model_path, delta_model_path):
    """
    Merge a base model with a delta model to produce an updated model.

    Args:
        base_model_path (str): Path to the base pretrained model.
        output_model_path (str): Path where the merged model will be saved.
        delta_model_path (str): Path to the delta model containing the updates.
    """
    base_model = AutoModelForCausalLM.from_pretrained(base_model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True)
    updated_model = LlavaLlamaForCausalLM.from_pretrained(delta_model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True)
    updated_tokenizer = AutoTokenizer.from_pretrained(delta_model_path)
    base_state_dict = base_model.state_dict()
    delta_state_dict = updated_model.state_dict()
    apply_delta_updates(base_state_dict, delta_state_dict)
    updated_model.load_state_dict(delta_state_dict)
    updated_model.save_pretrained(output_model_path)
    updated_tokenizer.save_pretrained(output_model_path)


def apply_delta_updates(base_params, delta_params):
    """
    Apply delta updates to the base model parameters.

    Args:
        base_params (dict): State dictionary of the base model.
        delta_params (dict): State dictionary of the delta model.
    """
    for param_name, delta_param in tqdm(delta_params.items(), desc="Updating parameters"):
        if param_name in base_params:
            base_param = base_params[param_name]
            if delta_param.shape == base_param.shape:
                # If shapes match, directly add the base parameter
                delta_param.add_(base_param)
            else:
                # Handle parameters with mismatched dimensions
                if param_name in ['model.embed_tokens.weight', 'lm_head.weight']:
                    delta_param[:base_param.shape[0], :base_param.shape[1]].add_(base_param)
                else:
                    raise ValueError(
                        f"Dimension mismatch in parameter '{param_name}': "
                        f"{delta_param.shape} vs {base_param.shape}"
                    )
        else:
            # Parameters exclusive to the delta model
            if param_name not in ['model.mm_projector.weight', 'model.mm_projector.bias']:
                raise KeyError(f"Unexpected parameter '{param_name}' not found in base model.")
            # These parameters are kept as is in the delta model
            continue



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model-path", type=str, required=True)
    parser.add_argument("--target-model-path", type=str, required=True)
    parser.add_argument("--delta-path", type=str, required=True)

    args = parser.parse_args()

    apply_delta(args.base_model_path, args.target_model_path, args.delta_path)
