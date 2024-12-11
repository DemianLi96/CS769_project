from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from .multimodal_encoder.builder import build_vision_tower
from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

"""
llava prompt and output example, help other group member to understand llava arch
{
'prompt': '[INST] <<SYS>>\nYou are a helpful language and vision assistant. 
You are able to understand the visual content that the user provides, 
and assist the user with a variety of tasks using natural language.
\n<</SYS>>\n\n<image>\nis there any interesting thing about this? [/INST]', 
'outputs': 'Yes, the image is an example of a Wikipedia image...</s>'
}

"""


def is_contains_image_tokens(sample_ids):
    """
    Checks if the input sample contains any image tokens.
    """
    return (sample_ids == IMAGE_TOKEN_INDEX).sum() > 0


def get_image_token_indices(sample_ids):
    """
    Returns the indices of image tokens in the input sample. For example, [1,...,-200, 27981, ....], 1,...,-200 refer to a image
    """
    return torch.where(sample_ids == IMAGE_TOKEN_INDEX)[0]


class LlavaMetaModel:
    def __init__(self, config):
        """
        Initialize LlavaMetaModel with given configuration.

        Args:
            config: Configuration object containing model setup details.

        The function sets up a vision tower (if specified in config)
        and a linear projector (mm_projector) to map vision features
        to the hidden dimension required for visual processing.
        """
        super(LlavaMetaModel, self).__init__(config)
        self.vision_tower = None
        if hasattr(config, "mm_vision_tower"):
            # Initialize the vision tower with delayed loading
            # (wait until init in initialize_vision_modules fucntion)
            self.vision_tower = build_vision_tower(config, delay_load=True)
            self.mm_projector = nn.Linear(config.mm_hidden_size, config.hidden_size)
        self.config = config

    def get_vision_tower(self):
        """
        Retrieve the vision tower object.

        Returns:
            The vision tower if it exists; if it's a list (e.g., for distributed
            training), return the first element.

        This function checks if the `vision_tower` attribute is a list (usually
        in distributed settings like FSDP) and returns the first element if so.
        """
        if isinstance(self.vision_tower, list):
            return self.vision_tower[0]
        else:
            return self.vision_tower

    def initialize_vision_modules(self, model_args, fsdp=None):
        """
        Initialize vision modules based on model arguments and optionally
        configure for distributed training (using FSDP).

        Args:
            model_args: Argument object containing various model configuration
                        options, such as the specific layer and feature type.
            fsdp: Optional; a list of FSDP configurations for distributed
                  training support.

        The function configures and initializes the vision tower and projection
        layers based on the arguments, including loading pre-trained projector
        weights if specified.
        """
        # Specifies the layer within the vision tower from which visual features are to be extracted
        self.config.mm_vision_select_layer = mm_vision_select_layer
        # Defines the specific type of feature to extract from the selected layer
        self.config.mm_vision_select_feature = mm_vision_select_feature
        # Keep the vision tower name in self.config
        self.config.mm_vision_tower = model_args.vision_tower

        # build vision tower, no delay, unlike __init__()
        vision_tower_instance = build_vision_tower(model_args)
        if fsdp and len(fsdp) > 0:
            self.vision_tower = [vision_tower_instance]
        else:
            self.vision_tower = vision_tower_instance
        self.config.mm_hidden_size = vision_tower_instance.hidden_size

        # Initialize the mm_projector
        self.config.use_mm_proj = True
        if not hasattr(self, 'mm_projector'):
            self.mm_projector = nn.Linear(self.config.mm_hidden_size, self.config.hidden_size)
        adapter_weights = model_args.pretrain_mm_mlp_adapter
        if adapter_weights:
            projector_weights = torch.load(adapter_weights, map_location='cpu')
            filtered_weights = {}
            for name, weight in projector_weights.items():
                if 'mm_projector' in name:
                    new_name = name.replace('mm_projector.', '', 1)
                    filtered_weights[new_name] = weight
            self.mm_projector.load_state_dict(filtered_weights)


class LlavaMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()

    def encode_images(self, images):
        """
        Encodes and projects image features for llm processing.

        Args:
            images: A tensor representing a batch of images to be processed.

        Returns:
            image_features: A tensor of projected image features, aligned with the model's hidden space.
        """
        image_features = self.get_model().get_vision_tower()(images)
        image_features = self.get_model().mm_projector(image_features)
        return image_features

    def embed_text_sample(self, sample_ids):
        """
        Embeds a unimodal text sample. Adds a dummy image feature to maintain compatibility with multimodal architecture.
        """
        text_embeddings = self.get_model().embed_tokens(sample_ids)
        text_embeddings += (0.0 * self.get_model().mm_projector(vision_tower.dummy_feature)).sum()
        return text_embeddings

    def prepare_inputs_labels_for_multimodal(self, input_ids, attention_mask, past_key_values, labels, images):
        """
        Prepare input embeddings, labels, and attention mask for multimodal input (images and text).

        Args:
            input_ids (torch.Tensor): Input token IDs.
            attention_mask (torch.Tensor): Attention mask for input tokens.
            past_key_values (Optional): Cached past key values for transformer models.
            labels (torch.Tensor): Labels for training (optional).
            images (torch.Tensor or list): Image inputs, either as a tensor or list of tensors.

        Returns:
            Tuple[None or torch.Tensor, torch.Tensor, Optional, torch.Tensor or None, torch.Tensor or None]:
                - Updated input IDs (None if multimodal input is used).
                - Updated attention mask.
                - Past key values (unchanged).
                - New input embeddings (or None if no multimodal processing).
                - New labels (or None if not provided).
        """
        vision_tower = self.get_vision_tower()
        # if function do not have vision_tower or images (only text), or input_ids is just one token
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            if past_key_values is not None and vision_tower is not None and images is not None and input_ids.shape[1] == 1:
                attention_length = past_key_values[-1][-1].shape[-2] + 1
                attention_mask = torch.ones((attention_mask.shape[0], attention_length), dtype=attention_mask.dtype, device=attention_mask.device)
            return input_ids, attention_mask, past_key_values, None, labels

        # Process image features and prepare new multimodal input embeddings
        image_features = self._generate_image_embeddings(images)
        new_input_embeds, new_labels = self._create_multimodal_embeddings(input_ids, labels, image_features)
        new_input_embeds, new_labels, attention_mask = self._pad_and_align_sequences(new_input_embeds, new_labels, attention_mask, input_ids, labels)
        return None, attention_mask, past_key_values, new_input_embeds, new_labels


    def _generate_image_embeddings(self, image_data):
        """
        Generate embeddings for image inputs using the vision module.

        Args:
            image_data (torch.Tensor or list): Image inputs.

        Returns:
            torch.Tensor or list: Image embeddings.
        """
        if isinstance(image_data, list) or image_data.ndim == 5:
            merged_images = torch.cat(image_data, dim=0)
            image_embeddings = self.encode_images(merged_images)
            split_sizes = [img.shape[0] for img in image_data]
            image_embeddings = torch.split(image_embeddings, split_sizes, dim=0)
            image_embeddings = [embedding.reshape(-1, embedding.shape[-1]) for embedding in image_embeddings]
        else:
            image_embeddings = self.encode_images(image_data)
        return image_embeddings

    def _create_multimodal_embeddings(self, token_ids,  target_labels, image_embeddings):
        """
        Create multimodal embeddings by combining text and image embeddings.

        Args:
            token_ids (torch.Tensor): Input token IDs.
            target_labels (torch.Tensor or None): Target labels for training.
            image_embeddings (torch.Tensor or list): Image embeddings.

        Returns:
            Tuple[list, list or None]: New input embeddings and updated target_labels.
        """
        embeddings_list = []
        target_labels_list = [] if target_labels is not None else None
        image_counter = 0

        for idx, tokens in enumerate(token_ids):
            embeddings, target, image_counter = self._process_single_sample(tokens, image_embeddings, image_counter, target_labels, idx)
            embeddings_list.append(embeddings)
            if target_labels is not None:
                target_labels_list.append(target)
        return embeddings_list, target_labels_list

    def _process_single_sample(self, tokens, image_embeddings, image_counter, target_labels, idx):
        """
        Process a single sample by inserting image embeddings where necessary.

        Args:
            tokens (torch.Tensor): Tokens for the current sample.
            image_embeddings (torch.Tensor or list): Image embeddings.
            image_counter (int): Current index in the image embeddings list.
            target_labels (torch.Tensor or None): Target labels for training.
            idx (int): Index of the current sample.

        Returns:
            Tuple[torch.Tensor, torch.Tensor or None, int]: Updated embeddings, target_labels, and image counter.
        """
        if (tokens == IMAGE_TOKEN_INDEX).sum() == 0:
            # Handle text-only samples
            return self._embed_text_only_sample(tokens, target_labels, idx), None, image_counter

        sample_embeddings = []
        sample_target_labels = [] if target_labels is not None else None
        current_target_labels = target_labels[idx] if target_labels is not None else None
        image_positions = (tokens == IMAGE_TOKEN_INDEX).nonzero(as_tuple=True)[0]

        while len(image_positions) > 0:
            position = image_positions[0].item()
            tokens, current_target_labels = self._insert_image_embedding(
                tokens, current_target_labels, image_embeddings[image_counter], position, sample_embeddings, sample_target_labels)
            image_counter += 1
            image_positions = (tokens == IMAGE_TOKEN_INDEX).nonzero(as_tuple=True)[0]

        # Append remaining tokens after processing images
        sample_embeddings, sample_target_labels = self._append_remaining_tokens(tokens, current_target_labels, sample_embeddings, sample_target_labels)
        sample_embeddings = torch.cat([emb.to(self.device) for emb in sample_embeddings], dim=0)

        return sample_embeddings, sample_target_labels, image_counter

    def _embed_text_only_sample(self, tokens, target_labels, idx):
        """
        Embed tokens for samples without images.

        Args:
            tokens (torch.Tensor): Tokens for the current sample.
            target_labels (torch.Tensor or None): Target labels for training.
            idx (int): Index of the current sample.

        Returns:
            torch.Tensor: Embeddings for the text-only sample.
        """
        embeddings = self.get_model().embed_tokens(tokens) + (0.0 * self.get_vision_tower().dummy_feature).sum()
        return embeddings, target_labels[idx] if target_labels is not None else None

    def _insert_image_embedding(self, tokens, target_labels, image_embedding, position, sample_embeddings, sample_target_labels):
        """
        Insert image embedding into the sample embeddings at the specified position.

        Args:
            tokens (torch.Tensor): Tokens for the current sample.
            target_labels (torch.Tensor or None): Current target_labels.
            image_embedding (torch.Tensor): Image embedding to insert.
            position (int): Position of the image token in the tokens.
            sample_embeddings (list): List to store embeddings of the sample.
            sample_target_labels (list or None): List to store target_labels of the sample.

        Returns:
            Tuple[torch.Tensor, torch.Tensor or None, int]: Updated tokens, target_labels, and incremented image counter.
        """
        sample_embeddings.append(self.get_model().embed_tokens(tokens[:position]))
        sample_embeddings.append(image_embedding)
        if target_labels is not None:
            sample_target_labels.append(target_labels[:position])
            ignore_mask = torch.full((image_embedding.shape[0],), IGNORE_INDEX, dtype=target_labels.dtype, device=target_labels.device)
            sample_target_labels.append(ignore_mask)
        tokens = tokens[position + 1:]
        if target_labels is not None:
            target_labels = target_labels[position + 1:]
        return tokens, target_labels

    def _append_remaining_tokens(self, tokens, target_labels, sample_embeddings, sample_target_labels):
        """
        Append remaining tokens and target_labels after image insertions.

        Args:
            tokens (torch.Tensor): Remaining tokens.
            target_labels (torch.Tensor or None): Remaining target_labels.
            sample_embeddings (list): List of sample embeddings.
            sample_target_labels (list or None): List of sample target_labels.

        Returns:
            Tuple[list, list or None]: Updated sample embeddings and target_labels.
        """
        if tokens.numel() > 0:
            sample_embeddings.append(self.get_model().embed_tokens(tokens))
            if target_labels is not None:
                sample_target_labels.append(target_labels)
        return sample_embeddings, sample_target_labels

    def _pad_and_align_sequences(self, embeddings_list, target_labels_list, attention_masks, token_ids, labels):
        """
        Pad and align sequences to ensure consistent lengths across the batch.

        Args:
            embeddings_list (list): List of embeddings sequences.
            target_labels_list (list or None): List of target sequences.
            attention_masks (torch.Tensor): Original attention masks.
            token_ids (torch.Tensor): Original token IDs.
            labels (torch.Tensor): Original labels (before any processing).

        Returns:
            Tuple[torch.Tensor, torch.Tensor or None, torch.Tensor]: Padded embeddings, target_labels, and attention masks.
        """
        max_length = max(emb.shape[0] for emb in embeddings_list)
        if any(emb.shape[0] != max_length for emb in embeddings_list):
            if target_labels_list is not None:
                embeddings_list, target_labels_list, unpadded_target_labels = self._pad_sequences(embeddings_list, target_labels_list, max_length)
            else:
                embeddings_list, _, _ = self._pad_sequences(embeddings_list, None, max_length)

            if attention_masks is not None and target_labels_list is not None:
                attention_masks = self._update_attention_masks(attention_masks, unpadded_target_labels, target_labels_list, labels)
        else:
            embeddings_list = torch.stack(embeddings_list, dim=0)
            if target_labels_list is not None:
                target_labels_list = torch.stack(target_labels_list, dim=0)
            if attention_masks is not None:
                padding = torch.full((attention_masks.shape[0], embeddings_list.shape[1] - token_ids.shape[1]),True, dtype=attention_masks.dtype, device=attention_masks.device)
                attention_masks = torch.cat((padding, attention_masks), dim=1)
                assert attention_masks.shape == embeddings_list.shape[:2]
        return embeddings_list, target_labels_list, attention_masks

    def _pad_sequences(self, embeddings_list, target_labels_list, max_length):
        """
        Pad embeddings and target_labels to the maximum sequence length.

        Args:
            embeddings_list (list): List of embeddings sequences.
            target_labels_list (list or None): List of target sequences.
            max_length (int): Maximum sequence length.

        Returns:
            Tuple[torch.Tensor, torch.Tensor or None, List[torch.Tensor]]: Padded embeddings, target_labels, and unpadded target labels.
        """
        padded_embeddings = []
        padded_target_labels = []
        unpadded_target_labels = []
        for emb, tgt in zip(embeddings_list, target_labels_list or []):
            pad_size = max_length - emb.shape[0]
            pad_emb = torch.zeros((pad_size, emb.shape[1]), dtype=emb.dtype, device=emb.device)
            padded_embeddings.append(torch.cat([emb, pad_emb], dim=0))

            if target_labels_list is not None:
                unpadded_target_labels.append(tgt)
                pad_tgt = torch.full((pad_size,), IGNORE_INDEX, dtype=tgt.dtype, device=tgt.device)
                padded_target_labels.append(torch.cat([tgt, pad_tgt], dim=0))
        if target_labels_list is not None:
            return torch.stack(padded_embeddings, dim=0), torch.stack(padded_target_labels, dim=0), unpadded_target_labels
        else:
            return torch.stack(padded_embeddings, dim=0), None, None

    def _update_attention_masks(self, attention_masks, unpadded_target_labels, padded_target_labels, labels):
        """
        Update attention masks to align with the padded embeddings.

        Args:
            attention_masks (torch.Tensor): Original attention masks.
            unpadded_target_labels (list of torch.Tensor): List of unpadded target labels.
            padded_target_labels (torch.Tensor): Padded target labels.
            labels (torch.Tensor): Original labels.

        Returns:
            torch.Tensor: Updated attention masks.
        """
        new_attention_masks = []
        for mask, unpadded_tgt, padded_tgt in zip(attention_masks, unpadded_target_labels, padded_target_labels):
            left_pad_len = unpadded_tgt.shape[0] - labels.shape[1]
            right_pad_len = padded_tgt.shape[0] - unpadded_tgt.shape[0]
            new_attn_mask_pad_left = torch.full((left_pad_len,), True, dtype=mask.dtype, device=mask.device)
            new_attn_mask_pad_right = torch.full((right_pad_len,), False, dtype=mask.dtype, device=mask.device)
            new_mask = torch.cat((new_attn_mask_pad_left, mask, new_attn_mask_pad_right), dim=0)
            new_attention_masks.append(new_mask)
        attention_masks = torch.stack(new_attention_masks, dim=0)
        assert attention_masks.shape == padded_target_labels.shape
        return attention_masks

    def initialize_vision_tokenizer(self, model_args, tokenizer):
        """
        Initialize the vision tokenizer by adding necessary special tokens and configuring model components.

        Args:
            model_args (Namespace): Model arguments containing configuration options.
            tokenizer (Tokenizer): The tokenizer to be updated with visual patch tokens.

        Returns:
            None
        """
        if model_args.mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

        if model_args.mm_use_im_patch_token and model_args.tune_mm_mlp_adapter:
            for param in self.get_input_embeddings().parameters():
                param.requires_grad = False
            for param in self.get_output_embeddings().parameters():
                param.requires_grad = False

