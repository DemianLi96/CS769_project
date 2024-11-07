import torch
import torch.nn as nn

from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig


class CLIPVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()
        self.is_loaded = False
        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch') # default to "patch"

        if not delay_load:
            # load config and model
            self.load_model()
        else:
            # only load config
            self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_tower_name)

    def load_model(self):
        """Load the pre-trained model and processor"""
        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
        self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name)
        # freeze the vision tower, only use the pre-trained weights
        self.vision_tower.requires_grad_(False)
        self.is_loaded = True

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        if self.select_feature == 'patch':
            # remove the cls token
            image_features = image_features[:, 1:]
        elif self.select_feature == 'cls_patch':
            # keep the cls token
            image_features = image_features
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return image_features

    @torch.no_grad()
    def forward(self, images):
        """Extract image features from the vision tower"""
        if isinstance(images, list):
            image_features = []
            for image in images:
                image = image.to(device=self.device, dtype=self.dtype).unsqueeze(0)
                image_forward_out = self.vision_tower(image, output_hidden_states=True)
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_features.append(image_feature)
        else:
            image = images.to(device=self.device, dtype=self.dtype)
            image_forward_outs = self.vision_tower(image, output_hidden_states=True)
            image_features = self.feature_select(image_forward_outs).to(images.dtype)

        return image_features

    @property
    def dummy_feature(self):
        """Dummy feature for the vision tower, could be used for initialization"""
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        """Data type of the vision tower"""
        return self.vision_tower.dtype

    @property
    def device(self):
        """Device of the vision tower, CPU or GPU"""
        return self.vision_tower.device

    @property
    def config(self):
        """Config of the vision tower"""
        return self.vision.tower.config if self.is_loaded else self.cfg_only

    @property
    def hidden_size(self):
        """Hidden size of the vision tower"""
        return self.config.hidden_size

    @property
    def num_patches(self):
        """Number of patches in the image"""
        img_size = self.config.image_size
        patch_size = self.config.patch_size
        return (img_size // patch_size) ** 2
