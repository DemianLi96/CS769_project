from .clip_encoder import CLIPVisionTower

def build_vision_tower(vision_tower_cfg, **kwargs):
    # Try to retrieve the vision tower name from the config, prioritizing 'mm_vision_tower' over 'vision_tower'
    vision_tower_name = getattr(vision_tower_cfg, 'mm_vision_tower', None) or getattr(vision_tower_cfg, 'vision_tower', None)
    if vision_tower_name is None:
        raise ValueError("Configuration missing required 'vision_tower' or 'mm_vision_tower' attribute")

    # Verify if the vision tower name has a valid prefix, either 'openai' or 'laion'
    if vision_tower_name.startswith(("openai", "laion")):
        return CLIPVisionTower(vision_tower_name, args=vision_tower_cfg, **kwargs)

    raise ValueError(f"Unknown vision tower: {vision_tower_name}")