# CS769 Project : Enhancing Multimodal Models for Dietary Assessment

An implementation of LLaVA (Large Language and Vision Assistant) specifically fine-tuned for food recognition and nutritional analysis.

This repository is dedicated to fine-tuning and evaluating the LLaVA model to build a system that recognizes food items from images and gives detailed nutritional information.

## Overview

This project extends LLaVA's capabilities to create a specialized system for dietary assessment. By fine-tuning the model on the Food-101 dataset and incorporating nutritional information, we've developed a system that can:
- Recognize food items from images
- Provide detailed nutritional information
- Engage in multi-turn conversations about food and nutrition
- Offer contextual dietary insights

## Files and Folders

- **`LLaVA_ori/`**: This folder contains the original source code of the LLaVA model, used as a baseline reference.

- **`LLaVA_reimplement/`**: This folder contains the reimplementation code of the LLava model.

- **`data/`**: This folder contains the data used for reimplementing the LLaVA model.

- **`llava_finetune_using_custom_dataset.ipynb`**: This file uses a custom dataset, combining natural language processing and vision tasks for multi-modal learning.

## Architecture

The project consists of three main components:

1. **CLIP Vision Tower**: Handles image processing and feature extraction
   - Uses pre-trained CLIP vision model
   - Supports both patch-based and CLS token features
   - Implements efficient batch processing for multiple images

2. **Vision Tower Builder**: Manages model initialization and configuration
   - Supports both 'openai' and 'laion' vision tower variants
   - Handles configuration validation and model setup

3. **LLaVA-Llama Integration**: Combines vision and language capabilities
   - Extends Llama model with multimodal features
   - Implements efficient generation with cached key-values
   - Provides seamless integration with Hugging Face's transformers library

## Performance Results(Double Check)

Our implementation achieves competitive results compared to the original LLaVA:

| Metric Type | Original LLaVA | Our Implementation |
|-------------|---------------|-------------------|
| Classification | 53.2% | 61.8% |
| Nutritional Information | 69.5% | 78.4% |
| Conversation | 83.1% | 84.7% |
| Detailed Description | 75.3% | 70.2% |
| Complex Reasoning | 96.5% | 87.0% |

## Setup and Installation

### Prerequisites

- **Docker** (recommended)
- **Python 3.8+**
- **PyTorch** (2.3.1+ with CUDA 12.1 support is recommended)
- **Transformers library**
- A **CUDA-capable GPU** is strongly recommended for training and inference.

### Using a Docker Image

You can use the official PyTorch Docker image as the base environment. For example:

```bash
docker pull pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel
docker run -it --gpus all --name llava_container pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel /bin/bash

```

After starting the container, clone this repository and install the required dependencies:

```bash
git clone [repository-url]
cd [repository-name]/LLaVA_reimplement
pip install -e .

```

This will install the project in editable mode, allowing you to modify the code and immediately see the changes.

### Building from a Dockerfile

Alternatively, you can build your own Docker image using the provided Dockerfile:

```bash
docker build -t llava_diet_assessment:latest -f Dockerfile .
docker run -it --gpus all --name llava_container llava_diet_assessment:latest /bin/bash

```

### Required Dataset: COCO2014

You need the COCO2014 dataset for training and evaluation:

1. Download the COCO2014 images and annotations from the [COCO dataset website](https://cocodataset.org/).
2. Extract the dataset into a directory, for example:
    
    ```bash
    mkdir -p /data/coco2014
    tar -xvf train2014.zip -C /data/coco2014/
    tar -xvf val2014.zip -C /data/coco2014/
    tar -xvf annotations_trainval2014.zip -C /data/coco2014/
    
    ```
    
3. Set environment variables or provide paths in configuration files to point the model to the COCO2014 dataset directory.

3. **Model Download**
   - Download the pre-trained LLaVA model
   - Download the Food-101 dataset
   - Set up the required environment variables

## Usage

```bash
python model_vqa.py \
    --model-path /workspace/llava-llama-2-7b-chat-finetune_reasoning_20k/ \
    --question-file ./qa90_questions.jsonl \
    --image-folder /workspace/val2014 \
    --answers-file ./answer-file-our.jsonl
```


## Training

To fine-tune the model on your own dataset:

1. **Data Preparation**
   - Organize your food images
   - Generate instruction pairs using GPT-4
   - Prepare nutritional information

2. **Fine-tuning**
   ```bash
   PROMPT_VERSION="llava_llama_2"
   MODEL_VERSION="llama-2-7b-chat"
   deepspeed train.py \
       --deepspeed /root/transfer/zero3.json \
       --model_name_or_path meta-llama/Llama-2-7b-chat-hf \
       --version $PROMPT_VERSION \
       --data_path /your_data.json \
       --image_folder /your_image_folder \
       --vision_tower openai/clip-vit-large-patch14 \
       --pretrain_mm_mlp_adapter /root/pretrain-llama-2-7b-chat/mm_projector.bin \
       --mm_vision_select_layer -2 \
       --mm_use_im_start_end False \
       --mm_use_im_patch_token False \
       --bf16 True \
       --output_dir ./checkpoints/llava-llama-2-7b-chat-finetune \
       --num_train_epochs 1 \
       --per_device_train_batch_size 8 \
       --per_device_eval_batch_size 4 \
       --gradient_accumulation_steps 16 \
       --evaluation_strategy "no" \
       --save_strategy "steps" \
       --save_steps 50000 \
       --save_total_limit 1 \
       --learning_rate 2e-5 \
       --weight_decay 0. \
       --warmup_ratio 0.03 \
       --lr_scheduler_type "cosine" \
       --logging_steps 1 \
       --tf32 True \
       --model_max_length 2048 \
       --gradient_checkpointing True \
       --dataloader_num_workers 4 \
       --lazy_preprocess True \
       --report_to wandb
   ```

## Evaluation

The model can be evaluated using:
- Food recognition accuracy
- Nutritional information accuracy
- Conversation quality
- Complex reasoning capabilities

## Future Work

- Implement fine-grained image detail representation
- Expand the dataset with more food categories 
- Improve model generalization
- Enhance nutritional information accuracy

## Contributors

- Dongyang Li (dli389@wisc.edu) - Team Lead, Model Fine-tuning
- Junkai Wang (jwang2774@wisc.edu) - Dataset Processing, Caption Generation
- Kessys Oliveira (peraltadeoli@wisc.edu) - Literature Review, Evaluation
