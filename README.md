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

## Setup and Installation(TO BE UPDATED)

1. **Prerequisites**
   - Python 3.8+
   - PyTorch
   - Transformers library
   - CUDA-capable GPU (recommended)

2. **Installation**
   ```bash
   git clone [repository-url]
   cd [repository-name]
   pip install -r requirements.txt
   ```

3. **Model Download**
   - Download the pre-trained LLaVA model
   - Download the Food-101 dataset
   - Set up the required environment variables

## Usage(TO BE UPDATED)

1. **Basic Image Recognition**
   ```python
   from llava_llama import LlavaLlamaForCausalLM
   from clip_encoder import CLIPVisionTower

   # Initialize the model
   model = LlavaLlamaForCausalLM.from_pretrained("path/to/model")
   
   # Process an image
   results = model.generate(image, prompt="What food is in this image?")
   ```

2. **Nutritional Analysis**
   ```python
   # Get nutritional information
   nutrition = model.generate(
       image, 
       prompt="What are the nutritional components of this dish?"
   )
   ```

## Training(TO BE UPDATED)

To fine-tune the model on your own dataset:

1. **Data Preparation**
   - Organize your food images
   - Generate instruction pairs using GPT-4
   - Prepare nutritional information

2. **Fine-tuning**
   ```bash
   python train.py \
       --data_path /path/to/data \
       --output_dir /path/to/save \
       --epochs 3 \
       --batch_size 8
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
