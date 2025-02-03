### Usage: python training_setup.py --config_path path_to_your_config.yaml

import os
import copy
import torch
import argparse
from omegaconf import OmegaConf
from pathlib import Path

from multimodalhugs.data import Pose2TextDataset, Pose2TextDataConfig
from multimodalhugs.processors import Pose2TextTranslationProcessor
from multimodalhugs.models.registry import get_model_class
from multimodalhugs.utils.tokenizer_utils import extend_tokenizer

from transformers import AutoTokenizer

def main(config_path):
    # Load config and initialize dataset
    config = OmegaConf.load(config_path)
    dataset_config = Pose2TextDataConfig(config)
    dataset = Pose2TextDataset(config=dataset_config)

    # Download, prepare, and save dataset
    data_path = Path(config.training.output_dir) / config.model.name / "datasets" / dataset.name
    dataset.download_and_prepare(data_path)
    dataset.as_dataset().save_to_disk(data_path)

    # Load the tokenizer (here, we use AutoTokenizer)
    pretrained_tokenizer = AutoTokenizer.from_pretrained(dataset_config.text_tokenizer_path)
    tokenizer, new_vocab_tokens = extend_tokenizer(
        dataset_config, 
        training_output_dir=config.training.output_dir, 
        model_name=config.model.name
    )

    # The preprocessor is created
    input_processor = Pose2TextTranslationProcessor(
            tokenizer=tokenizer,
            reduce_holistic_poses=dataset_config.reduce_holistic_poses,
    )

    # Save processor and set PROCESSOR_PATH environment variable
    processor_path = config.training.output_dir + f"/{config.model.name}" + f"/pose2text_translation_processor"
    input_processor.save_pretrained(save_directory=processor_path, push_to_hub=False)

    # --- Model creation becomes model-independent ---
    # Use the "type" field in the configuration (defaulting if not provided)
    model_type = config.model.get("type", None)
    if model_type is None:
        raise ValueError("model_type not found. Please specify a valid model type on the config.")
    model_class = get_model_class(model_type)

    # Convert the model section of the config to a dictionary.
    # This will include any extra parameters specific to the model.
    model_kwargs = OmegaConf.to_container(config.model, resolve=True)
    
    # Update with common arguments required by build_model().
    model_kwargs.update({
        "src_tokenizer": tokenizer,
        "tgt_tokenizer": pretrained_tokenizer,
        "config_path": config_path,
        "new_vocab_tokens": new_vocab_tokens,
    })

    # Build the model. Each model class can decide which arguments to use.
    model = model_class.build_model(**model_kwargs)

    model_path = os.path.join(config.training.output_dir, config.model.name, "trained_model")
    model.save_pretrained(model_path)

    print(f"MODEL_PATH={model_path}")
    print(f"PROCESSOR_PATH={processor_path}")
    print(f"DATA_PATH={data_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training script for multimodal models")
    parser.add_argument('--config_path', type=str, required=True, help="Path to the configuration file")
    
    args = parser.parse_args()
    
    main(args.config_path)