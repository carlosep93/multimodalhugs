### Usage: python training_setup.py --config_path path_to_your_config.yaml

import os
import copy
import torch
import argparse
from omegaconf import OmegaConf
from pathlib import Path

from multimodalhugs.data import BilingualImage2textMTDataConfig, BilingualImage2TextDataset, add_new_special_tokens_from_vocab_file
from multimodalhugs.processors import Image2TextTranslationProcessor
from multimodalhugs.models import MultiModalEmbedderModel

from transformers import AutoTokenizer

def main(config_path):
    # Load config and initialize dataset
    config = OmegaConf.load(config_path)
    dataset_config = BilingualImage2textMTDataConfig(config)
    dataset = BilingualImage2TextDataset(config=dataset_config)

    # Download, prepare, and save dataset
    data_path = Path(config.training.output_dir) / config.model.name / "datasets" / dataset.name
    dataset.download_and_prepare(data_path)
    dataset.as_dataset().save_to_disk(data_path)

    m2m_tokenizer = AutoTokenizer.from_pretrained(dataset_config.text_tokenizer_path)

    tokenizer = m2m_tokenizer

    vocab_files = []
    new_vocab_tokens = []
    if dataset_config.tokenizer_src_langs_path:
        vocab_files.append(dataset_config.tokenizer_src_langs_path)
    if dataset_config.new_task_tokens_dictionary_path:
        vocab_files.append(dataset_config.new_task_tokens_dictionary_path)

    for i, vocab_file in enumerate(vocab_files):
        output_dir = None
        if i == len(vocab_files) - 1:
            output_dir = f"{config.training.output_dir}/{config.model.name}"
        tokenizer, new_special_tokens = add_new_special_tokens_from_vocab_file(
            tokenizer=copy.deepcopy(tokenizer),
            vocab_file=vocab_file,
            output_dir=output_dir,
        )
        new_vocab_tokens = new_vocab_tokens + new_special_tokens
    

    input_processor = Image2TextTranslationProcessor(
        tokenizer=tokenizer,
        font_path=dataset_config.font_path,
        width=dataset_config.preprocess.width,
        height=dataset_config.preprocess.height,
        normalize_image=dataset_config.preprocess.do_normalize,
        mean=dataset_config.preprocess.dataset_mean,
        std=dataset_config.preprocess.dataset_std,
    )

    # Save processor and set PROCESSOR_PATH environment variable
    processor_path = config.training.output_dir + f"/{config.model.name}" + f"/image2text_translation_processor"
    input_processor.save_pretrained(save_directory=processor_path, push_to_hub=False)

    # Build and save the model, then set MODEL_PATH environment variable
    model = MultiModalEmbedderModel.build_model(
        cfg=config.model, 
        src_tokenizer=tokenizer, 
        tgt_tokenizer=m2m_tokenizer,
        config_path=config_path,
        new_vocab_tokens=new_vocab_tokens,
    )

    model_path = f"{config.training.output_dir}/{config.model.name}/trained_model"
    model.save_pretrained(model_path)

    print(f"MODEL_PATH={model_path}")
    print(f"PROCESSOR_PATH={processor_path}")
    print(f"DATA_PATH={data_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training script for multimodal models")
    parser.add_argument('--config_path', type=str, required=True, help="Path to the configuration file")
    
    args = parser.parse_args()
    
    main(args.config_path)