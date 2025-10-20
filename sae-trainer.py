import torch
from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer

from sparsify import SaeConfig, Trainer, TrainConfig
from sparsify.data import chunk_and_tokenize
from tqdm import tqdm


if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using Apple Silicon MPS device.")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using NVIDIA CUDA device.")
else:
    device = torch.device("cpu")
    print("Using CPU device.")

def main():
    dataset = load_dataset(
        "EleutherAI/SmolLM2-135M-10B", 
        split="train",
    )
    model_names = [
        "Qwen/Qwen3-Embedding-0.6B",
        # "intfloat/multilingual-e5-large-instruct",
        # "ibm-granite/granite-4.0-micro-base",
    ]
    model_combinations = [(model, transcoder) for model in model_names for transcoder in [False, True]]

    for model_name, transcoder in (pbar := tqdm(model_combinations)):
        pbar.set_description(f"{model_name}{" - Transcoder" if transcoder else ""}")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            device=device,
            device_map={"": str(device)},
        )
        tokenized = chunk_and_tokenize(dataset, tokenizer)

        model = AutoModel.from_pretrained(
            model_name, 
            # device=device,
            device_map={"": str(device)},
        )

        #print(list(model.base_model.named_modules())[0])
        #print("=====================================================")

        sae_config = SaeConfig()
        # sae_config.expansion_factor = 8 if "granite" in model_name else 16 # Memory constraints
        sae_config.expansion_factor = 16
        sae_config.transcode = transcoder

        train_config = TrainConfig(sae_config)
        train_config.log_to_wandb = False
        train_config.run_name = model_name.replace('/', '-') + ("-transcoder" if transcoder else "")
        train_config.hookpoints = ["layers.*.mlp"] # Qwen only

        trainer = Trainer(train_config, tokenized, model)

        trainer.fit()

        del tokenizer
        del tokenized
        del model
        del trainer


if __name__ == "__main__":
    main()
