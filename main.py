
import torch
import os
from datetime import datetime
from fastcore.utils import * # type: ignore # noqa: F403
from fedai.federated.agents import * # type: ignore # noqa: F403
from fedai.learner_utils import * # type: ignore # noqa: F403
from fedai.client_selector import *  # type: ignore # noqa: F403
from fedai.wandb_writer import *  # type: ignore # noqa: F403
from fedai.FLearner import * # type: ignore # noqa: F403
from torch import nn # type: ignore # noqa: F403
from omegaconf import OmegaConf # type: ignore # noqa: F403
import argparse
import yaml
from huggingface_hub import login # type: ignore # noqa: F403
from dotenv import load_dotenv # type: ignore # noqa: F403


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run Federated Learning Simulation')
    parser.add_argument('--config', type=str, help='Path to the YAML config file', required=True)
    parser.add_argument('--env_file', type=str, help='Path to the .env file', required=False)
    args = parser.parse_args()

    if args.env_file:
        load_dotenv(args.env_file)
        key = os.getenv("WANDB_API_KEY", None)
        hf_secret = os.getenv("HF_SECRET_CODE", None)

        if key:
            os.environ["WANDB_API_KEY"] = key
        if hf_secret:
            os.environ["HF_SECRET_CODE"] = hf_secret     

    try:
        with open(args.config, 'r') as file:
            cfg = yaml.safe_load(file)
            cfg = OmegaConf.create(cfg)
    except:
        print("Invalid config file path")
    
    client_selector = get_cls("fedai.client_selector", cfg.client_selector)
    client_cls = get_cls("fedai.federated.agents", cfg.client_cls)
    loss_fn = get_cls("torch.nn", cfg.loss_fn)
    writer = get_cls("fedai.wandb_writer", cfg.writer)

    learner = FLearner(cfg, client_fn, client_selector, client_cls, loss_fn, writer) # type: ignore
    learner.run_simulation()
