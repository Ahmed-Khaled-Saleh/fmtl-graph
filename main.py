
import torch # type: ignore
import os # type: ignore
import gdown # type: ignore
from datetime import datetime # type: ignore
from copy import deepcopy
import argparse
import yaml
from torch import nn
from fastcore.utils import * # type: ignore
from fedai.federated.agents import * # type: ignore
from fedai.learner_utils import *  # type: ignore
from fedai.client_selector import *  # type: ignore
from fedai.wandb_writer import *  # type: ignore
from fedai.FLearner import *  # type: ignore
from fedai.vision.models import * # type: ignore
from fedai.vision.VisionBlock import *
from torch import nn # type: ignore
from omegaconf import OmegaConf # type: ignore
from huggingface_hub import login # type: ignore
from dotenv import load_dotenv # type: ignore
from peft import *  # type: ignore # noqa: F403


def client_fn(client_cls, cfg, id, latest_round, t, loss_fn = None, optimizer = None, state_dir= None):
    
    model = get_model(cfg)
    criterion = get_criterion(loss_fn)
    train_block, test_block = get_block(cfg, id), get_block(cfg, id, train=False)

    state = {'model': model, 'optimizer': None, 'criterion': criterion, 't': t, 'h': None, 'h_c': None, "pers_model": None}

    
    if t == 1 and cfg.client_cls == "pFedMe" and cfg.agg  != "one_model":
        state = load_state_from_disk(cfg, state, latest_round, id, t, state_dir)  

    if t == 1:
        state['w0'] = deepcopy(state['model'])
        
    if t > 1:
        state = load_state_from_disk(cfg, state, latest_round, id, t, state_dir)  
        

    state['optimizer'] = get_cls("torch.optim", cfg.optimizer.name)(state['model'].parameters(), lr=cfg.optimizer.lr)      
    state['alignment_criterion']= get_cls("torch.nn", cfg.alignment_criterion)
    
    return client_cls(id, cfg, state, block= [train_block, test_block])





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run Federated Learning Simulation')
    parser.add_argument('--config', type=str, help='Path to the YAML config file', required=True)
    parser.add_argument('--timestamp', type=str, help='Time stamp', required=True)
    parser.add_argument('--env_file', type=str, help='Path to the .env file', required=False)
    

    parser.add_argument('--lr', type=float, help='Learning rate local', required=False)
    parser.add_argument('--batch_size', type=int, help='Batch size', required=False)
    parser.add_argument('--optimizer', type=str, help='Optimizer', required=False)
    parser.add_argument('--client_cls', type=str, help='Client class', required=False)
    parser.add_argument('--agg', type=str, help='Aggregation', required=False)
    parser.add_argument('--lambda_', type=str, help='lambda for fedu and dmtl', required=False)
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
            cfg = OmegaConf.load(file)
    except:
        print("Invalid config file path")

    cfg.now = args.timestamp if args.timestamp else datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    cfg.optimizer.lr = args.lr if args.lr else cfg.optimizer.lr

    cfg.data.batch_size = args.batch_size if args.batch_size else cfg.data.batch_size
    cfg.optimizer.name = args.optimizer if args.optimizer else cfg.optimizer.name

    cfg.client_cls = args.client_cls if args.client_cls else cfg.client_cls

    cfg.agg = args.agg if args.agg else cfg.agg
    cfg.lambda_ = args.lambda_ if args.lambda_ else cfg.lambda_

   
    client_selector = get_cls("fedai.client_selector", cfg.client_selector)
    client_cls = get_cls("fedai.federated.agents", cfg.client_cls)
    loss_fn = get_cls("torch.nn", cfg.loss_fn)
    writer = get_cls("fedai.wandb_writer", cfg.writer)

    if cfg.client_cls == "FLAgent":
        cfg.agg = "one_model"
    else:
        cfg.agg = "mtl"

    learner = FLearner(cfg, client_fn, client_selector, client_cls, loss_fn, writer) # type: ignore
    learner.run_simulation()
