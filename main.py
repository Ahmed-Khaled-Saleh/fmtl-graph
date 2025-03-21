
import torch # type: ignore
import os # type: ignore
import gdown # type: ignore
from datetime import datetime # type: ignore
from fastcore.utils import * # type: ignore
from fedai.federated.agents import * # type: ignore
from fedai.learner_utils import *  # type: ignore
from fedai.client_selector import *  # type: ignore
from fedai.wandb_writer import *  # type: ignore
from fedai.FLearner import *  # type: ignore
from fedai.vision.models import CIFAR10Model # type: ignore
from fedai.vision.VisionBlock import Cifar10_20clients # type: ignore
from torch import nn # type: ignore
from omegaconf import OmegaConf # type: ignore
import argparse
import yaml
from huggingface_hub import login # type: ignore
from dotenv import load_dotenv # type: ignore
from peft import *  # type: ignore # noqa: F403

def load_state_from_disk(cfg, state, latest_round, id, t, to_read_from):
    
    if cfg.agg == "one_model":
        global_model_path = os.path.join(cfg.save_dir,
                                        str(t-1),
                                        "global_model",
                                        "state.pth")

        gloabal_model_state = torch.load(global_model_path, weights_only= False)
        if isinstance(state["model"], torch.nn.Module):
            state["model"].load_state_dict(gloabal_model_state["model"])
            print(f"Loaded Global model state from {global_model_path}")
        else:
            set_peft_model_state_dict(state["model"],  # noqa: F405 # type: ignore
                                      gloabal_model_state["model"],
                                      "default")

    else:
        if id not in latest_round:
            return state

        latest_comm_round = latest_round[id]
        old_state_path = os.path.join(cfg.save_dir,
                                       str(latest_comm_round),
                                       f"{to_read_from}{id}",
                                       "state.pth")

        old_saved_state = torch.load(old_state_path, weights_only= False)

        if isinstance(state["model"], nn.Module) or isinstance(state["model"], dict) :
            state["model"].load_state_dict(old_saved_state["model"])

            if to_read_from == "aggregated_model_":
                state["h_c"] = old_saved_state["h_c"]
                
            print(f"Loaded client model state from {old_state_path}")
        else:
            set_peft_model_state_dict(state["model"],  # noqa: F405 # type: ignore
                                      old_saved_state["model"],
                                      "default")

    return state

def client_fn(client_cls, cfg, id, latest_round, t, loss_fn = None, optimizer = None, to_read_from= None, extra= False):

        model = CIFAR10Model()
        criterion = nn.CrossEntropyLoss()

        train_block = Cifar10_20clients(cfg, id)
        test_block = Cifar10_20clients(cfg, id, train=False)

        state = {'model': model, 'optimizer': None, 'criterion': criterion, 't': t, 'h': None, 'h_c': None}

        if t > 1 or extra:
            state = load_state_from_disk(cfg, state, latest_round, id, t, to_read_from)  # noqa: F405

        state['optimizer'] = get_cls("torch.optim", cfg.optimizer.name)(state['model'].parameters(), lr=cfg.lr)
                                                                        
        state['alignment_criterion']= get_cls("torch.nn", cfg.alignment_criterion)
        return client_cls(id, cfg, state, block= [train_block, test_block])



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run Federated Learning Simulation')
    parser.add_argument('--config', type=str, help='Path to the YAML config file', required=True)
    parser.add_argument('--env_file', type=str, help='Path to the .env file', required=False)

    parser.add_argument('--lr', type=float, help='Learning rate local', required=False)
    parser.add_argument('--lr2', type=float, help='Learning rate alginment', required=False)
    parser.add_argument('--batch_size', type=int, help='Batch size', required=False)
    parser.add_argument('--optimizer', type=str, help='Optimizer', required=False)
    parser.add_argument('--optimizer2', type=str, help='Optimizer2', required=False)
    parser.add_argument('--alignment_criterion', type=str, help='Alignment criterion', required=False)
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
    
    cfg.lr = args.lr if args.lr else cfg.lr
    cfg.lr2 = args.lr2 if args.lr2 else cfg.lr2

    cfg.data.batch_size = args.batch_size if args.batch_size else cfg.data.batch_size
    cfg.optimizer.name = args.optimizer if args.optimizer else cfg.optimizer.name
    cfg.optimizer2 = args.optimizer2 if args.optimizer2 else cfg.optimizer2

    cfg.alignment_criterion = args.alignment_criterion if args.alignment_criterion else cfg.alignment_criterion
    cfg.client_cls = args.client_cls if args.client_cls else cfg.client_cls

    cfg.agg = args.agg if args.agg else cfg.agg
    cfg.lambda_ = args.lambda_ if args.lambda_ else cfg.lambda_

   
    client_selector = get_cls("fedai.client_selector", cfg.client_selector)
    client_cls = get_cls("fedai.federated.agents", cfg.client_cls)
    loss_fn = get_cls("torch.nn", cfg.loss_fn)
    writer = get_cls("fedai.wandb_writer", cfg.writer)

    learner = FLearner(cfg, client_fn, client_selector, client_cls, loss_fn, writer) # type: ignore
    learner.run_simulation()
