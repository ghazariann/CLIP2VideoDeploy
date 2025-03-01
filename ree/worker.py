# worker.py

# -*- coding: utf-8 -*-
import sys 
import os

MAIN_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(MAIN_DIR)

import json
import torch
import torch.distributed as dist
import time 
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import io
import torch
import numpy as np
import random
from modules.tokenization_clip import SimpleTokenizer as ClipTokenizer
from modules.modeling import CLIP2Video
# from evaluation.eval import eval_epoch, my_eval_epoch_image
from torch.profiler import profile, record_function, ProfilerActivity

# from utils.config import get_args
# from utils.utils import get_logger
# from utils.dataloader import dataloader_msrvtt_test_video, dataloader_msrvtt_test_image

import csv
from PIL import Image
import csv
from tqdm import tqdm
import pickle
import argparse

def remove_module_prefix(state_dict):
    """Removes the 'module.' prefix from state_dict keys if present."""
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace("module.", "")
        new_state_dict[new_key] = value
    return new_state_dict

# -------------------------------
# Torch Distributed Initialization
# -------------------------------

def init_distributed():
    MASTER_ADDR = '18.116.182.172'  # Master server IP
    MASTER_PORT = '29500'           # Must match master's MASTER_PORT
    WORLD_SIZE = 2                  # Total number of processes (master + worker)
    RANK = 1                        # Worker rank

    os.environ['MASTER_ADDR'] = MASTER_ADDR
    os.environ['MASTER_PORT'] = MASTER_PORT
    init_method = f'tcp://{MASTER_ADDR}:{MASTER_PORT}'

    dist.init_process_group(
        backend='gloo',        # Use 'gloo' for CPU
        init_method=init_method,
        world_size=WORLD_SIZE,
        rank=RANK
    )
    print("Distributed process group initialized on worker.")

# -------------------------------
# Model Loading
# -------------------------------
class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)

class MyConfig:
    def __init__(self):
        self.do_eval = False
        self.val_csv = 'data/.val.csv'
        self.video_val_csv = './data/video_ids.csv'
        self.text_val_csv = './data/sentences.csv'
        self.data_path = './data/MSRVTT_data.json'
        self.features_path = 'no need, we have embeddings'
        self.num_thread_reader = 2
        self.batch_size_val = 64
        self.seed = 42
        self.max_words = 32
        self.max_frames = 12
        self.feature_framerate = 2
        self.cross_model = "cross-base"
        self.do_lower_case = False  # set to True if needed
        self.n_gpu = 1
        self.local_rank = 0
        self.cache_dir = ""
        self.fp16 = False  # set to True if needed
        self.fp16_opt_level = 'O1'
        self.cross_num_hidden_layers = 4
        self.sim_type = "seqTransf"
        self.checkpoint = './pretrained_weights'
        self.model_num = '2'
        self.local_rank = 0
        self.datatype = "msrvtt"
        self.vocab_size = 49408
        self.temporal_type = ''
        self.temporal_proj = 'sigmoid_selfA'
        self.center_type = 'TAB'
        self.centerK = 5
        self.center_weight = 0.5
        self.center_proj = 'TAB_TDB'
        self.clip_path = './pretrained_weights/ViT-B-32.pt'
        self.TR = False
        self.TC = False
        self.VR = False
        self.VC = False
        self.topk = 3  
     


def init_model(args, device):
    """Initialize model.

    if location of args.init_model exists, model will be initialized from the pretrained model.
    if no model exists, the training will be initialized from CLIP's parameters.

    Args:
        args: the hyper-parameters
        devices: cuda

    Returns:
        model: the initialized model

    """

    # resume model if pre-trained model exist.
    model_file = os.path.join(args.checkpoint, "pytorch_model.bin.{}".format(args.model_num))
    if os.path.exists(model_file):
        model_state_dict = torch.load(model_file, map_location='cpu')
        # if args.local_rank == 0:
            # logger.info("Model loaded from %s", model_file)
    else:
        model_state_dict = None

    # Prepare model
    model = CLIP2Video.from_pretrained(args.cross_model, cache_dir=None, state_dict=model_state_dict,
                                       task_config=args)
    model.to(device)

    return model

# -------------------------------
# Main Worker Function
# -------------------------------
def init_device(args, local_rank):
    """Initialize device to determine CPU or GPU

     Args:
         args: the hyper-parameters
         local_rank: GPU id

     Returns:
         devices: cuda
         n_gpu: number of gpu

     """
    global logger
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu", local_rank)
    n_gpu = 1
    # logger.info("device: {} n_gpu: {}".format(device, n_gpu))
    args.n_gpu = n_gpu

    if args.batch_size_val % args.n_gpu != 0:
        raise ValueError("Invalid batch_size/batch_size_val and n_gpu parameter: {}%{} and {}%{}, should be == 0".format(
            args.batch_size, args.n_gpu, args.batch_size_val, args.n_gpu))

    return device, n_gpu

def main():
    """Main loop for the worker to receive tensors, process them, and send back the results."""
    init_distributed()
    args = MyConfig()
    # set_seed_logger(args)
    device, n_gpu = init_device(args, args.local_rank)
    model = init_model(args, device)
    if hasattr(model, 'module'):
        model = model.module.to(device)
    else:
        model = model.to(device)

    # model.eval()

    print("Worker is ready to receive tensors.")

    try:
        while True:
            # Define fixed shapes
            batch_size = 1
            seq_length = 50
            hidden_dim = 768

            # Initialize tensors with fixed shapes
            embedding_output = torch.zeros(batch_size, seq_length, hidden_dim, dtype=torch.float)
            
            # Receive embedding_output and attn_mask from master
            dist.recv(tensor=embedding_output, src=0)
            print(f"Received embedding_output: shape={embedding_output.shape}, dtype={embedding_output.dtype}")
            embedding_output = embedding_output.to(torch.float16)
            # Perform Backbone_forward
            try:
                sequence_output = model.clip.visual.Backbone_forward(embedding_output)                
                print(f"Performed Backbone_forward: shape={sequence_output.shape}, dtype={sequence_output.dtype}")
            except Exception as e:
                print(f"Error during Backbone_forward on worker: {e}")
                # Optionally, send a tensor filled with zeros or a termination signal
                sequence_output = torch.zeros(batch_size, seq_length, hidden_dim)  # Adjust hidden_dim as necessary

            # Send sequence_output back to master
            dist.send(tensor=sequence_output, dst=0)
            print("Sent sequence_output back to master.")

    except KeyboardInterrupt:
        print("Worker interrupted by user. Shutting down.")

    except Exception as e:
        print(f"Worker encountered an error: {e}")

    finally:
        dist.destroy_process_group()
        print("Worker process group destroyed.")

if __name__ == "__main__":
    main()
