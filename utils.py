""" Utility functions
"""

import argparse
import yaml
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper
    
import os
import sys
import torch
import numpy as np
    
def write_config_to_file(path, param: dict, mode="w"):
    """
    Writes a dictionary to yaml file.
    
    Args:
    path
    param
    mode    
    """
    #path = str(path + ".yml")
    with open(path, mode) as f:
        yaml.dump(param, f, Dumper)

def read_config_from_file(path):
    if os.path.exists(path):
        with open(path, 'r') as f:
            param = yaml.load(f, Loader)
    else:
        print(f"File not found: {path}")
        param = None

    return param

def parse_args(argv: list):
    """
    Parse command line arguments.
    
    Args:
    argv	Command line arguments.
    """
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-m", "--mode", type=str, default="offline", help="Online, offline training or fill_buffer. Default %(default)s")
    parser.add_argument("-b", "--buffer", type=str, default="./buffers", help="Path to buffer. For offline training this overwrites buffer_pth in the config file.")
    parser.add_argument("-p", "--policy", type=str, default=None, help="Path to behavior policy. (If None for mode fill_buffer a random policy is used)")
    parser.add_argument("-s", "--steps", type=int, default=0, help="Number of steps to fill buffer with.")
    #parser.add_argument('--random', action=argparse.BooleanOptionalAction, help="Fill buffer with random actions.")
    parser.add_argument("-c", "--config", default=None, type=str, help="Path to config file.")
    parser.add_argument("-l", "--log", default=os.path.join(sys.path[0], "experiments"), type=str, help="Path to save log. Default %(default)s")
    parser.add_argument("-r", "--resume", default=None, type=str, help="Resume experiment at this path. Default %(default)s")

    parser.add_argument("--num_eval_eps", type=int, default=None, help="Number of evaluation episodes for online_eval/offline_eval.")

    args = parser.parse_args(argv)
    return args

def create_exp_dir(args, policy_type:str, env_name:str, time_str:str):

    exp_name = "{0}_{1}_{2}".format(policy_type,
                                    env_name,
                                    time_str)

    exp_pth = os.path.join(args.log, str(exp_name + time_str))
    tb_log_pth = os.path.join(exp_pth, "tb_log")
    config_pth = os.path.join(exp_pth, "config.yml")

    if not os.path.exists(args.log):
        os.mkdir(args.log)
    if not os.path.exists(exp_pth):
        # Create experiment directory
        os.mkdir(exp_pth)

    return exp_pth, tb_log_pth, config_pth

def save_checkpoint(iteration, loss, avg_reward, policy, exp_pth, train_collector=None):
    checkpoint_pth = os.path.join(exp_pth, "checkpoints")

    if not os.path.exists(checkpoint_pth):
        os.mkdir(checkpoint_pth)

    # Append file name
    checkpoint_name = os.path.join(checkpoint_pth, f"checkpoint_{iteration}.pth")
    
    torch.save({
                'training_iters': iteration,                            # Current training iteration
                'model_state_dict': policy.state_dict(),                # State of policy
                'optimizer_state_dict': policy.optimizer.state_dict(),  # State of optimizer
                'loss': loss,                                           # Current loss                    
                'reward': avg_reward                                    # Current average evaluation reward 
                }, checkpoint_name)                                     

    # For online training you also want to save the current replay buffer.
    if train_collector:
        buffer_pth = os.path.join(checkpoint_pth, f"train_buffer_{iteration}")
        #np.save(buffer_pth, train_collector)
        train_collector.save_buffer(buffer_pth)

def load_checkpoint(policy, checkpoint_pth, training_params, train_collector = None):
    checkpoint = torch.load(checkpoint_pth)
    policy.load_state_dict(checkpoint["model_state_dict"])
    policy.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    iteration = checkpoint["training_iters"]

    # Set iterations counter to continue from last value.
    training_iters = iteration + training_params["eval_freq"]

    if train_collector:
        buffer_pth = os.path.split(checkpoint_pth)[0]
        buffer_pth = os.path.join(buffer_pth, f"train_buffer_{iteration}.npz")

        train_collector.load_buffer(buffer_pth)
    
    return policy, training_iters, train_collector

