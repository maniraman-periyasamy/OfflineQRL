from torch.utils.tensorboard import SummaryWriter

import sys
import time
import gym

import numpy as np
import torch

from src.algorithms.policy import Policy
from src.algorithms.discreteBCQ import discreteBCQ
from src.algorithms.discreteCQL import discreteCQL
from src.algorithms.DQN import DQN
from src.models.mlp import Mlp
from src.models.skolikVqc import SkolikVqc
from src.models.hwSkolikVqc import HwSkolikVqc
from src.models.vqc2 import Vqc2
from src.models.vqc3 import Vqc3

from utils import *
from src.collector import Collector
from src.buffer import Buffer

from src.environments.quantumCartPoleWrapper import QuantumCartPoleWrapper
from src.environments.quantumIBliteWrapper import QuantumIBLiteWrapper
from src.environments.quantumHwCartPoleWrapper import QuantumHwCartPoleWrapper
import matplotlib.pyplot as plt

# For quantum agents fix device to cpu since there is no real GPU suppport.
# Furthermore, qiwrap is used for parallel execution of backward pass.
DEVICE = "cpu"

# Dictionary with policy constructors.
POLICIES = {
    "DQN":         DQN,
    "discreteBCQ": discreteBCQ,
    "discreteCQL": discreteCQL
}

# Dictionary with model constructors.
MODELS = {
    "MLP": Mlp,
    "SkolikVQC": SkolikVqc,
    "HwSkolikVQC": HwSkolikVqc
}

# Dictionary with environment wrappers.
ENVIRONMENTS = {
    "CartPole": QuantumCartPoleWrapper,
    "HardwareCartPole": QuantumHwCartPoleWrapper
}

def eval_policy(policy: Policy, 
                env: gym.Wrapper | gym.Env, 
                seed: int = None, 
                eval_episodes: int = 1, 
                max_return: float | int = 500.0) -> (int, bool):       
    """ Runs policy for eval_episodes and returns average return per episode.
    Args
    policy          Policy to evaluate.
    env             Environment to evaluate policy in.
    seed            Ignored for now. (seed for envionement)
    eval_episodes   Number of evaluation episodes.
    max_return      Maximum return from environment to determine end of episode if environement does not terminate or we want to stop early.
    
    Returns
    avg_return      Return averaged over all episodes.
    early_stopping  Flag indicating that all episodes reached max_return.
    """

    # Put policy in evaluation mode.
    policy.eval()
    avg_return = 0.
    for _ in range(eval_episodes):
        #state, _ = eval_env.reset(seed=seed)
        state, _ = env.reset()
        episode_avg = 0.0
        done = False
        while not done:
            action = policy.select_action(np.array(state), eval=True)
            state, reward, terminated, truncated, info = env.step(action)
            done = np.logical_or(terminated, truncated)
            avg_return += reward
            episode_avg += reward
            
            if episode_avg >= max_return:
                print(f"Max reward of {episode_avg} achieved.")
                done = True

    avg_return /= eval_episodes
    # Set flag that all environment evaluations returned max_return.
    early_stopping = True if avg_return >= max_return else False

    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_return:.3f}.")
    print("---------------------------------------")
    return avg_return, early_stopping

def train_online(training_config: dict, 
                   writer: SummaryWriter, 
                   exp_pth: str, 
                   train_collector: Collector, 
                   test_env: gym.Wrapper | gym.Env, 
                   policy: Policy,  
                   step_per_collect: int =10, 
                   max_training_steps: int = 25000, 
                   eval_episodes: int = 10, 
                   checkpoint_pth: str = None, 
                   create_checkpoints: bool = False,
                   num_random_steps: int = 5000) -> None:
    """ Training loop to train an online policy e.g., DQN.
    Args
    training_config     Configuration of train loop.
    writer              Tensorboard summary writer to log training.
    exp_pth             Path to experiment directory, where checkpoints etc. are saved.
    train_collector     Collector object used to interact with environment during training.
    test_env            Environment for validation during traing.
    policy              Online policy to train.
    step_per_collect    Number of steps to collect in between training steps.
    max_training_steps  Maximum number of training steps.
    eval_episodes       Number of evaluation episodes.
    checkpoint_pth      Path to a checkpoint to resume training from there. (Set to None to train new model)
    create_checkpoints  Flag to save checkpoints during training.
    num_random_steps    Number of random steps to pre-fill replay before training starts.
    """

    if checkpoint_pth:
        policy, training_iters, train_collector = load_checkpoint(policy, checkpoint_pth, training_config, train_collector)
    else:
        # Pre-fill buffer with random interactions to sample from during first training steps
        collect_result = train_collector.collect(num_random_steps, random=True)
        training_iters = 0

    avg_reward = 0

    while training_iters < max_training_steps: 

        policy.train()

        for _ in range(int(training_config["eval_freq"])):
            collect_result = train_collector.collect(n_steps=step_per_collect)
            loss = policy.train_step(train_collector.buffer)

        avg_reward, early_stopping = eval_policy(policy, test_env, 1, eval_episodes=eval_episodes, max_return=training_config["max_return"])

        training_iters += int(training_config["eval_freq"])

        # Create checkpoint after each evaluation.
        if create_checkpoints:
            save_checkpoint(training_iters, loss, avg_reward, policy, exp_pth, train_collector)

        print(f"Training iterations: {training_iters}")
        
        writer.add_scalar("Loss/train", loss, training_iters)
        writer.add_scalar("Reward/test", avg_reward, training_iters)

        if early_stopping:
            print("Training stopped, because average validation reward reached maximum reward.")
            break
    
    model_pth = os.path.join(exp_pth, "model")
    policy.save(model_pth)

def train_offline(training_config: dict, 
                  writer: SummaryWriter, 
                  exp_pth: str, 
                  behavior_env: gym.Wrapper | gym.Env, 
                  test_env: gym.Wrapper | gym.Env, 
                  policy: Policy, 
                  behavior_policy: Policy = None, 
                  buffer_size: int = int(1e6), 
                  buffer_pth: str = None, 
                  checkpoint_pth: str = None, 
                  create_checkpoints: bool = False,
                  max_training_steps: int = 25000, 
                  eval_episodes: int = 10,
                  early_stopping: bool = False):
    
    """ Training loop to train an offline policy e.g., discreteBCQ.
    Args
    training_config     Configuration of train loop.
    writer              Tensorboard summary writer to log training.
    exp_pth             Path to experiment directory, where checkpoints etc. are saved.
    behavior_policy     Behavior policy to fill training buffer with. (Set to None to fill buffer with random policy)
    buffer_size         Size of training buffer.
    buffer_pth          Path to saved buffer. (Either fill buffer with behavior_policy or load a buffer)
    checkpoint_pth      Path to a checkpoint to resume training from there. (Set to None to train new model)
    max_training_steps  Maximum number of training steps.
    eval_episodes       Number of evaluation episodes. (Set to 0 if you do not want to evaluate policy during training)
    early_stopping      Flag to stop training if max_return is reached in evaluation. (Caution: stricly speaking not offline learning anymore, but helpful to save computational resources)
    """

    # Check if checkpoint is available to resume training
    if checkpoint_pth:
        policy, training_iters, _ = load_checkpoint(policy, checkpoint_pth, training_config)

    else:
        training_iters = 0

    state_dim = test_env.state_dim
    buffer = Buffer(state_dim, 1, training_config["batch_size"], buffer_size, DEVICE)

    # Check for pre-filled buffer
    if buffer_pth is None:
        try:
            # Collect a buffer using the behavior policy.
            behavior_collector = Collector(behavior_env, buffer, behavior_policy)
            result = behavior_collector.collect(buffer_size)
            # Save buffer to resume training.
            buffer.save("cql_test_buffer")
            print("Buffer filled!")
            rew = result["rew"]
            print(f"Collector average reward: {rew}")
            buffer = behavior_collector.buffer
        except:
            raise ValueError(f"Environment to collect buffer invalid: {behavior_env}. \n Use fill_buffer and set buffer_pth in config of this experiment to the filled buffer.")
    else:
        buffer.load(buffer_pth)  
        print(f"Loaded Buffer: {buffer_pth}")  

    avg_reward = 0

    while training_iters < max_training_steps: 
        policy.train()

        for _ in range(int(training_config["eval_freq"])):
            loss = policy.train_step(buffer)

        if eval_episodes > 0:
            avg_reward, early_stopping = eval_policy(policy, test_env, 1, eval_episodes=eval_episodes, max_return=training_config["max_return"])
            writer.add_scalar("Reward/test", avg_reward, training_iters)
        else:
            avg_reward, early_stopping = None, False

        training_iters += int(training_config["eval_freq"])

        if create_checkpoints:
            save_checkpoint(training_iters, loss, avg_reward, policy, exp_pth)

        print(f"Training iterations: {training_iters}")
        
        writer.add_scalar("Loss/train", loss, training_iters)

        # Early stopping only if early_stopping_flag is True.
        if early_stopping and early_stopping:
            print("Training stopped, because average validation reward reached maximum reward.")
            break

    print(f"Final reward: {avg_reward}")

    # Save trained policy after final training step.
    model_pth = os.path.join(exp_pth, "model")
    policy.save(model_pth)
    writer.close()

def setup_experiment(args: argparse.Namespace) -> (str, SummaryWriter, dict):
    """ Setup an experiment directory, initialise tensorboard summary writer and load config file. In case you resume training also copies previous log to new directory.
    Args
    args    Comandline arguments.

    Returns
    exp_pth Path to experiment directory.
    writer  Summary writer to log training.
    config  Configuration loaded from config file.
    """

    time_str = time.strftime("%Y%m%d-%H%M%S")

    if args.resume is None:
        # Start new training.
        config = read_config_from_file(args.config)

        exp_pth, tb_log_pth, config_pth = create_exp_dir(args, 
                                                            config["policy"]["name"],
                                                            config["environment"]["name"],
                                                            time_str)
    else:
        # Resume training.
        # Expect checkpoint to be in directory checkpoints on the same level as config.yml
        prev_exp_pth = args.resume
        for i in range(2):
            # Remove tail from path.
            prev_exp_pth = os.path.split(prev_exp_pth)[0]

        # Load config from previous experiment.
        prev_config_pth = os.path.join(prev_exp_pth, "config.yml")
        # Ignores config given via command line and loads config of experiment to be resumed.
        config = read_config_from_file(prev_config_pth)

        exp_pth, tb_log_pth, config_pth = create_exp_dir(args, 
                                                            config["policy"]["name"],
                                                            config["environment"]["name"],
                                                            time_str)

        # Get path of tensorboard log from previous experiment
        prev_log_pth = os.path.join(prev_exp_pth, "tb_log")

        if not os.path.exists(tb_log_pth):
            os.mkdir(tb_log_pth)

        # Copy file os dependent
        if os.name == 'nt':
            # Windows
            os.system(str("copy " + prev_log_pth + " " + exp_pth))
        else:
            # Linux
            os.system(str("cp -r " + prev_log_pth + " " + exp_pth))

    writer = SummaryWriter(tb_log_pth)

    # Log path to used buffer.
    if args.buffer is not None:
        config["training"]["buffer_pth"] = args.buffer

    # Log config
    write_config_to_file(config_pth, config)
    return exp_pth, writer, config

def setup_online_training(config: dict) -> (Policy, Collector, gym.Wrapper | gym.Env):
    """ Setup online training: initialise environment, policy and training collector.
    Args
    config  Configuration loaded from config file.

    Returns
    policy          Initialised policy.
    train_collector Initilaised collector.
    test_env        Initialised environment for validation.
    """

    environment_config = config["environment"]
    model_config = config["models"]
    policy_config = config["policy"]
    training_config = config["training"]

    # Setup Environements
    train_env = ENVIRONMENTS[environment_config["name"]].fromDict(environment_config)
    test_env = ENVIRONMENTS[environment_config["name"]].fromDict(environment_config)
    
    # Setup model and optimizer
    model = MODELS[model_config["name"]].fromDict(model_config)
    optim = getattr(torch.optim, config["optimizer"])(model.parameters(), **config["optimizer_parameters"])

    state_dim = train_env.state_dim
    num_actions = train_env.action_space.shape or train_env.action_space.n

    policy = POLICIES[policy_config["name"]].fromDict(model,
                                            optim,
                                            num_actions,
                                            state_dim,
                                            policy_config)

    buffer = Buffer(state_dim, 1, training_config["batch_size"], training_config["replay_buffer_size"], DEVICE)

    train_collector = Collector(train_env, buffer, policy)

    return policy, train_collector, test_env

def setup_offline_training(config: dict) -> (Policy, gym.Wrapper | gym.Env | None, gym.Wrapper | gym.Env | None ):
    """ Setup offline training: initialise environment and policy
    Args
    config  Configuration loaded from config file.

    Returns
    policy          Initialised policy.
    behavior_env    For now just returns None, because user should use fill_buffer() to create buffer and load it for offline training.
    test_env        If config:training:eval_episodes > 0 initialized environment, otherwise None.
    """

    environment_config = config["environment"]
    model_config = config["models"]
    policy_config = config["policy"]
    training_config = config["training"]
    optim_config = config["optimizer"]

    # Check if online validation should be performed.
    if training_config["eval_episodes"] > 0:
        # behavior_env is used to fill buffer in case no pre-filled buffer is used. 
        # behavior_env = ENVIRONMENTS[environment_params["name"]].fromDict(environment_params)

        # Check that environment is implemented.
        assert environment_config["name"] in ENVIRONMENTS, "Environment {} not found.".format(environment_config["name"])
        # Not supported right now. Always us fill_buffer() and pass path to buffer for offline training.
        behavior_env = None
        test_env = ENVIRONMENTS[environment_config["name"]].fromDict(environment_config)
        # Get state dim, num_actions from test envrionment.
        state_dim = test_env.state_dim
        num_actions = test_env.action_space.shape or test_env.action_space.n
    else:
        # Get state dim, num_actions from config.
        state_dim = environment_config["state_dim"]
        num_actions = environment_config["num_actions"]

        behavior_env = None
        test_env = None
    
    # Setup model and optimizer
    if policy_config["name"] == "discreteBCQ":
        Q_config = model_config["qNet"]
        I_config = model_config["imitator"]

        model = MODELS[Q_config["name"]].fromDict(Q_config)
        imitator = MODELS[I_config["name"]].fromDict(I_config)

        if Q_config.get("load_pretrained", None) is not None:
            model.load_pretrained(Q_config["load_pretrained"], Q_config["load_parameter_dict"])
        
        if I_config.get("load_pretrained", None) is not None:
            imitator.load_pretrained(I_config["load_pretrained"], I_config["load_parameter_dict"])
        
        # Allow to have individual learning rates for each set of parameters.
        param_groups = []

        for name, param in model.named_parameters():
            # If a learning rate is specified in the config use it otherwise use general learning rate.
            param_groups.append({"params": [param], "lr": optim_config["learning_rates"]["qNet"].get(name, optim_config["general"]["lr"])})

        for name, param in imitator.named_parameters():
            # If a learning rate is specified in the config use it otherwise use general learning rate.
            param_groups.append({"params": [param], "lr": optim_config["learning_rates"]["imitator"].get(name, optim_config["general"]["lr"])})
        
        optim = getattr(torch.optim, optim_config["name"])(param_groups, **optim_config["general"])

        policy = POLICIES[policy_config["name"]].fromDict(model,
                                                 imitator,
                                                 optim,
                                                 num_actions,
                                                 state_dim,
                                                 policy_config)
    else:
        model = MODELS[model_config["name"]].fromDict(model_config)
        
        if model_config.get("load_pretrained", None) is not None:
            model.load_pretrained(model_config["load_pretrained"], model_config["load_parameter_dict"])

        param_groups = []
        
        for name, param in model.named_parameters():
            # If a learning rate is specified in the config use it otherwise use general learning rate.
            param_groups.append({"params": [param], "lr": optim_config["learning_rates"].get(name, optim_config["general"]["lr"])})

        optim = getattr(torch.optim, optim_config["name"])(param_groups, **optim_config["general"])

        policy = POLICIES[policy_config["name"]].fromDict(model,
                                                 optim,
                                                 num_actions,
                                                 state_dim,
                                                 policy_config)

    return policy, behavior_env, test_env

def fill_buffer(args: argparse.Namespace) -> None:
    """ Fill a buffer with a policy saved at args.policy (Expects path to experiment with trained policy).
    Note: saves buffer as numpy .npz file.
          If args.policy is None, use random policy.

    Args
    args Commandline arguments.
    """

    time_str = time.strftime("%Y%m%d-%H%M%S")

    if args.policy is None:
        policy = None
        config = read_config_from_file(args.config)
        
        environment_config = config["environment"]
        #model_config = config["models"]
        #policy_config = config["policy"]
        training_config = config["training"]

        # Setup Environements
        env = ENVIRONMENTS[environment_config["name"]].fromDict(environment_config)

        state_dim = env.state_dim
        num_actions = env.action_space.shape or env.action_space.n

        policy_name = "Random"
        env_name = environment_config["name"]

    else:
        prev_log_pth = os.path.join(args.policy, "config.yml")
        config = read_config_from_file(prev_log_pth)

        environment_config = config["environment"]
        model_config = config["models"]
        policy_config = config["policy"]
        training_config = config["training"]

        # Setup Environements
        env = ENVIRONMENTS[environment_config["name"]].fromDict(environment_config)
        model = MODELS[model_config["name"]].fromDict(model_config)
        #optim = getattr(torch.optim, parameters["optimizer"])(model.parameters(), **parameters["optimizer_parameters"])

        state_dim = env.state_dim
        num_actions = env.action_space.shape or env.action_space.n

        policy = POLICIES[policy_config["name"]].fromDict(model,
                                                None,       # No optimizer required for inference.
                                                num_actions,
                                                state_dim,
                                                policy_config)

        model_pth = os.path.join(args.policy, "model")
        policy.load(model_pth)

        # Put in evaluation mode. -> use eval_eps for action selection
        policy.eval()
        policy_name = policy_config["name"]
        env_name = environment_config["name"]

    buffer = Buffer(state_dim, 1, training_config["batch_size"], args.steps, DEVICE)

    collector = Collector(env, buffer, policy)
    result = collector.collect(args.steps)
    
    buffer_pth = os.path.join(args.buffer, f"{env_name}_{policy_name}_{args.steps}_{time_str}.npz")
    
    rew = result["rew"]
    
    collector.save_buffer(buffer_pth)
    num_ep = result["n/ep"]

    print(f"Buffer filled with {args.steps} steps with {num_ep} episodes and saved at {args.buffer}.")    

def main(argv):
    args = parse_args(argv)
    
    if args.mode == "fill_buffer":
        fill_buffer(args)

    else:
        exp_pth, writer, config = setup_experiment(args)
        
        if args.mode == "online":
            policy, train_collector, test_env = setup_online_training(config)
            train_online(config["training"], 
                         writer, 
                         exp_pth, 
                         train_collector, 
                         test_env, 
                         policy,
                         config["training"]["step_per_collect"],
                         config["training"]["max_training_steps"],
                         config["training"]["eval_episodes"],
                         args.resume,
                         config["training"]["create_checkpoints"],
                         config["training"]["num_random_steps"])
        
        elif args.mode == "offline":
            policy, behavior_env, test_env = setup_offline_training(config)
            train_offline(config["training"],
                          writer,
                          exp_pth,
                          behavior_env,
                          test_env,
                          policy,
                          None,     # Use fill_buffer to create a buffer with a behavior policy.
                          config["training"]["buffer_size"],
                          config["training"]["buffer_pth"],
                          args.resume, # Checkpoint path
                          config["training"]["create_checkpoints"],
                          config["training"]["max_training_steps"],
                          config["training"]["eval_episodes"],
                          config["training"]["early_stopping_flag"])

        elif args.mode == "online_eval":
            policy, train_collector, test_env = setup_online_training(config)
            policy, _, _ = load_checkpoint(policy, args.resume, config["training"])

            eval_policy(policy, test_env, eval_episodes=args.num_eval_eps, max_return=config["training"]["max_return"])

        elif args.mode == "offline_eval":
            policy, behavior_env, test_env = setup_offline_training(config)
            policy, _, _ = load_checkpoint(policy, args.resume, config["training"])

            eval_policy(policy, test_env, eval_episodes=args.num_eval_eps, max_return=config["training"]["max_return"])
        else:
            raise NotImplementedError(f"Mode: {args.mode} not implemented.")

if __name__ == "__main__":
    main(sys.argv[1:])