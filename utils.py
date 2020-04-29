# -*- coding: utf-8

import os
import yaml
import torch
import pickle as pkl
import datetime as dt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device  = torch.device("cpu")

def load_cfg(filepath="./config.yaml"):
    """Load YAML configuration file

    Params
    ======
    filepath (str): The path of the YAML configuration file
    """
    with open(filepath, "r") as f:
        return yaml.load(f)

def get_state_action_sizes(env):
    """Get the state and action space dimensions of a Gym environment

    Params
    ======
    env (gym.env): A Gym training environment.
    """
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.shape[0]
    return state_size, action_size

def get_model_parameters(model_list):
    """Gather model parameters into a single list.

    Params
    ======
    model_list (List[torch.nn.Module]): A list of PyTorch models."""
    all_parameters = []
    for model in model_list:
        for params in model.parameters():
            all_parameters.append(params)
    return all_parameters

def pkl_dump(data, fname):
    """Dump python object into a pickle file.

    Params
    ======
    data (object): A python object to be persisted to disk.
    fname (str): The path of the output pickle file.
    """
    with open(fname, "wb") as f:
        pkl.dump(data, f)

def pkl_load(fname):
    """Dump python object into a pickle file.

    Params
    ======
    data (object): A python object to be persisted to disk.
    fname (str): The path of the output pickle file.
    """
    with open(fname, "rb") as f:
        return pkl.load(f)


def yaml_dump(data, fname):
    """Save file in YAML format.

    Params
    ======
    data (dict): The dictionary to be persisted to a YAML file.
    fname (str): The path of the YAML output
    """
    with open(fname, "w") as f:
        yaml.dump(data, f)

def setup_experiment(cfg):
    """Setup directory to persist experiment results.

    Params
    ======
    cfg (dict): The dictionary containing the experiment's configuration.
    """
    t_experiment = dt.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    experiment_dir = "experiments/{}".format(t_experiment)
    if not os.path.isdir("experiments"):
        os.mkdir("experiments")
    os.mkdir(experiment_dir)
    yaml_dump(cfg, "./{}/config.yaml".format(experiment_dir))
    return experiment_dir

def persist_experiment(t_experiment, i_episode, agent, scores):
    """Persist experimental results.

    Params
    ======
    t_experiment (str): The relative path of the experimental results directory
    i_episode (int): The index of the episode where data is being persisted
    agent (class Agent): A DDPG agent initialized for this environment
    scores (List[])
    """
    os.chdir(t_experiment)
    torch.save(agent.actor_local.state_dict(), "checkpoint_actor_{}.pth".format(i_episode))
    torch.save(agent.critic_local.state_dict(), "checkpoint_critic_{}.pth".format(i_episode))
    pkl_dump(scores, "scores_{}.pkl".format(i_episode))
    os.chdir("../..")

def load_pretrained(agent, fname_actor, fname_critic):
    agent.actor_local.load_state_dict(torch.load(fname_actor))
    agent.actor_target.load_state_dict(torch.load(fname_actor))
    agent.critic_local.load_state_dict(torch.load(fname_critic))
    agent.critic_target.load_state_dict(torch.load(fname_critic))
    return agent
