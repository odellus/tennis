import sys
import gym
import torch
from ddpg_agent import Agent


def get_agents_unity(cfg):
    # Unpack configuration
    unity_pythonpath = cfg["Environment"]["Unity_pythonpath"]
    file_name = cfg["Environment"]["Filepath"]
    seed = cfg["Environment"]["Random_seed"]
    brain_index = cfg["Agent"]["Brain_index"]
    # Append unityagents directory to sys.path
    sys.path.append(unity_pythonpath)
    # Now this will work.
    from unityagents import UnityEnvironment
    # Create an environment
    env = UnityEnvironment(file_name=file_name, seed=seed)
    # Get information about the environment
    brain_name = env.brain_names[brain_index]
    brain = env.brains[brain_name]
    state_size = brain.vector_observation_space_size * brain.num_stacked_vector_observations
    action_size = brain.vector_action_space_size
    # Create an agent using state and action sizes of environment
    agent1 = Agent(state_size=state_size, action_size=action_size, random_seed=seed, cfg=cfg)
    agent2 = Agent(state_size=state_size, action_size=action_size, random_seed=seed, cfg=cfg)
    return env, agent1, agent2

def get_agent_unity(cfg):
    # Unpack configuration
    unity_pythonpath = cfg["Environment"]["Unity_pythonpath"]
    file_name = cfg["Environment"]["Filepath"]
    seed = cfg["Environment"]["Random_seed"]
    brain_index = cfg["Agent"]["Brain_index"]
    pretrained = cfg["Training"]["Pretrained"]
    # Append unityagents directory to sys.path
    sys.path.append(unity_pythonpath)
    # Now this will work.
    from unityagents import UnityEnvironment
    # Create an environment
    env = UnityEnvironment(file_name=file_name, seed=seed)
    # Get information about the environment
    brain_name = env.brain_names[brain_index]
    brain = env.brains[brain_name]
    state_size = brain.vector_observation_space_size * brain.num_stacked_vector_observations
    action_size = brain.vector_action_space_size
    # Create an agent using state and action sizes of environment
    agent = Agent(state_size=state_size, action_size=action_size, random_seed=seed, cfg=cfg)
    # if pretrained:
    #     agent = load_agent_weights(agent, cfg)
    #     print("Loaded pretrained weights!")
    return env, agent


def load_agent_weights(agent, cfg):
    actor_fname = cfg["Training"]["Actor_fname"]
    critic_fname = cfg["Training"]["Critic_fname"]
    agent.actor_local.load_state_dict(torch.load(actor_fname))
    agent.critic_local.load_state_dict(torch.load(critic_fname))
    return agent

def step_unity(
    env,
    actions,
    brain_name
    ):
    """Step Unity environment forward one timestep

    Params
    ======
        env (UnityEnvironment): The Unity environment to step forwards
        action (int): The action index to take during this timestep
        brain_index (int): The brain index of the agent we wish to act
        brain_name (str): The name of the brain we wish to act
    """
    env_info = env.step(actions)[brain_name]
    return env_info.vector_observations, \
           env_info.rewards, \
           env_info.local_done, \
           env_info
