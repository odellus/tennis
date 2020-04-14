import sys
import gym
from ddpg_agent import Agent


def get_agent_unity(cfg):
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



def get_agent_gym(cfg):
    # Unpack configuration
    random_seed = cfg["Environment"]["Random_seed"]

    # Get Pendulum environment and seed
    env = gym.make("Pendulum-v0")
    env.seed(random_seed)

    # Get stat and action sizes and initialize an agent
    state_size, action_size = get_state_action_sizes(env)
    agent = Agent(state_size=state_size, action_size=action_size, random_seed=random_seed, cfg=cfg)
    return env, agent
