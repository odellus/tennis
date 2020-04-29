# import gym
import sys
import random
import torch
import numpy as np
import pickle as pkl
from collections import deque
import matplotlib.pyplot as plt
# %matplotlib inline

from ddpg_agent import Agent



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





# env = gym.make('BipedalWalker-v2')
# env.seed(10)
# agent = Agent(state_size=env.observation_space.shape[0], action_size=env.action_space.shape[0], random_seed=10)
def get_agent_unity():
    """Step Unity environment forward one timestep

    Params
    ======
        env (UnityEnvironment): The Unity environment to step forwards
        action (int): The action index to take during this timestep
        brain_index (int): The brain index of the agent we wish to act
        brain_name (str): The name of the brain we wish to act
    """
    unity_pythonpath = "./ml-agents/python"
    file_name = "./Tennis_Linux_NoVis/Tennis.x86_64"
    seed = 4
    brain_index = 0
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
    agent = Agent(state_size=state_size, action_size=action_size, random_seed=seed)
    return env, agent

def step_unity(env, actions, brain_name):
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

def ddpg(env, agent, n_episodes=4000, max_t=1500, success=0.5):
    """Core DDPG learning

    Params
    ======
        env (UnityEnvironment): The Unity environment to step forwards
        agent (ddpg_agent.Agent): A Deep RL agent written in PyTorch.
        n_episodes (int): Number of episodes to train the agent.
        max_t (int): Maximum length in timesteps of episode.
        success (float): Average score where environment is considered solved.
    """
    brain_index = 0
    scores_deque = deque(maxlen=100)
    scores = []
    brain_name = env.brain_names[brain_index]
    max_score = -np.Inf
    n_pretrain = None
    if n_pretrain is not None:
        for i_pretrain in range(n_pretrain):
            env_info = env.reset(train_mode=True)[brain_name]
            n_agents = len(env_info.agents)
            states = env_info.vector_observations
            agent.reset()
            actions = np.zeros((n_agents, agent.action_size))
            score = np.zeros(n_agents)
            for t in range(max_t):
                # action = agent.act(state)
                # for i_agent in range(n_agents):
                #     actions[i_agent, :] = agent.act(states[i_agent, :])
                actions = 2 * np.random.rand(n_agents, agent.action_size) - 1.0
                next_states, rewards, dones, _ = step_unity(env, actions, brain_name)
                for i_agent in range(n_agents):
                    agent.step(
                        states[i_agent, :],
                        actions[i_agent, :],
                        rewards[i_agent],
                        next_states[i_agent, :],
                        dones[i_agent]
                        )
                # agent.step(state, action, reward, next_state, done)
                states = next_states
                score += rewards
                if np.any(dones):
                    break
            scores_deque.append(score)
            scores.append(score)
            mean_score = np.vstack(scores_deque).max(axis=1).mean()
            print('\rPretrain episode {}\tAverage Score: {:.2f}\tScore: {:.4f} {:.4f}'.format(i_pretrain, mean_score, score[0], score[1]), end="")
    print("\nResetting weights of neural networks.")
    agent.actor_local.reset_parameters()
    agent.actor_target.reset_parameters()
    agent.critic_local.reset_parameters()
    agent.critic_target.reset_parameters()
    scores_deque = deque(maxlen=100)
    print(f"Pretraining random sampling finished after {n_pretrain} episodes")
    scores = []
    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name]
        n_agents = len(env_info.agents)
        states = env_info.vector_observations
        agent.reset()
        actions = np.zeros((n_agents, agent.action_size))
        score = np.zeros(n_agents)

        for t in range(max_t):
            # action = agent.act(state)
            for i_agent in range(n_agents):
                actions[i_agent, :] = agent.act(states[i_agent, :])
            next_states, rewards, dones, _ = step_unity(env, actions, brain_name)
            for i_agent in range(n_agents):
                agent.step(
                    states[i_agent, :],
                    actions[i_agent, :],
                    rewards[i_agent],
                    next_states[i_agent, :],
                    dones[i_agent]
                    )
            # agent.step(state, action, reward, next_state, done)
            states = next_states
            score += rewards
            if np.any(dones):
                break
        scores_deque.append(score)
        scores.append(score)
        mean_score = np.vstack(scores_deque).max(axis=1).mean()
        print('\rEpisode {}\tAverage Score: {:.2f}\tScore: {:.4f} {:.4f}'.format(i_episode, mean_score, score[0], score[1]), end="")
        if i_episode % 100 == 0:
            torch.save(agent.actor_local.state_dict(), f'checkpoint_actor_{i_episode}.pth')
            torch.save(agent.critic_local.state_dict(), f'checkpoint_critic_{i_episode}.pth')
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, mean_score))
        if mean_score >= success:
            # This is going to be the first thing I'd be digging in the database for,
            # so here you go.
            fpath_actor = "checkpoint_actor_winner_{}.pth".format(i_episode)
            fpath_critic = "checkpoint_critic_winner_{}.pth".format(i_episode)
            pkl_dump(scores, f'scores_winner_{i_episode}.pkl')
            torch.save(agent.actor_local.state_dict(), fpath_actor)
            torch.save(agent.critic_local.state_dict(), fpath_critic)
            break
    return scores

def plot_scores(scores):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(1, len(scores)+1), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()

def main():
    env, agent = get_agent_unity()
    scores = ddpg(env, agent)
    plot_scores(scores)
    pkl_dump(scores, 'scores.pkl')


# scores = ddpg()
# agent.actor_local.load_state_dict(torch.load('checkpoint_actor.pth'))
# agent.critic_local.load_state_dict(torch.load('checkpoint_critic.pth'))


if __name__ == "__main__":
    main()
