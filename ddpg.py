# -*- coding: utf-8
import sys
import os
import gym
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl

from collections import deque

from utils import load_cfg, setup_experiment, persist_experiment
from environment import get_agent_unity, step_unity
from experiment_db import get_db, setup_experiment, persist_experiment


def ddpg(env, agent, cfg, db):
    # Get configuration
    n_episodes = cfg["Training"]["Number_episodes"]
    max_t = cfg["Training"]["Max_timesteps"]
    print_every = cfg["Training"]["Score_window"]
    brain_index = cfg["Agent"]["Brain_index"]

    #Initialize score lists
    scores_deque = deque(maxlen=print_every)
    scores = []
    # Create a directory to save the findings.
    # experiment_dir = setup_experiment(cfg)
    experiment_id = setup_experiment(db, cfg)
    brain_name = env.brain_names[brain_index]
    # Train for n_episodes
    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name]
        n_agents = len(env_info.agents)
        states = env_info.vector_observations
        agent.reset()
        actions = np.zeros((n_agents, agent.action_size))
        score = np.zeros(n_agents)
        for t in range(max_t):
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
            states = next_states
            score += rewards
            if np.any(dones):
                break
        scores_deque.append(np.max(score))
        scores.append(np.max(score))
        print("\rEpisode {}\tAverage Score: {:.4f}".format(i_episode, np.mean(scores_deque)), end="")
        # print("\rEpisode {}\tAverage Score: {}".format(i_episode, scores_deque), end="")

        visualize = False
        if i_episode % print_every == 0:
            # persist_experiment(experiment_dir, i_episode, agent, scores)
            persist_experiment(db, experiment_id, i_episode, agent, scores, print_every)
            print("\rEpisode {}\tAverage Score: {:.4f}".format(i_episode, np.mean(scores_deque)))


    return scores

def main():
    cfg = load_cfg()
    db = get_db()
    env, agent = get_agent_unity(cfg)
    # agent = load_pretrained(agent)
    scores = ddpg(env ,agent, cfg, db)

if __name__ == "__main__":
    main()
