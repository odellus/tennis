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

from utils import load_cfg, setup_experiment, persist_experiment, pkl_dump
from environment import get_agents_unity, get_agent_unity, step_unity
from experiment_db import get_db, setup_experiment, persist_experiment


def ddpg_multiagent(env, agent1, agent2, cfg, db):
    # Get configuration
    n_episodes = cfg["Training"]["Number_episodes"]
    max_t = cfg["Training"]["Max_timesteps"]
    print_every = cfg["Training"]["Score_window"]
    random_episodes = cfg["Training"]["Random_episodes"]
    starting_random = cfg["Training"]["Starting_random"]
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
        agent1.reset()
        agent2.reset()
        actions = np.zeros((n_agents, agent1.action_size))
        score = np.zeros(n_agents)
        for t in range(max_t):
            if i_episode % random_episodes == 0 or i_episode <= starting_random:
                actions = 2 * np.random.rand(n_agents, agent1.action_size) - 1.0
            else:
                actions[0, :] = agent1.act(states[0, :])
                actions[1, :]= agent2.act(states[1, :])
                # for i_agent in range(n_agents):
                #     actions[i_agent, :] = agent.act(states[i_agent, :])
            next_states, rewards, dones, _ = step_unity(env, actions, brain_name)
            for i_agent in range(n_agents):
                # Add experience to both of the agent's replay buffers
                agent1.step(
                    states[i_agent, :],
                    actions[i_agent, :],
                    rewards[i_agent],
                    next_states[i_agent, :],
                    dones[i_agent]
                )
                agent2.step(
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
        scores_deque.append(score)
        scores.append(score)
        mean_score = np.vstack(scores_deque).mean(axis=0).max()
        print("\rEpisode {}\tAverage Score: {:.4f}\tNoise Modulation: {:.3f}".format(i_episode, mean_score, agent1.noise_modulation), end="")
        # print("\rEpisode {}\tAverage Score: {}".format(i_episode, scores_deque), end="")

        visualize = False
        if i_episode % print_every == 0:
            # persist_experiment(experiment_dir, i_episode, agent, scores)
            persist_experiment(db, experiment_id, i_episode, agent1, scores, print_every)
            persist_experiment(db, experiment_id, i_episode, agent2, scores, print_every)
            print("\rEpisode {}\tAverage Score: {:.4f}".format(i_episode, mean_score))


    return scores

def ddpg_selfplay(env, agent, cfg, db):
    # Get configuration
    n_episodes = cfg["Training"]["Number_episodes"]
    max_t = cfg["Training"]["Max_timesteps"]
    print_every = cfg["Training"]["Score_window"]
    random_episodes = cfg["Training"]["Random_episodes"]
    starting_random = cfg["Training"]["Starting_random"]
    brain_index = cfg["Agent"]["Brain_index"]
    dump_agent = cfg["Training"]["Dump_agent"]
    success = cfg["Environment"]["Success"]


    #Initialize score lists
    scores_deque = deque(maxlen=print_every)
    scores = []
    # Create a directory to save the findings.
    # experiment_dir = setup_experiment(cfg)
    experiment_id = setup_experiment(db, cfg)
    print("Experiment ID: {}".format(experiment_id))
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
            if i_episode % random_episodes == 0 or i_episode <= starting_random:
                actions = 2 * np.random.rand(n_agents, agent.action_size) - 1.0
            else:
                for i_agent in range(n_agents):
                    actions[i_agent, :] = agent.act(states[i_agent, :])
            next_states, rewards, dones, _ = step_unity(env, actions, brain_name)
            for i_agent in range(n_agents):
                # Add experience to both of the agent's replay buffers
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
        scores_deque.append(score)
        scores.append(score)
        mean_score = np.vstack(scores_deque).mean(axis=0).max()
        print("\rEpisode {}\tAverage Score: {:.4f}\tNoise Modulation: {:.3f}".format(i_episode, mean_score, agent.noise_modulation), end="")
        # print("\rEpisode {}\tAverage Score: {}".format(i_episode, scores_deque), end="")

        visualize = False
        if i_episode % print_every == 0:
            # persist_experiment(experiment_dir, i_episode, agent, scores)
            persist_experiment(db, experiment_id, i_episode, agent, scores, print_every)
            # persist_experiment(db, experiment_id, i_episode, agent2, scores, print_every)
            print("\rEpisode {}\tAverage Score: {:.4f}".format(i_episode, mean_score))
        if i_episode % dump_agent == 0:
            # To heck with it let's save some to disk even though I have utilities to load.
            # It's important.
            fpath_actor = "checkpoint_actor_{}.pth".format(i_episode)
            fpath_critic = "checkpoint_critic_{}.pth".format(i_episode)
            torch.save(agent.actor_local.state_dict(), fpath_actor)
            torch.save(agent.critic_local.state_dict(), fpath_critic)
        if mean_score >= success:
            # This is going to be the first thing I'd be digging in the database for,
            # so here you go.
            fpath_actor = "checkpoint_actor_winner_{}.pth".format(i_episode)
            fpath_critic = "checkpoint_critic_winner_{}.pth".format(i_episode)
            torch.save(agent.actor_local.state_dict(), fpath_actor)
            torch.save(agent.critic_local.state_dict(), fpath_critic)
            pkl_dump(scores, "scores_winner.pkl")
            break

    return scores

def multiagent(cfg):
    cfg = load_cfg()
    db = get_db()
    env, agent1, agent2 = get_agents_unity(cfg)
    return ddpg_multiagent(env, agent1, agent2, cfg, db)

def selfplay(cfg):
    cfg = load_cfg()
    db = get_db()
    env, agent = get_agent_unity(cfg)
    # agent = load_pretrained(agent)
    return ddpg_selfplay(env ,agent, cfg, db)

def main():
    cfg = load_cfg()
    self_play = cfg["Training"]["Self_play"]
    if self_play:
        scores = selfplay(cfg)
    else:
        scores = multiagent(cfg)


if __name__ == "__main__":
    main()
