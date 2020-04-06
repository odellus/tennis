# -*- coding: utf-8
"""
file: experiment_db.py
"""
import io
import torch
from datetime import datetime as dt
from pymongo import MongoClient

def get_client():
    return MongoClient('localhost', 27017)

def get_experiment_db(client):
    return client["experiments"]

def get_db():
    return get_experiment_db(get_client())

def get_tennis_collection(experiments):
    return experiments["tennis"]

def get_weights_collection(experiments):
    return experiments["weights"]

def setup_experiment(experiments, cfg):
    tennis = get_tennis_collection(experiments)
    date = dt.utcnow()
    experiment = {
        "date": date,
        "config": cfg
    }
    return tennis.insert_one(experiment).inserted_id

def update_scores(experiment, scores, score_window):
    assert isinstance(scores, list)
    if experiment.get("scores") is None:
        experiment["scores"] = scores
    else:
        experiment["scores"].extend(scores[-score_window:])
    return experiment

def extract_actor(agent):
    with io.BytesIO() as f:
        torch.save(agent.actor_local.state_dict(), f)
        return f.getvalue()

def extract_critic(agent):
    with io.BytesIO() as f:
        torch.save(agent.critic_local.state_dict(), f)
        return f.getvalue()

def insert_weights(weights, agent, i_episode, date, experiment_id):
    actor = extract_actor(agent)
    critic = extract_critic(agent)
    entry = {
        "actor": actor,
        "critic": critic,
        "date": date,
        "i_episode": i_episode,
        "experiment_id": experiment_id
        }
    return weights.insert_one(entry).inserted_id

def update_weights(weights, experiment, agent, i_episode, date, experiment_id):
    weight_id = insert_weights(weights, agent, i_episode, date, experiment_id)
    if experiment.get("weights") is None:
        experiment["weights"] = [weight_id]
    else:
        experiment["weights"].append(weight_id)
    return experiment

def persist_experiment(experiments, experiment_id, i_episode, agent, scores, score_window):
    tennis = get_tennis_collection(experiments)
    weights = get_weights_collection(experiments)
    experiment = tennis.find_one({"_id": experiment_id})
    # Date is determined when experiment is initially run and must be passed to update_weights.
    date = experiment["date"]
    experiment = update_scores(experiment, scores, score_window)
    experiment = update_weights(weights, experiment, agent, i_episode, date, experiment_id)
    res = tennis.replace_one({"_id": experiment_id}, experiment)
    assert res.modified_count == 1, "Problem with modified count"
    assert res.matched_count == 1, "Problem with matched count"
