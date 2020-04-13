# -*- coding: utf-8
"""
file: experiment_db.py
"""
import io
import torch
from datetime import datetime as dt
import numpy as np
from pymongo import MongoClient

def get_client():
    """
    """
    return MongoClient('localhost', 27017)

def get_experiment_db(client):
    """
    """
    return client["experiments"]

def get_db():
    """
    """
    return get_experiment_db(get_client())

def get_tennis(db):
    """
    """
    return db["tennis"]

def drop_tennis(db):
    """
    """
    if "tennis" in db.list_collection_names():
        res = db.drop_collection("tennis")
        assert res["ok"] == 1.0

def drop_weights(db):
    """
    """
    if "weights" in db.list_collection_names():
        res = db.drop_collection("weights")
        assert res["ok"] == 1.0

def get_weights(experiments):
    """
    """
    return experiments["weights"]

def setup_experiment(experiments, cfg):
    """
    """
    tennis = get_tennis(experiments)
    date = dt.utcnow()
    experiment = {
        "date": date,
        "config": cfg
    }
    return tennis.insert_one(experiment).inserted_id

def update_scores(experiment, scores, score_window):
    """
    """
    assert isinstance(scores, list)
    assert len(scores) > 0
    if experiment.get("scores") is None:
        experiment["scores_player1"] = [x[0] for x in scores]
        experiment["scores_player2"] = [x[1] for x in scores]
    else:
        experiment["scores"].extend(scores[-score_window:])
    return experiment

def extract_actor(agent):
    """
    """
    with io.BytesIO() as f:
        torch.save(agent.actor_local.state_dict(), f)
        return f.getvalue()

def extract_critic(agent):
    """
    """
    with io.BytesIO() as f:
        torch.save(agent.critic_local.state_dict(), f)
        return f.getvalue()

def insert_weights(weights, agent, i_episode, date, experiment_id):
    """
    """
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

def extract_weights(weights_db, weight_id, agent):
    """
    """
    weights = weights_db.find_one({"_id": weight_id})
    actor_weights = weights["actor"]
    critic_weights = weight["critic"]
    agent = load_actor(actor_weights, agent)
    agent = load_critic(critic_weights, agent)
    return agent

def find_max_score(tennis_db, score_window):
    """
    """
    max_scores = []
    for document in tennis_db.find():
        max_score, weight_id = extract_max_score(weight_db, document, score_window)
        max_scores.append((max_score, weight_id))
    max_scores = remove_nulls(max_scores)
    scores, weight_ids = split_tuples(max_scores)
    # return optimal weight_id
    return weight_ids[ np.argmax(scores) ]

def get_optimal_agent(tennis_db, weights_db, agent, score_window):
    weight_id = find_max_score(tennis_db, score_window)
    agent = extract_weights(weights_db, weight_id, agent)


def remove_nulls(y):
    """
    """
    return [x for x in y if x != (None, None)]

def split_tuples(y):
    """
    """
    x1 = [x[0] for x in y]
    x2 = [x[1] for x in y]
    return x1, x2


def extract_max_score(document, score_window):
    """
    """
    max_score, weight_idx = None, None
    p1_scores = document.get("scores_player1")
    p2_scores = document.get("scores_player2")
    if p1_scores is None or p2_scores is None:
        return max_score, weight_idx
    # Not going to include and else statement, rest of function is else:
    p1_avg = sliding_window_average(p1_scores, score_window)
    p2_avg = sliding_window_average(p2_scores, score_window)
    avg_mat = np.array([p1_avg, p2_avg])

    p_argmax = np.array([np.argmax(avg_mat[0,:]), np.argmax(avg_mat[1,:])])

    p_max = [avg_mat[k, p_argmax[k]] for k in range(p_argmax.shape[0])]

    winner = np.argmax(p_max)
    i_episode = p_argmax[winner]

    max_score = p_max[winner]
    weight_idx = i_episode // score_window
    # weight_idx[0] is for episode 100 so we want (weight_idx-1)-th element.
    weight_id = document["weights"][weight_idx-1]
    return max_score, weight_id



def sliding_window_average(x, n):
    """
    """
    m = len(x)
    assert m >= n
    return [np.mean(x[k:k+n]) for k in range(len(x)-n+1)]

def load_actor(weights, agent):
    """
    """
    agent.actor_local.load_state_dict(torch.load(io.BytesIO(weights)))
    agent.actor_target.load_state_dict(torch.load(io.BytesIO(weights)))
    return agent

def load_critic(weights, agent):
    """
    """
    with io.BytesIO(weights) as w:
        agent.actor_local.load_state_dict(torch.load(w))
        agent.actor_target.load_state_dict(torch.load(w))
    return agent


def update_weights(weights, experiment, agent, i_episode, date, experiment_id):
    weight_id = insert_weights(weights, agent, i_episode, date, experiment_id)
    if experiment.get("weights") is None:
        experiment["weights"] = [weight_id]
    else:
        experiment["weights"].append(weight_id)
    return experiment

def persist_experiment(experiments, experiment_id, i_episode, agent, scores, score_window):
    tennis = get_tennis(experiments)
    weights = get_weights(experiments)
    experiment = tennis.find_one({"_id": experiment_id})
    # Date is determined when experiment is initially run and must be passed to update_weights.
    date = experiment["date"]
    experiment = update_scores(experiment, scores, score_window)
    experiment = update_weights(weights, experiment, agent, i_episode, date, experiment_id)
    res = tennis.replace_one({"_id": experiment_id}, experiment)
    assert res.modified_count == 1, "Problem with modified count"
    assert res.matched_count == 1, "Problem with matched count"
