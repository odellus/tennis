# **tennis**

Effort to train an agent (or agents) to play tennis collaboratively.

## **ENVIRONMENT**
Some important information about the `tennis` environment from Unity's ML-Agents: the size of the action space is two. The size of the naive state space is 8, but we use three consecutive states as one so the dimension of the state space used to map states onto actions is 24. The environment is considered solved when an average score of 0.5 over the last 100 episodes is reached.

## **SETUP**
If you don't wish to use MongoDB to persist experimental results be sure you have `Persist_mongodb` set to False in the configurations, and just run `bash setup.sh` to install.

If you wish to use MongoDB to persist the results of the experiments, please [install MongoDB](https://docs.mongodb.com/manual/tutorial/install-mongodb-on-ubuntu/), then run in the terminal

```bash
# Install dependencies
python3 -m pip install --user torch matplotlib numpy pymongo
# Get Tennis environment
curl -o Tennis_Linux_NoVis.zip https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux_NoVis.zip
unzip -d ./ddpg Tennis_Linux_NoVis.unzip
# Get the python API
git clone -b 0.4.0b https://github.com/Unity-Technologies/ml-agents.git ddpg/ml-agents
```  

## **RUN**
To train a DDPG network open a terminal and enter
```bash
python3 ddpg.py
```  
Good luck and happy hunting!
