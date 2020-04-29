# **tennis**

Effort to train an agent (or agents) to play tennis collaboratively.

## **ENVIRONMENT**
Some important information about the `tennis` environment from Unity's ML-Agents.
```yaml
# The number of agents in the environment
n_agents:     2
# The dimension of a single state in the environment.
state_space: 24
# The dimension of the environment's action space
action_space: 2
# Average score over 100 episodes where environment is considered solved.
success:    0.5   

```
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
