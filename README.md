# **tennis**

Effort to train an agent (or agents) to play tennis collaboratively.

## **SETUP**
If you don't wish to use MongoDB to persist experimental results, just run `bash setup.sh` to install.

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
cd ddpg  
python3 ddpg.py
```  
Good luck!
