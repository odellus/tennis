# **tennis**

Effort to train an agent (or agents) to play tennis collaboratively.

## **SETUP**
[Install MongoDB](https://docs.mongodb.com/manual/tutorial/install-mongodb-on-ubuntu/), then run in the terminal
```bash
# Install dependencies
python3 -m pip install --user torch pymongo matplotlib numpy
# Get Tennis environment
curl -o Tennis_Linux_NoVis.zip https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux_NoVis.zip
unzip Tennis_Linux_NoVis.unzip
cp -r Tennis_Linux_NoVis ddpg/Tennis_Linux_NoVis
# Get the python API
git clone -b 0.4.0b https://github.com/Unity-Technologies/ml-agents.git ml-agents
cp -r ml-agents ddpg/ml-agents
```  

## **RUN**
To train a DDPG network open a terminal and enter
```bash
cd ddpg
python3 ddpg.py
```  
Good luck!
