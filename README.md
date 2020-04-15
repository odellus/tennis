# **tennis**

Effort to train an agent (or agents) to play tennis collaboratively.

To install the pre-requisite libraries first:
1. [install MongoDB](https://docs.mongodb.com/manual/tutorial/install-mongodb-on-ubuntu/) then
2. Run in the terminal
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
3. Open [`ddpg/config.yaml`](https://github.com/odellus/tennis/blob/master/ddpg/config.yaml#L11) and change `Environment:Unity_pythonpath` to `./ml-agents/python`  
4. To train a DDPG network open a terminal and enter
```bash
cd ddpg
python3 ddpg.py
```
5. Good luck!
