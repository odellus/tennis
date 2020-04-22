python3 -m pip install --user torch matplotlib numpy
# Get Tennis environment
curl -o Tennis_Linux_NoVis.zip https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux_NoVis.zip
unzip -d ./ddpg Tennis_Linux_NoVis.unzip
# Get the python API
git clone -b 0.4.0b https://github.com/Unity-Technologies/ml-agents.git ddpg/ml-agents
