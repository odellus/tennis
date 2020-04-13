import sys

def get_tennis_env(cfg):
    # Unpack configuration
    unity_pythonpath = cfg["Environment"]["Unity_pythonpath"]
    file_name = cfg["Environment"]["Filepath"]
    seed = cfg["Environment"]["Random_seed"]
    brain_index = cfg["Agent"]["Brain_index"]
    # Append unityagents directory to sys.path
    sys.path.append(unity_pythonpath)
    # Now this will work.
    from unityagents import UnityEnvironment
    # Create an environment
    return UnityEnvironment(file_name=file_name, seed=seed)

def collect_trajectories(env, policy, tmax):
    pass
    # return old_actions, states, actions, rewards
