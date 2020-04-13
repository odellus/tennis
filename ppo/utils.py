import torch
import yaml

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device  = torch.device("cpu")

def load_cfg(filepath="./config.yaml"):
    """Load YAML configuration file

    Params
    ======
    filepath (str): The path of the YAML configuration file
    """
    with open(filepath, "r") as f:
        return yaml.load(f)
