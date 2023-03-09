import mat73
import yaml
import os
import numpy as np
import scipy.sparse as sparse


def get_root():
    """Methods for getting the root of the repo"""
    return os.getcwd()


def get_config() -> dict:
    """Method for extracting the configuration from the config file"""
    file = get_root() + "/clustering/config.yml"
    with open(file) as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            print(exc)
    return config


def load_data(patient_id: int) -> dict:
    """Method to  load the .mat file and return is as an python dict"""
    config = get_config()
    path = config["general_path"] + "/" + str(patient_id) + "/" + config["file_path"]
    if os.path.exists(path):
        return mat73.loadmat(path)
    else:
        print("File doesn't exist")
        print(path)


def get_A(patient_id: int) -> sparse:
    """Method to extract the A (adjacent matrix)"""
    return load_data(patient_id)["G"]["A"]


def get_ind(patient_id: int) -> np.array:
    """Method to extract the indices"""
    return np.asarray(load_data(patient_id)["G"]["indices"], dtype=int)


def get_WM_ind(patient_id: int) -> np.array:
    """Method to extract the WM indices"""
    return np.asarray(load_data(patient_id)["G"]["indices_wm"], dtype=int)
