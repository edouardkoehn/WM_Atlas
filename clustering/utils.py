import datetime
import logging
import os
from pathlib import Path

import numpy as np
import yaml
from scipy import io, sparse


def get_root():
    """Methods for getting the root of the repo"""
    return str(Path(__file__).resolve().parent.parent)


def get_config() -> dict:
    """Method for extracting the configuration from the config file"""
    file = get_root() + "/clustering/config.yml"
    with open(file) as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            print(exc)
    return config


def get_output_dir() -> str:
    """Method for returning the output_dir"""
    return get_config()["output_dir"]


def check_Laplacien(ouptput_path: str, L_type: str, threshold: int):
    if L_type == "comb":
        Laplacien = [
            f for f in os.listdir(ouptput_path) if f.endswith(f"{threshold}_L.npz")
        ]
        if len(Laplacien) != 1:
            print(f"No matching laplacien matrix:{ouptput_path}")
            return False
        else:
            return True

    elif L_type == "rw":
        Laplacien = [
            f for f in os.listdir(ouptput_path) if f.endswith(f"{threshold}_Lrw.npz")
        ]
        if len(Laplacien) != 1:
            print(f"No matching laplacien matrix:{ouptput_path}")
            return False
        else:
            return True
    else:
        print("Laplacien type unknown")
        return False


def check_nifti(ouptput_path: str, L_type: str, nifti_type: str, threshold: int):
    if check_Laplacien(ouptput_path, L_type, threshold):
        if nifti_type == "acpc":
            nifti = [
                f
                for f in os.listdir(ouptput_path)
                if f.endswith(f"{threshold}_extraction_acpc.nii.gz")
            ]
            if len(nifti) != 1:
                print(f"No matching  nifti file:{ouptput_path}")
                return False
            else:
                return ouptput_path + "/" + nifti[0]
        elif nifti_type == "mni":
            nifti = [
                f
                for f in os.listdir(ouptput_path)
                if f.endswith(f"{threshold}_extraction_mni.nii.gz")
            ]
            if len(nifti) != 1:
                print(f"No matching  nifti file:{ouptput_path}")
                return False
            else:
                return ouptput_path + "/" + nifti[0]
    else:
        print(f"Can not find the correct nifti file:{ouptput_path}")

    return


def check_output_folder(output_path: str, patient_id: int) -> bool:
    path = output_path + "/" + f"{patient_id}"
    if os.path.isdir(path):
        return True
    else:
        return False


def create_output_folder(output_path: str, patient_id: int, type=str):
    """Method for creating the output folder if doesn't exist"""
    if type == "subject":
        path = output_path + "/" + f"{patient_id}"
    elif type == "population":
        path = output_path + "/population_cluster"
    if not (os.path.isdir(path)):
        try:
            os.mkdir(path)
        except os.error as er:
            print(er)
    return path


def create_logs(
    path_logs: str,
    patient_id: int,
    date: datetime,
    method: str,
    threshold: float,
    k_eigen: str,
    nifti: str,
    save: bool,
):
    """Method for creating the logs"""
    if save:
        logging.basicConfig(
            filename=path_logs,
            level=logging.INFO,
            format="%(asctime)s--%(levelname)s-%(message)s",
            filemode="w",
        )
        logging.info(f"Clustering - Patient_id: {patient_id} - Date: {date}")
        logging.info(
            f"Parameters: -i:{patient_id} , -m:{method}, -t:{threshold},-n:{nifti}"
            f"-k:{k_eigen}, -s:{save} "
        )
    return


def load_data(patient_id: int) -> dict:
    """Method to  load the .mat file and return is as an python dict"""
    config = get_config()
    path = config["general_path"] + "/" + str(patient_id) + "/" + config["file_path"]
    try:
        return io.loadmat(path, simplify_cells=True)
    except FileNotFoundError as err:
        print("File data doesn't exist")
        return err


def get_A(patient_id: int) -> sparse:
    """Method to extract the A (adjacent matrix)"""
    A = load_data(patient_id)["G"]["A"]
    logging.info(f"A, shape:{A.shape}")
    return A


def get_ind(patient_id: int) -> np.array:
    """Method to extract the indices"""
    return np.asarray(load_data(patient_id)["G"]["indices"], dtype=int)


def get_WM_ind(patient_id: int) -> np.array:
    """Method to extract the WM indices"""
    return np.asarray(load_data(patient_id)["G"]["indices_wm"], dtype=int)


def load_npy(path: str) -> np.array:
    """Method for loading the .npy files"""
    try:
        return np.load(path)
    except FileNotFoundError as err:
        print("File data doesn't exist")
        return err


def get_transformation_file(subject_id: int, type: str):
    """Method that return the path to the transformation file"""
    config = get_config()
    path = config["general_path"] + "/" + str(subject_id) + "/MNINonLinear/xfms"
    if type == "acpc2nmi":
        path = path + "/acpc_dc2standard.nii.gz"
        return path
    if type == "nmi2acpc":
        path = path + "/standard2acpc_dc.nii.gz"
        return path
    else:
        print("Transformation type unknown")
        return os.error.errno


def get_reference_file(subject_id: int):
    config = get_config()
    path = (
        config["general_path"]
        + "/"
        + str(subject_id)
        + "/MNINonLinear/T1w_restore_brain.nii.gz"
    )
    return path
