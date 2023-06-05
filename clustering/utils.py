import datetime
import logging
import os
from pathlib import Path

import numpy as np
import yaml
from scipy import io, sparse


def get_root():
    """Methods for getting the root of the repository"""
    return str(Path(__file__).resolve().parent.parent)


def get_config() -> dict:
    """Method for extracting the information from the config file"""
    file = get_root() + "/clustering/config.yml"
    with open(file) as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            print(exc)
    return config


def get_output_dir() -> str:
    """Method for returning the general output_dir from the config_file"""
    return get_config()["output_dir"]


def get_output_dir_subject(subject_id: int) -> str:
    """Method that return the path to a specific patient output_dir"""
    return f"{get_output_dir()}/{subject_id}"


def get_Laplacien_path(subject_id: int, L_type: str, threshold=2.0) -> str:
    """Method for getting the specific laplacien file of a subject"""
    output_path = get_output_dir_subject(subject_id)
    if L_type == "comb":
        Laplacien = [
            f for f in os.listdir(output_path) if f.endswith(f"{threshold}_L.npz")
        ]
        if len(Laplacien) != 1:
            print(f"No matching laplacien matrix:{output_path}{threshold}_L.npz")
            return None
        else:
            return output_path + "/" + Laplacien[0]

    elif L_type == "rw":
        Laplacien = [
            f for f in os.listdir(output_path) if f.endswith(f"{threshold}_Lrw.npz")
        ]
        if len(Laplacien) != 1:
            print(f"No matching laplacien matrix:{output_path}{threshold}_L.npz")
            return None
        else:
            return output_path + "/" + Laplacien[0]
    elif L_type == "sym":
        Laplacien = [
            f for f in os.listdir(output_path) if f.endswith(f"{threshold}_Lsym.npz")
        ]
        if len(Laplacien) != 1:
            print(f"No matching laplacien matrix:{output_path}{threshold}_L.npz")
            return None
        else:
            return output_path + "/" + Laplacien[0]

    else:
        print("Laplacien type unknown")
        return None


def get_nifti_path(subject_id: int, L_type: str, nifti_type: str, threshold: int()):
    """Method for getting the path to a specific nifti of a patient"""
    output_path = get_output_dir_subject(subject_id)
    work_id = get_Laplacien_path(subject_id, L_type, threshold)[
        len(output_path) + 1 : len(output_path) + 22
    ]

    if nifti_type == "acpc":
        nifti = [
            f
            for f in os.listdir(output_path)
            if f.endswith(f"{work_id}{threshold}_extraction_acpc.nii.gz")
        ]
        if len(nifti) != 1:
            print(f"No matching  nifti file:{output_path}")
            return None
        else:
            return output_path + "/" + nifti[0]
    elif nifti_type == "mni":
        nifti = [
            f
            for f in os.listdir(output_path)
            if f.endswith(f"{work_id}{threshold}_extraction_mni.nii.gz")
        ]
        if len(nifti) != 1:
            print(f"No matching  nifti file:{output_path}")
            return None
        else:
            return output_path + "/" + nifti[0]
    elif nifti_type == "reslice":
        nifti = [
            f
            for f in os.listdir(output_path)
            if f.endswith(f"{work_id}{threshold}_extraction_mni_reslice.nii.gz")
        ]
        if len(nifti) != 1:
            print(f"No matching  nifti file:{output_path}")
            return None
        else:
            return output_path + "/" + nifti[0]

    else:
        print(f"Can not find the correct nifti file:{output_path}")

    return None


def get_U_path(subject_id: int, L_type: str, threshold: int) -> str:
    """Method for extracting the path to the U matrix .npy file of a patient"""
    path_output_dir = create_output_folder(get_output_dir(), subject_id, "subject")
    work_id = get_Laplacien_path(subject_id, L_type, threshold)[
        len(path_output_dir) + 1 : len(path_output_dir) + 22
    ]
    files = [
        f
        for f in os.listdir(path_output_dir)
        if f.endswith(f"{work_id}{threshold}_U.npy")
    ]
    if len(files) > 1:
        print("Multiple files found for U")
        return " "
    else:
        return path_output_dir + "/" + files[0]


def get_v_path(subject_id: int, L_type: str, threshold: int) -> str:
    """Method for extracting the path to the v matrix .npy file of a patient"""
    path_output_dir = create_output_folder(get_output_dir(), subject_id, "subject")
    work_id = get_Laplacien_path(subject_id, L_type, threshold)[
        len(path_output_dir) + 1 : len(path_output_dir) + 22
    ]
    files = [
        f
        for f in os.listdir(path_output_dir)
        if f.endswith(f"{work_id}{threshold}_v.npy")
    ]
    if len(files) != 1:
        print("Multiple files found for v")
        return " "
    else:
        return path_output_dir + "/" + files[0]


def get_A(patient_id: int) -> sparse:
    """Method to extract the A (adjacent matrix) from the matlab file"""
    A = load_data(patient_id)["G"]["A"]
    logging.info(f"A, shape:{A.shape}")
    return A


def get_src_nifti(patient_id: int) -> str:
    config = get_config()
    return config["general_path"] + "/" + str(patient_id) + config["src_nifi_path"]


def get_ind(patient_id: int) -> np.array:
    """Method to extract the indices from the matlab file"""
    return np.asarray(load_data(patient_id)["G"]["indices"], dtype=int)


def get_WM_ind(patient_id: int) -> np.array:
    """Method to extract the WM indices from the matlab file"""
    return np.asarray(load_data(patient_id)["G"]["indices_wm"], dtype=int)


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
    """Method that return the path to the reference file"""
    config = get_config()
    path = (
        config["general_path"]
        + "/"
        + str(subject_id)
        + "/MNINonLinear/T1w_restore_brain.nii.gz"
    )
    return path


def get_mask_path(type: str):
    """Method that returns the path to the WM mask in the mni space"""
    path = get_config()["general_path"]
    path = path + "/Templates"
    if type == "95":
        path = path + "/HCP100_cerebrum_graph_wm_template_binary_thresh95Percent.nii.gz"
    elif type == "50":
        path = path + "/HCP100_cerebrum_graph_wm_template_binary_thresh50Percent.nii.gz"
    return path


def check_Laplacien(subject_id: int, L_type: str, threshold=2.0) -> bool:
    """Method for checking if an output folder contains a specific Laplacian matrix"""
    if get_Laplacien_path is None:
        print("Laplacien not found:", subject_id, L_type, threshold)
        return False
    else:
        return True


def check_nifti(subject_id: int, L_type: str, nifti_type: str, threshold: int()):
    """Method for checking if an output folder contains a specific nifti file"""
    if get_nifti_path is None:
        print("Nifti file not found:", subject_id, L_type, threshold)
        return False
    else:
        return True


def check_output_folder(output_path: str, patient_id: int) -> bool:
    """Boolean function for checking if the ouptut folder of a specific subject
    exits.
    """
    path = output_path + "/" + f"{patient_id}"
    if os.path.isdir(path):
        return True
    else:
        return False


def create_output_folder(output_path: str, patient_id: int, type=str):
    """Method for creating the output folder of a subject if doesn't exist"""
    if type == "subject":
        path = output_path + "/" + f"{patient_id}"
    elif type == "population":
        path = output_path + "/population_cluster"
    elif type == "boostrap":
        path = path = output_path + "/population_cluster/Bootstrap"

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
    value_type=["cluster"],
    save: bool = False,
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
            f"-k:{k_eigen},-values {value_type} , -s:{save} "
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


def load_npy(path: str) -> np.array:
    """Method for loading the .npy files"""
    try:
        return np.load(path)
    except FileNotFoundError as err:
        print("File data doesn't exist")
        return err
