import os

import pytest

from clustering import utils


def test_get_root():
    root = utils.get_root()
    assert root[-8:] == "WM_Atlas"


def test_get_config():
    keys = ["root_dir", "output_dir", "general_path", "dir_path", "file_path"]

    config = list(utils.get_config().keys())
    for i in range(0, len(config)):
        assert config[i] == keys[i]


def test_output_dir():
    output = utils.get_output_dir()
    assert os.path.exists(output)


@pytest.mark.parametrize("output_dir,patient_id", [(utils.get_output_dir(), 404)])
def test_output_folder(output_dir, patient_id):
    output = utils.check_output_folder(output_dir, patient_id)
    assert os.path.exists(output)
    os.rmdir(output)


@pytest.mark.parametrize("patient_id", [(404)])
def test_load_data(patient_id):
    out = utils.load_data(patient_id)
    print(type(out))
    assert type(out) == FileNotFoundError
