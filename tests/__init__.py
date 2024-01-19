import os
from omegaconf import OmegaConf
_TEST_ROOT = os.path.dirname(__file__)  # root of test folder
_PROJECT_ROOT = os.path.dirname(_TEST_ROOT)  # root of project
_PATH_DATA = os.path.join(_PROJECT_ROOT, "Data")  # root of data

config_path = os.path.join(_TEST_ROOT, "config/test_conf.yml")

CONFIG = OmegaConf.load(config_path)
