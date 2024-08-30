# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# CroCo submodule import
# --------------------------------------------------------

import sys
import os.path as path
# HERE_PATH = path.normpath(path.dirname(__file__))
# CROCO_REPO_PATH = path.normpath(path.join(HERE_PATH, '../../croco'))
# CROCO_MODELS_PATH = path.join(CROCO_REPO_PATH, 'models')

# print(CROCO_MODELS_PATH)
CROCO_MODELS_PATH="/local/home/rsaxena/Documents/Github/Master-Thesis-Project/Experiments/dust3r_experiments/dust3r/croco/models"

#"/home/rupal/Master-Thesis-Project/Experiments/dust3r_experiments/dust3r/croco/models"

# print(CROCO_REPO_PATH)
# check the presence of models directory in repo to be sure its cloned
if path.isdir(CROCO_MODELS_PATH):
    # workaround for sibling import
    sys.path.insert(0, "/local/home/rsaxena/Documents/Github/Master-Thesis-Project/Experiments/dust3r_experiments/dust3r/croco")
else:
    raise ImportError(f"croco is not initialized, could not find: {CROCO_MODELS_PATH}.\n "
                      "Did you forget to run 'git submodule update --init --recursive' ?")
