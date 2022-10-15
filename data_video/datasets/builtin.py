# Copyright (c) 2021-2022, NVIDIA Corporation & Affiliates. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://github.com/NVlabs/MinVIS/blob/main/LICENSE

# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/sukjunhwang/IFC

import os,sys
sys.path.append(os.path.dirname(os.path.realpath('/home/featurize/sparseinstovis/data_video/datasets/ytvis_api')))

from ytvis import (
    register_ytvis_instances,
    _get_ovis_instances_meta,
)

# ==== Predefined splits for OVIS ===========
_PREDEFINED_SPLITS_OVIS = {
    "ovis_train": ("OVIS/train",
                         "OVIS/annotations/annotations_train.json"),
    "ovis_val": ("OVIS/valid",
                       "OVIS/annotations/annotations_valid.json"),
    "ovis_test": ("OVIS/test",
                        "OVIS/annotations/annotations_test.json"),
}


def register_all_ovis(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_OVIS.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_ytvis_instances(
            key,
            _get_ovis_instances_meta(),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
        )


if __name__ == "__main__":
    # Assume pre-defined datasets live in `./datasets`.
    _root = "/home/featurize"
    register_all_ovis(_root)
