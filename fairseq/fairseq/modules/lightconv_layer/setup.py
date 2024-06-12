#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, mpsExtension


setup(
    name="lightconv_layer",
    ext_modules=[
        mpsExtension(
            "lightconv_mps",
            [
                "lightconv_mps.cpp",
                "lightconv_mps_kernel.cu",
            ],
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
