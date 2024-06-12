/**
 * Copyright 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <torch/extension.h>

torch::Tensor LevenshteinDistancemps(
    torch::Tensor source,
    torch::Tensor target,
    torch::Tensor source_length,
    torch::Tensor target_length);

torch::Tensor GenerateDeletionLabelmps(
    torch::Tensor source,
    torch::Tensor operations);

std::pair<torch::Tensor, torch::Tensor> GenerateInsertionLabelmps(
    torch::Tensor source,
    torch::Tensor operations);
