/**
 * Copyright 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "alignment_train_mps.h"
#include "utils.h"

namespace {

void alignmentTrainmps(
    const torch::Tensor& p_choose,
    torch::Tensor& alpha,
    float eps) {
  CHECK_INPUT(p_choose);
  CHECK_INPUT(alpha);

  alignmentTrainmpsWrapper(p_choose, alpha, eps);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "alignment_train_mps",
      &alignmentTrainmps,
      "expected_alignment_from_p_choose (mps)");
}

} // namespace
