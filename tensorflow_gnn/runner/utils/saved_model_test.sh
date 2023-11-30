#!/bin/bash
#
# Copyright 2021 The TensorFlow GNN Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

get_binary () {
  echo "runner/utils/$1"
}

# Generate the saved model.
gen_test_data_par="${TEST_SRCDIR}/$(get_binary 'saved_model_gen_testdata')"
readonly gen_test_data_par

$gen_test_data_par --filepath=${TEST_TMPDIR}/saved_model_testdata || die "Failed to execute $gen_test_data_par"

# Attempt to load the saved model.
testpar="${TEST_SRCDIR}/$(get_binary 'saved_model_load_testdata')"
readonly testpar

$testpar --filepath=${TEST_TMPDIR}/saved_model_testdata || die "Failed to execute $testpar"

echo "PASS"