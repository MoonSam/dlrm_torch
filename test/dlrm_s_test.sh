#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

#WARNING: must have compiled PyTorch and caffe2

#check if extra argument is passed to the test
if [[ $# == 1 ]]; then
    dlrm_extra_option=$1
else
    dlrm_extra_option=""
fi
#echo $dlrm_extra_option

dlrm_py="python dlrm_s_pytorch.py"

echo "Running commands ..."
#run pytorch
echo $dlrm_py
$dlrm_py --mini-batch-size=1 --data-size=1 --use-gpu --nepochs=1 --arch-interaction-op=dot --learning-rate=0.1 --debug-mode $dlrm_extra_option > ppp1
$dlrm_py --mini-batch-size=2 --data-size=4 --nepochs=1 --arch-interaction-op=dot --learning-rate=0.1 --debug-mode $dlrm_extra_option > ppp2
$dlrm_py --mini-batch-size=2 --data-size=5 --nepochs=1 --arch-interaction-op=dot --learning-rate=0.1 --debug-mode $dlrm_extra_option > ppp3
$dlrm_py --mini-batch-size=2 --data-size=5 --nepochs=3 --arch-interaction-op=dot --learning-rate=0.1 --debug-mode $dlrm_extra_option > ppp4

