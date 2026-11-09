#!/bin/bash
# Recreate the baseline files that are not in git: the AnySleep repository at its pinned
# commit (it holds the AnySleep and U-Sleep checkpoints), the sleep-graph-benchmark repository
# (GraphSleepNet, MSTGCN, ProductGraphSleepNet) and the CBraMod pretrained weights.
# Safe to run again; existing files are checked, not replaced.
set -euo pipefail
cd "$(dirname "$0")/.."

anysleep=baselines/anysleep/upstream
anysleep_commit=899bfff284f3eb3e7448bc90990f2230757470a2
if [ ! -d "${anysleep}/.git" ]; then
  git clone https://github.com/dslaborg/AnySleep "${anysleep}"
fi
git -C "${anysleep}" checkout --quiet "${anysleep_commit}"
ls -l "${anysleep}"/models/anysleep-run?.pth "${anysleep}"/models/usleep-run?.pth

sleepgraph=baselines/sleepgraph/upstream
sleepgraph_commit=51440c3ab9d88d41f669f09214a6eca385e955a2
if [ ! -d "${sleepgraph}/.git" ]; then
  git clone https://github.com/simona-juvina/sleep-graph-benchmark "${sleepgraph}"
fi
git -C "${sleepgraph}" checkout --quiet "${sleepgraph_commit}"

weights=baselines/cbramod/weights/pretrained_weights.pth
weights_sha256=0792cb808c14e6b7a2bb2ce1dff379bc47bc54c49a779825bdfeb33bf8157178
if [ ! -f "${weights}" ]; then
  mkdir -p "$(dirname "${weights}")"
  curl -fL -o "${weights}" \
    https://huggingface.co/weighting666/CBraMod/resolve/500543c7e30bda1b22bfd51a49301b238dee21fd/pretrained_weights.pth
fi
echo "${weights_sha256}  ${weights}" | sha256sum --check
