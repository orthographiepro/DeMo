#!/bin/bash
set -e # Exit immediately if a command exits with a non-zero status

# git clone git@github.com:OpenGVLab/VideoMamba.git
cd VideoMamba
pip install -e causal-conv1d
pip install -e mamba
