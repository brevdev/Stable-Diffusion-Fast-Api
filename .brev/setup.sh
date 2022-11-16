#!/bin/bash

set -eo pipefail

conda init zsh
conda init bash 
eval "$(conda shell.bash hook)"
conda env create -f environment.yaml
conda activate sd-api-server