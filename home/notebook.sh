#!/bin/bash
source /home/u/uhengart/ezzeldi/.virtualenvs/jupyter/bin/activate
XDG_DATA_HOME=$SCRATCH/.share
XDG_CACHE_HOME=$SCRATCH/.cache
XDG_CONFIG_HOME=$SCRATCH/.config
XDG_RUNTIME_DIR=$SCRATCH/.runtime
JUPYTER_CONFIG_DIR=$SCRATCH/.config/.jupyter
jupyter ${1:-notebook} --ip $(hostname -f) --no-browser --notebook-dir=$PWD
