#!/usr/bin/env bash
torch-model-archiver --model-name "trocr-hand-written" --version 1.0 \
--serialized-file  ./raw_model/pytorch_model.bin \
--extra-files "./raw_model/config.json" \
--handler "handler.py"