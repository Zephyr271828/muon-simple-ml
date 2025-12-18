#!/bin/bash

LIMIT=10000

python moonlight/toy_train.py \
  --optimizer adamw \
  --limit $LIMIT

python moonlight/toy_train.py \
  --optimizer muon \
  --limit $LIMIT