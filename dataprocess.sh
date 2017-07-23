#!/bin/bash

directory="/scratch/aolpin/testing/rawdata/images/"

for file in "$directory"/*
do
  sqsub -r 5m -n 1 -f serial --mpp=256m -o dataset.log python preprocessor-p.py $file
done