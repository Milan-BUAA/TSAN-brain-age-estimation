#!/bin/bash
#                  model      loss   batch   lbd    beta      num_pair   extra
# ============ CNN ================= #
# bash bash_train.sh CNN        mae    18       0     1          40
# wait


# ============ DenseNet ================= #
bash bash_train.sh CNN    mae    12      30     0.1        60  
wait
bash bash_train.sh CNN    mae    12      30     0.2        60  
wait
bash bash_train.sh CNN    mae    12      30     0.4        60  
wait
bash bash_train.sh CNN    mae    12      30     0.5        60  
wait
bash bash_train.sh CNN    mae    12      30     0.8        60  
wait




