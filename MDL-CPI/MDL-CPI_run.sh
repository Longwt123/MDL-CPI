#!/bin/bash

DATASET=human
#DATASET=celegans


radius=2
ngram=3

dim=32  #10

layer_gnn=3 #3
side=5  # 5
window=$((2*side+1))
layer_cnn=3
layer_output=3 # 3
lr=1e-3
lr_decay=0.5
decay_interval=10
weight_decay=1e-6
iteration=101

setting=$DATASET--radius$radius--ngram$ngram--dim$dim--layer_gnn$layer_gnn--window$window--layer_cnn$layer_cnn--layer_output$layer_output--lr$lr--lr_decay$lr_decay--decay_interval$decay_interval--weight_decay$weight_decay--iteration$iteration

nohup python -u MDL-CPI.py $DATASET $radius $ngram $dim $layer_gnn $window $layer_cnn $layer_output $lr $lr_decay $decay_interval $weight_decay $iteration $setting > Test.log 2>&1 & echo $! > MDL-CPI_PID.txt
