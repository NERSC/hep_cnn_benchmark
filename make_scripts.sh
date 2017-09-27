#!/bin/bash

#some module loads
module load python/3.6-anaconda-4.4

#set up directories
mkdir -p scripts/networks

#convert classifier
jupyter-nbconvert --to=python networks/binary_classifier_tf.ipynb
cp networks/binary_classifier_tf.py scripts/networks/

#convert run script
jupyter-nbconvert --to=python hep_classifier_tf_train.ipynb
mv hep_classifier_tf_train.py scripts/

