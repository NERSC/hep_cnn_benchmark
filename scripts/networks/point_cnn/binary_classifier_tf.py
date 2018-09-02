
# coding: utf-8

# In[ ]:

#*** License Agreement ***                                                                                                                                                                                                                                                                                  
#                                                                                                                                                                                                                                                                                                           
#High Energy Physics Deep Learning Convolutional Neural Network Benchmark (HEPCNNB) Copyright (c) 2017, The Regents of the University of California,                                                                                                                                                        
#through Lawrence Berkeley National Laboratory (subject to receipt of any required approvals from the U.S. Dept. of Energy). All rights reserved.                                                                                                                                                           
#                                                                                                                                                                                                                                                                                                           
#Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:                                                                                                                                                             
#(1) Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.                                                                                                                                                                           
#(2) Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer                                                                                                                                                                         
#in the documentation and/or other materials provided with the distribution.                                                                                                                                                                                                                                
#(3) Neither the name of the University of California, Lawrence Berkeley National Laboratory, U.S. Dept. of Energy nor the names                                                                                                                                                                            
#of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.                                                                                                                                                                       
#THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING,                                                                                                                                                                              
#BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE                                                                                                                                                                   
#COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT                                                                                                                                                           
#LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF                                                                                                                                                      
#LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,                                                                                                                                                          
#EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
#You are under no obligation whatsoever to provide any bug fixes, patches, or upgrades to the features,                                                                                                                                                                                                     
#functionality or performance of the source code ("Enhancements") to anyone; however,                                                                                                                                                                                                                       
#if you choose to make your Enhancements available either publicly, or directly to Lawrence Berkeley National Laboratory,                                                                                                                                                                                   
#without imposing a separate written license agreement for such Enhancements, then you hereby grant the following license: a non-exclusive,                                                                                                                                                                 
#royalty-free perpetual license to install, use, modify, prepare derivative works, incorporate into other computer software,                                                                                                                                                                                
#distribute, and sublicense such enhancements or derivative works thereof, in binary and source code form.                                                                                                                                                                                                  
#---------------------------------------------------------------      


# In[1]:

#os stuff
import os
import root_numpy as rnp
import itertools

#numpy
import numpy as np
from numpy.random import RandomState as rng

#tensorflow
import tensorflow as tf
import tensorflow.contrib.keras as tfk

#pointcnn stuff
import pointcnn as pcnn

## ## HEP PCNN Model
#
## In[4]:
#
class Settings():
    
    def __init__(self, data_dim):
        self.sampling = "random"
        self.data_dim = data_dim
        self.with_global = False
        self.with_X_transformation = True
        self.sorting_method = 'cyxz'
        self.use_extra_features = False
        self.fc_params = []
        self.xconv_params = []
        
    def set_xconv_params(self, conv_params):
        self.xconv_params = conv_params
        
    def set_fc_params(self, fc_params):
        self.fc_params = fc_params
        

def build_pcnn_model(args):
    
    #some general parameters
    xconv_param_name = ('K', 'D', 'P', 'C', 'links')
    fc_param_name = ('C', 'dropout_rate')
    
    #datatype
    dtype=args["precision"]
    
    #find out which device to use:
    device='/cpu:0'
    if args['arch']=='gpu':
        device='/gpu:0'
    
    #define empty variables dict
    variables={}
    
    #create input tensors
    handle = tf.placeholder(tf.string, shape=[], name="iterator-placeholder")
    iterator = tf.data.Iterator.from_string_handle(handle, (tf.float32, tf.int32),
                                                            ((args["train_batch_size"], args['num_points'], 6), \
                                                            (args["train_batch_size"])))
    next_elem = iterator.get_next()
    variables['iterator_'] = iterator
    variables['iterator_handle_'] = handle
    variables['hits_'] = next_elem[0]
    variables['labels_'] = next_elem[1]
    variables['weights_'] = tf.constant(1., tf.float32, [args["train_batch_size"]])
    variables['keep_prob_'] = tf.placeholder(dtype)
    
    #build the calorimeter part
    points = variables['hits_'][...,:3]
    features = variables['hits_'][...,3:]
    
    x_fact = 3
    xconv_params = [dict(zip(xconv_param_name, xconv_param)) for xconv_param in
                    [(8, 1, -1, 16 * x_fact, []),
                     (12, 2, 384, 32 * x_fact, []),
                     (12, 2, 256, 48 * x_fact, []),
                     (16, 2, 128, 64 * x_fact, []),
                     (16, 3, 128, 128 * x_fact, [])]]
    fc_params = [dict(zip(fc_param_name, fc_param)) for fc_param in
                        [(128 * x_fact, 1.-variables['keep_prob_']),
                        (64 * x_fact, 1.-variables['keep_prob_'])]]
    #create settings class
    setting = Settings(data_dim = 6)
    setting.set_xconv_params(xconv_params)
    setting.set_fc_params(fc_params)
    with tf.variable_scope("point_cnn"):
        point_cnn = pcnn.PointCNN(points, features, True, setting);
    
    #reduce the mean of the output
    with tf.variable_scope("classifier"):
        features = tf.reduce_mean(point_cnn.layer_fts[-1], axis=1, keepdims=False)
        logits = tf.layers.dense(features, 2, activation=None, name="output")
    
    return variables, logits


#build the functions
def build_functions(args, variables, logits):
    
    #loss function
    prediction = tf.nn.softmax(logits)
    tf.add_to_collection('prediction_op', prediction)
    
    #compute loss, important: use unscaled version!
    weights = 1. #variables['weights_']
    loss = tf.losses.sparse_softmax_cross_entropy(variables['labels_'],
                                                  logits,
                                                  weights=weights)
    
    #compute accuracy
    accuracy = tf.metrics.accuracy(variables['labels_'],
                                   tf.round(prediction[:,1]),
                                   weights=variables['weights_'],
                                   name='accuracy')
    
    #compute AUC
    auc = tf.metrics.auc(variables['labels_'],
                         prediction[:,1],
                         weights=variables['weights_'],
                         num_thresholds=5000,
                         curve='ROC',
                         name='AUC')
    
    #return functions
    return variables, prediction, loss, accuracy, auc

