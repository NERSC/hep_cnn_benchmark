
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
    iterator = tf.data.Iterator.from_string_handle(handle, (tf.float32, tf.float32, tf.int32),
                                                            ((args["train_batch_size"], args['num_calorimeter_hits'], 5), \
                                                            (args["train_batch_size"], args['num_tracks'], 3), \
                                                            (args["train_batch_size"])))
    next_elem = iterator.get_next()
    variables['iterator_'] = iterator
    variables['iterator_handle_'] = handle
    variables['calorimeter_hits_'] = next_elem[0]
    variables['tracks_'] = next_elem[1]
    variables['labels_'] = next_elem[2]
    variables['keep_prob_'] = tf.placeholder(dtype)
    
    #build the calorimeter part
    calo_points = variables['calorimeter_hits_'][...,:3]
    calo_features = variables['calorimeter_hits_'][...,3:]
    
    x_fact_conv = 3
    calo_xconv_params = [dict(zip(xconv_param_name, xconv_param)) for xconv_param in
                        [(6, 1, -1, 16 * x_fact_conv, []),
                        (8, 2, 384, 32 * x_fact_conv, []),
                        (12, 2, 256, 64 * x_fact_conv, []),
                        (12, 3, 128, 64 * x_fact_conv, [])]]
    #calo_fc_params = [dict(zip(fc_param_name, fc_param)) for fc_param in
    #                    [(128 * x_fact, 0.0),
    #                      (64 * x_fact, 1.-variables['keep_prob_'])]]
    #create settings class
    calo_setting = Settings(data_dim = 5)
    calo_setting.set_xconv_params(calo_xconv_params)
    #calo_setting.set_fc_params(calo_fc_params)
    with tf.variable_scope("calo_cnn"):
        calo_cnn = pcnn.PointCNN(calo_points, calo_features, True, calo_setting);
    
    
    #build the track part
    track_points = variables['tracks_']
    track_features = None
    
    x_fact_conv = 3
    track_xconv_params = [dict(zip(xconv_param_name, xconv_param)) for xconv_param in
                        [(4, 1, -1, 8 * x_fact_conv, []),
                        (6, 2, 256, 16 * x_fact_conv, []),
                        (8, 2, 192, 32 * x_fact_conv, []),
                        (8, 3, 128, 64 * x_fact_conv, [])]]
    #x_fact_fc = 3
    #track_fc_params = [dict(zip(fc_param_name, fc_param)) for fc_param in
    #                    [(64 * x_fact_fc, 0.0),
    #                      (64 * x_fact_fc, 1.-variables['keep_prob_'])]]
    #create settings class
    track_setting = Settings(data_dim = 3)
    track_setting.set_xconv_params(track_xconv_params)
    #track_setting.set_fc_params(track_fc_params)
    with tf.variable_scope("track_cnn"):
        track_cnn = pcnn.PointCNN(track_points, track_features, True, track_setting);
    
    
    print(calo_cnn.layer_fts[-1].shape, track_cnn.layer_fts[-1].shape)
    
    #build the combined part
    combined_points = tf.concat([calo_cnn.layer_pts[-1], track_cnn.layer_pts[-1]], axis=1)
    combined_features = tf.concat([calo_cnn.layer_fts[-1], track_cnn.layer_fts[-1]], axis=1)
    
    x_fact_conv = 3
    combined_xconv_params = [dict(zip(xconv_param_name, xconv_param)) for xconv_param in
                        [(12, 2, -1, 64 * x_fact_conv, []),
                        (14, 3, 192, 128 * x_fact_conv, []),
                        (14, 3, 128, 256 * x_fact_conv, [])]]
    combined_fc_params = [dict(zip(fc_param_name, fc_param)) for fc_param in
                           [(2, 1.-variables['keep_prob_'])]]
    #create settings class
    combined_setting = Settings(data_dim = 3)
    combined_setting.set_xconv_params(combined_xconv_params)
    combined_setting.set_fc_params(combined_fc_params)
    with tf.variable_scope("combined_cnn"):
        combined_cnn = pcnn.PointCNN(combined_points, combined_features, True, combined_setting);
    
    
    
    return variables, calo_cnn

#    
#    #rotate input shape depending on data format
#    data_format=args['conv_params']['data_format']
#    input_shape = args['input_shape']
#    
#    #create graph handle
#    args['graph'] = tf.Graph()
#    
#    

#    
#    #empty network:
#    network = []
#    
#    #input layer
#    network.append(variables['images_'])
#    
#    #get all the conv-args stuff:
#    activation=args['conv_params']['activation']
#    initializer=args['conv_params']['initializer']
#    ksize=args['conv_params']['filter_size']
#    num_filters=args['conv_params']['num_filters']
#    padding=str(args['conv_params']['padding'])
#        
#    #conv layers:
#    prev_num_filters=args['input_shape'][0]
#    if data_format=="NHWC":
#        prev_num_filters=args['input_shape'][2]
#    
#    for layerid in range(1,args['num_layers']+1):
#    
#        #create weight-variable
#        #with tf.device(device):
#        variables['conv'+str(layerid)+'_w']=tf.Variable(initializer([ksize,ksize,prev_num_filters,num_filters],dtype=dtype),
#                                                        name='conv'+str(layerid)+'_w',dtype=dtype)
#        prev_num_filters=num_filters
#    
#        #conv unit
#        network.append(tf.nn.conv2d(network[-1],
#                                    filter=variables['conv'+str(layerid)+'_w'],
#                                    strides=[1, 1, 1, 1], 
#                                    padding=padding,
#                                    data_format=data_format,
#                                    name='conv'+str(layerid)))
#        
#        #batchnorm if desired
#        outshape=network[-1].shape[1:]
#        if args['batch_norm']:
#            #add batchnorm
#            #with tf.device(device):
#            #mu
#            variables['bn'+str(layerid)+'_m']=tf.Variable(tf.zeros(outshape,dtype=dtype),
#                                                          name='bn'+str(layerid)+'_m',dtype=dtype)
#            #sigma
#            variables['bn'+str(layerid)+'_s']=tf.Variable(tf.ones(outshape,dtype=dtype),
#                                                          name='bn'+str(layerid)+'_s',dtype=dtype)
#            #gamma
#            variables['bn'+str(layerid)+'_g']=tf.Variable(tf.ones(outshape,dtype=dtype),
#                                                          name='bn'+str(layerid)+'_g',dtype=dtype)
#            #beta
#            variables['bn'+str(layerid)+'_b']=tf.Variable(tf.zeros(outshape,dtype=dtype),
#                                                          name='bn'+str(layerid)+'_b',dtype=dtype)
#            #add batch norm layer
#            network.append(tf.nn.batch_normalization(network[-1],
#                            mean=variables['bn'+str(layerid)+'_m'],
#                            variance=variables['bn'+str(layerid)+'_s'],
#                            offset=variables['bn'+str(layerid)+'_b'],
#                            scale=variables['bn'+str(layerid)+'_g'],
#                            variance_epsilon=1.e-4,
#                            name='bn'+str(layerid)))
#        else:
#            bshape = (variables['conv'+str(layerid)+'_w'].shape[3])
#            variables['conv'+str(layerid)+'_b']=tf.Variable(tf.zeros(bshape,dtype=dtype),
#                                                            name='conv'+str(layerid)+'_b',dtype=dtype)
#            #add bias
#            if dtype!=tf.float16:
#                network.append(tf.nn.bias_add(network[-1],variables['conv'+str(layerid)+'_b'],data_format=data_format))
#            else:
#                print("Warning: bias-add currently not supported for fp16!")
#        #add relu unit
#        #with tf.device(device):
#        network.append(activation(network[-1]))
#        
#        #add maxpool
#        #with tf.device(device):
#        kshape=[1,1,2,2]
#        sshape=[1,1,2,2]
#        if data_format=="NHWC":
#            kshape=[1,2,2,1]
#            sshape=[1,2,2,1]
#        network.append(tf.nn.max_pool(network[-1],
#                                      ksize=kshape,
#                                      strides=sshape,
#                                      padding=args['conv_params']['padding'],
#                                      data_format=data_format,
#                                      name='maxpool'+str(layerid)))
#        
#        #add dropout
#        #with tf.device(device):
#        network.append(tf.nn.dropout(network[-1],
#                                     keep_prob=variables['keep_prob_'],
#                                     name='drop'+str(layerid)))
#    
#    if args['scaling_improvements']:
#        #add another conv layer with average pooling to the mix
#        #with tf.device(device):
#        variables['conv'+str(layerid+1)+'_w']=tf.Variable(initializer([ksize,ksize,prev_num_filters,num_filters],dtype=dtype),
#                                                          name='conv'+str(layerid+1)+'_w',dtype=dtype)
#        prev_num_filters=num_filters
#    
#        #conv unit
#        network.append(tf.nn.conv2d(network[-1],
#                                    filter=variables['conv'+str(layerid+1)+'_w'],
#                                    strides=[1, 1, 1, 1], 
#                                    padding=padding,
#                                    data_format=data_format,
#                                    name='conv'+str(layerid+1)))
#            
#        #bias
#        bshape = (variables['conv'+str(layerid+1)+'_w'].shape[3])
#        variables['conv'+str(layerid+1)+'_b']=tf.Variable(tf.zeros(bshape,dtype=dtype), name='conv'+str(layerid+1)+'_b',dtype=dtype)
#        #add bias
#        if dtype!=tf.float16:
#            network.append(tf.nn.bias_add(network[-1],variables['conv'+str(layerid+1)+'_b'],data_format=data_format))
#        else:
#            print("Warning: bias-add currently snot supported for fp16!")
#            
#        #add relu unit
#        #with tf.device(device):
#        network.append(activation(network[-1]))
#        
#        #add average-pool
#        #with tf.device(device):
#        #pool over everything
#        imsize = network[-1].shape[2]
#        kshape = [1,1,imsize,imsize]
#        sshape = [1,1,imsize,imsize]
#        if data_format == "NHWC":
#            kshape = [1,imsize,imsize,1]
#            sshape = [1,imsize,imsize,1]
#        network.append(tf.nn.avg_pool(network[-1],
#                                      ksize=kshape,
#                                      strides=sshape,
#                                      padding=args['conv_params']['padding'],
#                                      data_format=data_format,
#                                      name='avgpool1'))
#    
#    #reshape
#    outsize = np.prod(network[-1].shape[1:]).value
#    #with tf.device(device):
#    network.append(tf.reshape(network[-1],shape=[-1, outsize],name='flatten'))
#    
#    if not args['scaling_improvements']:
#        #now do the MLP
#        #fc1
#        #with tf.device(device):
#        variables['fc1_w']=tf.Variable(initializer([outsize, args['num_fc_units']],dtype=dtype),name='fc1_w',dtype=dtype)
#        variables['fc1_b']=tf.Variable(tf.zeros([args['num_fc_units']],dtype=dtype),name='fc1_b',dtype=dtype)
#        network.append(tf.matmul(network[-1], variables['fc1_w']) + variables['fc1_b'])
#    
#        #add relu unit
#        #with tf.device(device):
#        network.append(activation(network[-1]))
#    
#        #add dropout
#        #with tf.device(device):
#        network.append(tf.nn.dropout(network[-1],
#                                     keep_prob=variables['keep_prob_'],
#                                     name='drop'+str(layerid)))
#        #fc2
#        #with tf.device(device):
#        variables['fc2_w']=tf.Variable(initializer([args['num_fc_units'],2],dtype=dtype),name='fc2_w',dtype=dtype)
#        variables['fc2_b']=tf.Variable(tf.zeros([2],dtype=dtype),name='fc2_b',dtype=dtype)
#        network.append(tf.matmul(network[-1], variables['fc2_w']) + variables['fc2_b'])
#            
#    else:
#        #only one FC layer here
#        #with tf.device(device):
#        variables['fc1_w']=tf.Variable(initializer([outsize,2],dtype=dtype),name='fc1_w',dtype=dtype)
#        variables['fc1_b']=tf.Variable(tf.zeros([2],dtype=dtype),name='fc1_b',dtype=dtype)
#        network.append(tf.matmul(network[-1], variables['fc1_w']) + variables['fc1_b'])
#                
#    #add softmax
#    #with tf.device(device):
#    network.append(tf.nn.softmax(network[-1]))
#    
#    #return the network and variables
#    return variables,network
#
#
## # Build Functions from the Network Output
#
## In[ ]:
#
##build the functions
#def build_functions(args,variables,network):
#    
#    #loss function
#    prediction = network[-1]
#    tf.add_to_collection('prediction_op', prediction)
#    
#    #compute loss, important: use unscaled version!
#    weights = 1. #variables['weights_']
#    loss = tf.losses.sparse_softmax_cross_entropy(variables['labels_'],
#                                                  network[-2],
#                                                  weights=weights)
#    
#    #compute accuracy
#    accuracy = tf.metrics.accuracy(variables['labels_'],
#                                   tf.round(prediction[:,1]),
#                                   weights=variables['weights_'],
#                                   name='accuracy')
#    
#    #compute AUC
#    auc = tf.metrics.auc(variables['labels_'],
#                         prediction[:,1],
#                         weights=variables['weights_'],
#                         num_thresholds=5000,
#                         curve='ROC',
#                         name='AUC')
#    
#    #return functions
#    return variables, prediction, loss, accuracy, auc

