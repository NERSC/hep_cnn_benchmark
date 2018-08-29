
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


# # General Functions

# ## Input Handler

# In[3]:

class suppress_stdout_stderr(object):
    """
    A context manager for doing a "deep suppression" of stdout and stderr in
    Python, i.e. will suppress all print, even if the print originates in a
    compiled C/Fortran sub-function.
       This will not suppress raised exceptions, since exceptions are printed
    to stderr just before a script exits, and after the context manager has
    exited (at least, I think that is why it lets exceptions through).
    """
    def __init__(self):
        # Open a pair of null files
        self.null_fds =  [os.open(os.devnull,os.O_RDWR) for x in range(2)]
        # Save the actual stdout (1) and stderr (2) file descriptors.
        self.save_fds = (os.dup(1), os.dup(2))

    def __enter__(self):
        # Assign the null pointers to stdout and stderr.
        os.dup2(self.null_fds[0],1)
        os.dup2(self.null_fds[1],2)

    def __exit__(self, *_):
        # Re-assign the real stdout/stderr back to (1) and (2)
        os.dup2(self.save_fds[0],1)
        os.dup2(self.save_fds[1],2)
        # Close the null files
        os.close(self.null_fds[0])
        os.close(self.null_fds[1])


class DataSet(object):
    
    def reset(self):
        self._epochs_completed = 0
        self._file_index = 0
        self._data_index = 0
    
    def load_next_file(self):
        #only load a new file if there are more than one file in the list:
        if self._num_files > 1 or not self._initialized:
            try:
                with suppress_stdout_stderr():
                    self._tree = rnp.root2array(self._filelist[self._file_index], treename='Delphes',
                                          branches=self._branch_dict.keys(), stop=self._max_events,
                                          warn_missing_tree=True)
            except EnvironmentError:
                raise EnvironmentError("Cannot open file "+self._filelist[self._file_index])
            
            # Rename the branches
            self._tree.dtype.names = self._branch_dict.values()
            self._initialized = True
        
            ##set number of samples
            #self._num_examples = self._labels.shape[0]
            #
            ##reshape labels and weights
            #self._labels = np.expand_dims(self._labels, axis=1).astype(np.int32, copy=False)
            #self._normweights = np.expand_dims(self._normweights, axis=1)
            #self._weights = np.expand_dims(self._weights, axis=1)
            #self._psr = np.expand_dims(self._psr, axis=1)
            #
            ##transpose images if data format is NHWC
            #if self._data_format == "NHWC":
            #    #transform for NCHW to NHWC
            #    self._images = np.transpose(self._images, (0,2,3,1))
        
        print(dir(self._tree))
        
        #if self._shuffle:
        #  #create permutation
        #  perm = np.arange(self._num_examples)
        #  np.random.shuffle(perm)
        #  #shuffle
        #  self._images = self._images[perm]
        #  self._labels = self._labels[perm]
        #  self._normweights = self._normweights[perm]
        #  self._weights = self._weights[perm]
        #  self._psr = self._psr[perm]
        
    def __init__(self, filelist, num_tasks=1, taskid=0, split_filelist=False, shuffle=False, max_events=None):
        """Construct DataSet"""
        #general dict for extracting data
        self._branch_dict = {
            'Tower.Eta' : 'clusEta',
            'Tower.Phi' : 'clusPhi',
            'Tower.E' : 'clusE',
            'Tower.Eem' : 'clusEM',
            'Track.Eta' : 'trackEta',
            'Track.Phi' : 'trackPhi'
        }
        
        #multinode stuff
        self._num_tasks = num_tasks
        self._taskid = taskid
        self._split_filelist = split_filelist
        self._shuffle = shuffle
        self._max_events = max_events
        
        #split filelist?
        self._num_files = len(filelist)
        start = 0
        end = self._num_files
        if self._split_filelist:
            self._num_files = int(np.floor(len(filelist)/float(self._num_tasks)))
            start = self._taskid * self._num_files
            end = start + self._num_files
        
        assert self._num_files > 0, ('filelist is empty')
        
        self._filelist = filelist[start:end]
        self._initialized = False
        self.reset()
        self.load_next_file()

    @property
    def num_files(self):
        return self._num_files
    
    @property
    def num_samples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed
    
    #def next(self):
    #    for i in itertools.count(1): 
    #        try:
    #            images, labels, normweights, weights, psr = self.next_batch(1)
    #            
    #            #squeeze dims:
    #            images = np.squeeze(images, axis=0)
    #            labels = np.squeeze(labels, axis=0)
    #            normweights = np.squeeze(normweights, axis=0)
    #            weights = np.squeeze(weights, axis=0)
    #            psr = np.squeeze(psr, axis=0)
    #            
    #            yield images, labels, normweights, weights, psr
    #        except:
    #            return
    #
    #def next_batch(self, batch_size):
    #    """Return the next `batch_size` examples from this data set."""
    #    start = self._data_index
    #    self._data_index += batch_size
    #    end=int(np.min([self._num_examples,self._data_index]))
    #    
    #    #take what is there
    #    images = self._images[start:end]
    #    labels = self._labels[start:end]
    #    normweights = self._normweights[start:end]
    #    weights = self._weights[start:end]
    #    psr = self._psr[start:end]
    #    
    #    if self._data_index > self._num_examples:
    #        #remains:
    #        remaining = self._data_index-self._num_examples
    #        
    #        #first, reset data_index and increase file index:
    #        self._data_index=0
    #        self._file_index+=1
    #        
    #        #check if we are at the end of the file list
    #        if self._file_index >= self._num_files:
    #            #epoch is finished
    #            self._epochs_completed += 1
    #            #reset file index and shuffle list
    #            self._file_index=0
    #            if self._shuffle:
    #                np.random.shuffle(self._filelist)
    #            return
    #        
    #        #load the next file
    #        self.load_next_file()
    #        #assert batch_size <= self._num_examples
    #        #call rerucsively
    #        tmpimages,tmplabels,tmpnormweights,tmpweights,tmppsr = self.next_batch(remaining)
    #        #join
    #        images = np.concatenate([images,tmpimages],axis=0)    
    #        labels = np.concatenate([labels,tmplabels],axis=0)
    #        normweights = np.concatenate([normweights,tmpnormweights],axis=0)
    #        weights = np.concatenate([weights,tmpweights],axis=0)
    #        psr = np.concatenate([psr,tmppsr],axis=0)
    #    
    #    return images, labels, normweights, weights, psr


#load model wrapper
def load_model(sess, saver, checkpoint_dir):
    print("Looking for model in {}".format(checkpoint_dir))
    #get list of checkpoints
    checkpoints = [x.replace(".index","") for x in os.listdir(checkpoint_dir) if x.startswith("model.ckpt") and x.endswith(".index")]
    checkpoints = sorted([(int(x.split("-")[1]),x) for x in checkpoints], key=lambda tup: tup[0])
    latest_ckpt = os.path.join(checkpoint_dir,checkpoints[-1][1])
    print("Restoring model {}".format(latest_ckpt))
    try:
        saver.restore(sess, latest_ckpt)
        print("Model restoration successful.")
    except:
        print("Loading model failed, starting fresh.")


## ## HEP PCNN Model
#
## In[4]:
#
def build_pcnn_model(args):
    return None, None
#    
#    #datatype
#    dtype=args["precision"]
#    
#    #find out which device to use:
#    device='/cpu:0'
#    if args['arch']=='gpu':
#        device='/gpu:0'
#    
#    #define empty variables dict
#    variables={}
#    
#    #rotate input shape depending on data format
#    data_format=args['conv_params']['data_format']
#    input_shape = args['input_shape']
#    
#    #create graph handle
#    args['graph'] = tf.Graph()
#    
#    
#    #create input tensors
#    handle = tf.placeholder(tf.string, shape=[], name="iterator-placeholder")
#    iterator = tf.data.Iterator.from_string_handle(handle, (tf.float32, tf.int32, tf.float32, tf.float32, tf.float32),
#                                                            ((args['train_batch_size_per_node'], input_shape[0], input_shape[1], input_shape[2]),
#                                                            (args['train_batch_size_per_node'], 1),
#                                                            (args['train_batch_size_per_node'], 1),
#                                                            (args['train_batch_size_per_node'], 1),
#                                                            (args['train_batch_size_per_node'], 1)
#                                                            ))
#    next_elem = iterator.get_next()
#    variables['iterator_'] = iterator
#    variables['iterator_handle_'] = handle
#    variables['images_'] = next_elem[0]
#    variables['labels_'] = next_elem[1]
#    variables['normweights_'] = next_elem[2]
#    variables['weights_'] = next_elem[3]
#    variables['keep_prob_'] = tf.placeholder(dtype)
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

