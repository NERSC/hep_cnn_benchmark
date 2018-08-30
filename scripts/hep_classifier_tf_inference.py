#*** License Agreement ***
#
# High Energy Physics Deep Learning Convolutional Neural Network Benchmark
# (HEPCNNB) Copyright (c) 2017, The Regents of the University of California,
# through Lawrence Berkeley National Laboratory (subject to receipt of any
# required approvals from the U.S. Dept. of Energy). All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# (1) Redistributions of source code must retain the above copyright notice,
#     this list of conditions and the following disclaimer.
# (2) Redistributions in binary form must reproduce the above copyright notice,
#     this list of conditions and the following disclaimer in the documentation
#     and/or other materials provided with the distribution.
# (3) Neither the name of the University of California, Lawrence Berkeley
#     National Laboratory, U.S. Dept. of Energy nor the names of its
#     contributors may be used to endorse or promote products derived from this
#     software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# You are under no obligation whatsoever to provide any bug fixes, patches, or
# upgrades to the features, functionality or performance of the source code
# ("Enhancements") to anyone; however, if you choose to make your Enhancements
# available either publicly, or directly to Lawrence Berkeley National
# Laboratory, without imposing a separate written license agreement for such
# Enhancements, then you hereby grant the following license: a non-exclusive,
# royalty-free perpetual license to install, use, modify, prepare derivative
# works, incorporate into other computer software, distribute, and sublicense
# such enhancements or derivative works thereof, in binary and source code form.
#---------------------------------------------------------------      

# Compatibility
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

#os stuff
import os
import sys
import h5py as h5
import re
import json

#argument parsing
import argparse

#timing
import time

#numpy
import numpy as np

#tensorflow
import tensorflow as tf
import tensorflow.contrib.keras as tfk

#housekeeping
import networks.binary_classifier_tf as bc

#sklearn stuff
from sklearn import metrics

#matplotlib
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt


# Useful Functions
def plot_roc_curve(predictions, labels, weights, psr, outputdir):
    fpr, tpr, _ = metrics.roc_curve(labels, predictions, pos_label=1, sample_weight=weights)
    fpr_cut, tpr_cut, _ = metrics.roc_curve(labels, psr, pos_label=1, sample_weight=weights)
        
    #plot the data
    plt.figure()
    lw = 2
    #full curve
    plt.plot(fpr, tpr, lw=lw, linestyle="-", label='ROC curve (area = {:0.2f})'.format(metrics.auc(fpr, tpr, reorder=True)))
    
    plt.scatter([fpr_cut],[tpr_cut], label='standard cuts')
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(outputdir,'ROC_1400_850.png'),dpi=300)

    #zoomed-in
    plt.xlim([0.0, 0.0004])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(outputdir,'ROC_1400_850_zoom.png'),dpi=300)
    

def evaluate_loop(sess, ops, args, iterator_test_init_op, feed_dict_test, prefix):
    
    #reinit the test iterator
    sess.run(iterator_test_init_op, feed_dict=feed_dict_test)
    
    #compute test loss:
    #reset variables
    test_loss = 0.
    test_batches = 0
    
    #initialize lists
    predlist = []
    labellist = []
    weightlist = []
    psrlist = []
    
    #iterate over test set
    while test_batches < args["test_max_steps"]:
        
        try:
            #compute loss
            pred, label, weight, psr, tmp_loss, _, _ = sess.run([ops["prediction_eval"], \
                                                                     ops["label_eval"], \
                                                                     ops["weight_eval"], \
                                                                     ops["psr_eval"], \
                                                                     ops["loss_eval"], 
                                                                     ops["acc_update"], 
                                                                     ops["auc_update"]], 
                                                                     feed_dict=feed_dict_test)
            
            predlist.append(pred[:,1])
            labellist.append(label)
            weightlist.append(weight)
            psrlist.append(psr)
    
            #add loss
            test_loss += tmp_loss
            test_batches += 1
            
        except:
            break
    
    #report the results
    test_accuracy, test_auc = sess.run([ops["acc_eval"], ops["auc_eval"]])
    tstamp = time.time()
    print("%.2f EVALUATION %s: average loss %.6f"%(tstamp, prefix, test_loss/float(test_batches)))
    print("%.2f EVALUATION %s: average accu %.6f"%(tstamp, prefix, test_accuracy))
    print("%.2f EVALUATION %s: average auc %.6f"%(tstamp, prefix, test_auc))
    
    #do the ROC curve
    preds = np.concatenate(predlist, axis=0)
    labels = np.concatenate(labellist, axis=0)
    weights = np.concatenate(weightlist, axis=0)
    psrs = np.concatenate(psrlist, axis=0)
    plot_roc_curve(preds, labels, weights, psrs, args["plotpath"])


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="specify a config file in json format")
    parser.add_argument("--checkpoint_id", type=int, default=-1, help="select which checkpoint to load")
    pargs = parser.parse_args()
    
    #load the json:
    with open(pargs.config,"r") as f:
        args = json.load(f)

    args["checkpoint_id"] = pargs.checkpoint_id
    
    #modify the activations
    if args['conv_params']['activation'] == 'ReLU':
        args['conv_params']['activation'] = tf.nn.relu
    else:
        raise ValueError('Only ReLU is supported as activation')
        
    #modify the initializers
    if args['conv_params']['initializer'] == 'HE':
        args['conv_params']['initializer'] = tfk.initializers.he_normal()
    else:
        raise ValueError('Only HE is supported as initializer')
    
    #now, see if all the paths are there
    args['logpath'] = args['outputpath']+'/logs'
    args['modelpath'] = args['outputpath']+'/models'
    args['plotpath'] = args['outputpath']+'/plots'
    
    if not os.path.isdir(args['logpath']):
        print("Creating log directory ",args['logpath'])
        os.makedirs(args['logpath'])
    if not os.path.isdir(args['modelpath']):
        print("Creating model directory ",args['modelpath'])
        os.makedirs(args['modelpath'])
    if not os.path.isdir(args['inputpath']) and not args['dummy_data']:
        raise ValueError("Please specify a valid path with input files in hdf5 format")
    if not os.path.isdir(args['plotpath']):
        print("Creating plot directory ",args['plotpath'])
        os.makedirs(args['plotpath'])
    
    #precision:
    args['precision'] = tf.float32
    
    return args


def main():
    # Parse Parameters
    args = parse_arguments()
        
    #general stuff
    args["test_batch_size_per_node"]=int(args["test_batch_size"])
    args['train_batch_size_per_node']=args["test_batch_size_per_node"]
    
    #check how many validation steps we will do
    if "test_max_steps" not in args or args["test_max_steps"] <= 0:
        args["test_max_steps"] = np.inf
    
    # On-Node Stuff
    os.environ["KMP_BLOCKTIME"] = "1"
    os.environ["KMP_SETTINGS"] = "1"
    os.environ["KMP_AFFINITY"]= "granularity=fine,compact,1,0"
    
    #arch-specific stuff
    if args['arch']=='hsw':
        num_inter_threads = 2
        num_intra_threads = 16
    elif args['arch']=='knl':
        num_inter_threads = 2
        num_intra_threads = 33
    elif args['arch']=='gpu':
        #use default settings
        p = tf.ConfigProto()
        num_inter_threads = int(getattr(p,'INTER_OP_PARALLELISM_THREADS_FIELD_NUMBER'))
        num_intra_threads = int(getattr(p,'INTRA_OP_PARALLELISM_THREADS_FIELD_NUMBER'))
    else:
        raise ValueError('Please specify a valid architecture with arch (allowed values: hsw, knl, gpu)')
    
    #set the rest
    os.environ['OMP_NUM_THREADS'] = str(num_intra_threads)
    sess_config=tf.ConfigProto(inter_op_parallelism_threads=num_inter_threads,
                               intra_op_parallelism_threads=num_intra_threads,
                               log_device_placement=False,
                               allow_soft_placement=True)
    
    print("Using ",num_inter_threads,"-way task parallelism with ",num_intra_threads,"-way data parallelism.")
    
    
    # Build Network and Functions
    print("Building model") 
            
    variables, network = bc.build_cnn_model(args)
    variables, pred_fn, loss_fn, accuracy_fn, auc_fn = bc.build_functions(args,variables,network)

    print("Variables:",variables)
    print("Network:",network)    
    
    # Setup Iterators
    print("Setting up iterators")
        
    #test files
    testfiles = [args['inputpath']+'/'+x for x in os.listdir(args['inputpath']) if 'test' in x and (x.endswith('.h5') or x.endswith('.hdf5'))]
    testset = bc.DataSet(testfiles, 1,0, split_filelist=False, split_file=False, data_format=args["conv_params"]['data_format'], shuffle=False)
        
    #create tensorflow datasets
    #test
    dataset_test = tf.data.Dataset.from_generator(testset.next, 
                                                        output_types = (tf.float32, tf.int32, tf.float32, tf.float32, tf.float32), 
                                                        output_shapes = (args['input_shape'], (1), (1), (1), (1)))
    dataset_test = dataset_test.prefetch(args['test_batch_size_per_node'])
    dataset_test = dataset_test.apply(tf.contrib.data.batch_and_drop_remainder(args['test_batch_size_per_node']))
    dataset_test = dataset_test.repeat(1)
    #do some weight-preprocessing
    #dataset_test = dataset_test.map(lambda im,lb,wg,nw,ps: (im, lb, wg, nw, ps), num_parallel_calls=2)
    iterator_test = dataset_test.make_initializable_iterator()
    iterator_test_handle_string = iterator_test.string_handle()
    iterator_test_init_op = iterator_test.make_initializer(dataset_test) 
            
    # Add an op to initialize the variables.
    init_global_op = tf.global_variables_initializer()
    init_local_op = tf.local_variables_initializer()
            
    #saver class:
    model_saver = tf.train.Saver()            
            
    with tf.Session(config=sess_config) as sess:
        
        #initialize variables
        sess.run([init_global_op, init_local_op])
                    
        #init iterator handle
        iterator_test_handle = sess.run(iterator_test_handle_string)
                    
        #restore weights belonging to graph
        bc.load_model(sess, model_saver, args['modelpath'])
                    
        #feed dicts
        feed_dict_test={variables['iterator_handle_']: iterator_test_handle, variables['keep_prob_']: 1.}
                    
        #ops dict
        ops = {"loss_eval": loss_fn,
               "acc_update": accuracy_fn[1],
               "acc_eval": accuracy_fn[0],
               "auc_update": auc_fn[1],
               "auc_eval": auc_fn[0],
               "prediction_eval": network[-1],
               "label_eval": variables['labels_'],
               "weight_eval": variables['weights_'],
               "psr_eval": variables['psr_']
            }
        
        #do the evaluation loop
        evaluate_loop(sess, ops, args, iterator_test_init_op, feed_dict_test, "SUMMARY")

#main
if "__main__" in __name__:
    main()
