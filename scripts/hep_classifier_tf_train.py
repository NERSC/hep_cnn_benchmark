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

#slurm helpers
import slurm_tf_helper.setup_clusters as sc

#housekeeping
import networks.binary_classifier_tf as bc

# Useful Functions

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="specify a config file in json format")
    parser.add_argument("--num_tasks", type=int, default=1, help="specify the number of tasks")
    parser.add_argument("--num_ps", type=int, default=0, help="specify the number of parameters servers")
    parser.add_argument("--precision", type=str, default="fp32", help="specify the precision. supported are fp32 and fp16")
    parser.add_argument('--dummy_data', action='store_const', const=True, default=False, 
                        help='use dummy data instead of real data')
    pargs = parser.parse_args()
    
    #load the json:
    with open(pargs.config,"r") as f:
        args = json.load(f)
    
    #set the rest
    args['num_tasks'] = pargs.num_tasks
    args['num_ps'] = pargs.num_ps
    args['dummy_data'] = pargs.dummy_data
    
    #modify the activations
    if args['conv_params']['activation'] == 'ReLU':
        args['conv_params']['activation'] = tf.nn.relu
    else:
        raise ValueError('Only ReLU is supported as activation')
        
    #modify the initializers
    if args['conv_params']['initializer'] == 'HE':
        args['conv_params']['initializer'] = tfk.initializers.he_normal()
    else:
        raise ValueError('Only ReLU is supported as initializer')
        
    #modify the optimizers
    args['opt_args'] = {"learning_rate": args['learning_rate']}
    if args['optimizer'] == 'ADAM':
        args['opt_func'] = tf.train.AdamOptimizer
    else:
        raise ValueError('Only ADAM is supported as optimizer')
    
    #now, see if all the paths are there
    args['logpath'] = args['outputpath']+'/logs'
    args['modelpath'] = args['outputpath']+'/models'
    
    if not os.path.isdir(args['logpath']):
        print("Creating log directory ",args['logpath'])
        os.makedirs(args['logpath'])
    if not os.path.isdir(args['modelpath']):
        print("Creating model directory ",args['modelpath'])
        os.makedirs(args['modelpath'])
    if not os.path.isdir(args['inputpath']) and not args['dummy_data']:
        raise ValueError("Please specify a valid path with input files in hdf5 format")
    
    #precision:
    args['precision'] = tf.float32
    if pargs.precision == "fp16":
        args['precision'] = tf.float16
    
    return args


def train_loop(sess, ops, args, feed_dict_train, feed_dict_validation):
    
    #restore weights belonging to graph
    epochs_completed = 0
    
    #losses
    train_loss=0.
    train_batches=0
    total_batches=0
    train_time=0
    
    #extract ops
    train_step = ops["train_step"]
    global_step = ops["global_step"]
    loss_eval = ops["loss_eval"]
    acc_update = ops["acc_update"]
    acc_eval = ops["acc_eval"]
    auc_update = ops["auc_update"]
    auc_eval = ops["auc_eval"]
    
    #do training
    while not sess.should_stop():
        
        while True:
            #increment total batch counter
            total_batches+=1
                    
            try:
                start_time = time.time()
                if args['create_summary']:
                    _, gstep, summary, tmp_loss = sess.run([train_step, global_step, train_summary, loss_eval], feed_dict=feed_dict_train)
                else:
                    _, gstep, tmp_loss = sess.run([train_step, global_step, loss_eval], feed_dict=feed_dict_train)        
        
                end_time = time.time()
                train_time += end_time-start_time
        
                #increment train loss and batch number
                train_loss += tmp_loss
                total_batches += 1
                train_batches += 1
        
                #determine if we give a short update:
                if gstep%args['display_interval']==0:
                    print(time.time(),"REPORT rank",args["task_index"],"global step %d., average training loss %g (%.3f sec/batch)"%(gstep,
                                                                                                            train_loss/float(train_batches),
                                                                                                            train_time/float(train_batches)))
        
            except:
                print(time.time(),"COMPLETED rank",args["task_index"],"epoch %d, average training loss %g (%.3f sec/batch)"%(epochs_completed, 
                                                                                     train_loss/float(train_batches),
                                                                                     train_time/float(train_batches)))
            
                #reset counters
                train_loss=0.
                train_batches=0
                train_time=0
            
                #compute validation loss:
                #reset variables
                validation_loss=0.
                validation_batches=0
            
                #iterate over batches
                while True:
                    
                    try:
                        start_time = time.time()
                        #compute loss
                        if args['create_summary']:
                            gstep, summary, tmp_loss, _, _ =sess.run([global_step, validation_summary, loss_eval, acc_update, auc_update],
                                                                feed_dict=feed_dict_validation)
                        else:
                            gstep, tmp_loss, _, _ = sess.run([global_step, loss_fn, acc_update, auc_update], feed_dict=feed_dict_validation)
                
                        #add loss
                        validation_loss += tmp_loss[0]
                        validation_batches += 1
                        
                    except:
                        print(time.time(),"COMPLETED epoch %d, average validation loss %g"%(epochs_completed, validation_loss/float(validation_batches)))
                        validation_accuracy = sess.run(acc_eval)
                        print(time.time(),"COMPLETED epoch %d, average validation accu %g"%(epochs_completed, validation_accuracy))
                        validation_auc = sess.run(auc_eval)
                        print(time.time(),"COMPLETED epoch %d, average validation auc %g"%(epochs_completed, validation_auc))


def main():
    # Parse Parameters
    args = parse_arguments()


    # Multi-Node Stuff
    
    #decide who will be worker and who will be parameters server
    if args['num_tasks'] > 1:
        args['cluster'], args['server'], args['task_index'], args['num_workers'], args['node_type'] = sc.setup_slurm_cluster(num_ps=args['num_ps'])    
        if args['node_type'] == "ps":
            args['server'].join()
        elif args['node_type'] == "worker":
            args['is_chief']=(args['task_index'] == 0)
        args['target']=args['server'].target
        if args['num_hot_spares']>=args['num_workers']:
            raise ValueError("The number of hot spares has be be smaller than the number of workers.")
    else:
        args['cluster']=None
        args['num_workers']=1
        args['server']=None
        args['task_index']=0
        args['node_type']='worker'
        args['is_chief']=True
        args['target']=''
        args['hot_spares']=0
        
    #general stuff
    if not args["batch_size_per_node"]:
        args["train_batch_size_per_node"]=int(args["train_batch_size"]/float(args["num_workers"]))
        args["validation_batch_size_per_node"]=int(args["validation_batch_size"]/float(args["num_workers"]))
    else:
        args["train_batch_size_per_node"]=args["train_batch_size"]
        args["validation_batch_size_per_node"]=args["validation_batch_size"]
    
    
    # On-Node Stuff
    
    if (args['node_type'] == 'worker'):
        #common stuff
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
    
        print("Rank",args['task_index'],": using ",num_inter_threads,"-way task parallelism with ",num_intra_threads,"-way data parallelism.")
    
    
    # Build Network and Functions
    
    if args['node_type'] == 'worker':
        print("Rank",args["task_index"],":","Building model") 
        args['device'] = tf.train.replica_device_setter(worker_device="/job:worker/task:%d" % args['task_index'],
                                                        cluster=args['cluster'])
            
        with tf.device(args['device']):
            variables, network = bc.build_cnn_model(args)
            variables, pred_fn, loss_fn, accuracy_fn, auc_fn = bc.build_functions(args,variables,network)
            #variables, pred_fn, loss_fn = bc.build_functions(args,variables,network)
            #tf.add_to_collection('pred_fn', pred_fn)
            #tf.add_to_collection('loss_fn', loss_fn)
            #tf.add_to_collection('accuracy_fn', accuracy_fn[0])
            print("Variables for rank",args["task_index"],":",variables)
            print("Network for rank",args["task_index"],":",network)
    
    
    # Setup Iterators
    
    if args['node_type'] == 'worker':
        print("Rank",args["task_index"],":","Setting up iterators")
        
        trainset=None
        validationset=None
        if not args['dummy_data']:
            #training files
            trainfiles = [args['inputpath']+'/'+x for x in os.listdir(args['inputpath']) if 'train' in x and (x.endswith('.h5') or x.endswith('.hdf5'))]
            trainset = bc.DataSet(trainfiles,args['num_workers'],args['task_index'],split_filelist=True,split_file=False,data_format=args["conv_params"]['data_format'],shuffle=True)
        
            #validation files
            validationfiles = [args['inputpath']+'/'+x for x in os.listdir(args['inputpath']) if 'val' in x and (x.endswith('.h5') or x.endswith('.hdf5'))]
            validationset = bc.DataSet(validationfiles,args['num_workers'],args['task_index'],split_filelist=True,split_file=False,data_format=args["conv_params"]['data_format'],shuffle=False)
            
        else:
            #training files and validation files are just dummy sets then
            trainset = bc.DummySet(input_shape=args['input_shape'], samples_per_epoch=10000, task_index=args['task_index'])
            validationset = bc.DummySet(input_shape=args['input_shape'], samples_per_epoch=1000, task_index=args['task_index'])
        
        #create tensorflow datasets
        #training
        dataset_train = tf.data.Dataset.from_generator(trainset.next, 
                                                      output_types = (tf.float32, tf.int32, tf.float32, tf.float32, tf.float32), 
                                                      output_shapes = (args['input_shape'], (1), (1), (1), (1)))
        dataset_train = dataset_train.prefetch(args['train_batch_size_per_node'])
        dataset_train = dataset_train.batch(args['train_batch_size_per_node'], drop_remainder=True)
        dataset_train = dataset_train.repeat()
        iterator_train = dataset_train.make_initializable_iterator()
        iterator_train_handle_string = iterator_train.string_handle()
        iterator_train_init_op = iterator_train.make_initializer(dataset_train)
        
        #validation
        dataset_validation = tf.data.Dataset.from_generator(validationset.next, 
                                                            output_types = (tf.float32, tf.int32, tf.float32, tf.float32, tf.float32), 
                                                            output_shapes = (args['input_shape'], (1), (1), (1), (1)))
        dataset_validation = dataset_validation.prefetch(args['validation_batch_size_per_node'])
        dataset_validation = dataset_validation.batch(args['validation_batch_size_per_node'], drop_remainder=True)
        dataset_validation = dataset_validation.repeat()
        iterator_validation = dataset_validation.make_initializable_iterator()
        iterator_validation_handle_string = iterator_validation.string_handle()
        iterator_validation_init_op = iterator_validation.make_initializer(dataset_validation)
        
        
    #Determine stopping point, i.e. compute last_step:
    args["last_step"] = int(args["trainsamples"] * args["num_epochs"] / (args["train_batch_size_per_node"] * args["num_workers"]))
    print("Stopping after %d global steps"%(args["last_step"]))
    
    
    # Train Model
    
    #determining which model to load:
    metafilelist = [args['modelpath']+'/'+x for x in os.listdir(args['modelpath']) if x.endswith('.meta')]
    if not metafilelist:
        #no model found, restart from scratch
        args['restart']=True
    
    
    #initialize session
    if (args['node_type'] == 'worker'):
        
        #use default graph
        with args['graph'].as_default():
        
            #a hook that will stop training at a certain number of steps
            hooks=[tf.train.StopAtStepHook(last_step=args["last_step"])]
        
            with tf.device(args['device']):
            
                #global step that either gets updated after any node processes a batch (async) or when all nodes process a batch for a given iteration (sync)
                global_step = tf.train.get_or_create_global_step()
                opt = args['opt_func'](**args['opt_args'])
            
                #decide whether we want to do sync or async training
                if args['mode'] == "sync" and args['num_tasks'] > 1:
                    print("Rank",args["task_index"],"performing synchronous updates")
                    #if sync we make a data structure that will aggregate the gradients form all tasks (one task per node in thsi case)
                    opt = tf.train.SyncReplicasOptimizer(opt, 
                                                         replicas_to_aggregate=args['num_workers']-args['num_hot_spares'],
                                                         total_num_replicas=args['num_workers'],
                                                         use_locking=True)
                    hooks.append(opt.make_session_run_hook(args['is_chief']))
                else:
                    print("Rank",args["task_index"],"performing asynchronous updates")
                
                #create train step handle
                train_step = opt.minimize(loss_fn, global_step=global_step)
                
                #creating summary
                if args['create_summary']:
                    #var_summary = []
                    #for item in variables:
                    #    var_summary.append(tf.summary.histogram(item,variables[item]))
                    summary_loss = tf.summary.scalar("loss",loss_fn)
                    train_summary = tf.summary.merge([summary_loss])
                    hooks.append(tf.train.StepCounterHook(every_n_steps=100,output_dir=args['logpath']))
                    hooks.append(tf.train.SummarySaverHook(save_steps=100,output_dir=args['logpath'],summary_op=train_summary))
                
                # Add an op to initialize the variables.
                init_global_op = tf.global_variables_initializer()
                init_local_op = tf.local_variables_initializer()
            
                #saver class:
                model_saver = tf.train.Saver()
            
            
                print("Rank",args["task_index"],": starting training using "+args['optimizer']+" optimizer")
                with tf.train.MonitoredTrainingSession(config=sess_config, 
                                                       is_chief=args["is_chief"],
                                                       master=args['target'],
                                                       checkpoint_dir=args['modelpath'],
                                                       save_checkpoint_secs=300,
                                                       hooks=hooks) as sess:
        
                    #initialize variables
                    sess.run([init_global_op, init_local_op])
                    
                    #init iterator handle
                    iterator_train_handle, iterator_validation_handle = sess.run([iterator_train_handle_string, iterator_validation_handle_string])
                    #init iterators
                    sess.run(iterator_train_init_op, feed_dict={variable["iterator_handle_"]: iterator_train_handle})
                    sess.run(iterator_validation_init_op, feed_dict={variable["iterator_handle_"]: iterator_validation_handle})
                    
                    #restore weights belonging to graph
                    if not args['restart']:
                        last_model = tf.train.latest_checkpoint(args['modelpath'])
                        print("Restoring model %s.",last_model)
                        model_saver.restore(sess,last_model)
                    
                    #feed dicts
                    feed_dict_train={variables['iterator_handle_']: iterator_train_handle, variables['keep_prob_']: args['dropout_p']}
                    feed_dict_validation={variables['iterator_handle_']: iterator_validation_handle, variables['keep_prob_']: 1.0}
                    
                    #ops dict
                    ops = {"train_step" : train_step,
                            "loss_eval": loss_avg_fn,
                            "global_step": global_step,
                            "acc_update": accuracy_fn[1],
                            "acc_eval": accuracy_avg_fn,
                            "auc_update": auc_fn[1],
                            "auc_eval": auc_avg_fn
                            }
                
                    #determine if we need a summary
                    if args['create_summary'] and args["is_chief"]:
                        ops["train_summary"] = train_summary
                    else:
                        ops["train_summary"] = None
                    
                    #do the training loop
                    total_time = time.time()
                    train_loop(sess, ops, args, feed_dict_train, feed_dict_validation)
                    total_time -= time.time()
                    print("FINISHED Training. Total time %g"%(total_time))


#main
if "__main__" in __name__:
    main()