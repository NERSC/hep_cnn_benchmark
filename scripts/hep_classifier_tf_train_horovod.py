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
import horovod.tensorflow as hvd

#housekeeping
import networks.utils as utils
import networks.cnn.binary_classifier_tf as bc

#debugging
#tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)


# Initialize Horovod
hvd.init()

# Useful Functions

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="specify a config file in json format")
    parser.add_argument("--num_tasks", type=int, default=1, help="specify the number of tasks")
    parser.add_argument("--precision", type=str, default="fp32", help="specify the precision. supported are fp32 and fp16")
    parser.add_argument('--dummy_data', action='store_const', const=True, default=False, 
                        help='use dummy data instead of real data')
    pargs = parser.parse_args()
    
    #load the json:
    with open(pargs.config,"r") as f:
        args = json.load(f)
    
    #set the rest
    args['num_tasks'] = pargs.num_tasks
    args['num_ps'] = 0
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
    
    #precision:
    args['precision'] = tf.float32
    if pargs.precision == "fp16":
        args['precision'] = tf.float16
    
    return args

def evaluate_loop(sess, ops, args, iterator_validation_init_op, feed_dict_validation, prefix):
    
    #reinit the validation iterator
    sess.run(iterator_validation_init_op, feed_dict=feed_dict_validation)
    
    #compute validation loss:
    #reset variables
    validation_loss = 0.
    validation_batches = 0
    
    #get global step:
    gstep = sess.run(ops["global_step"])
    
    #iterate over validation set
    while validation_batches < args["validation_max_steps"]:
        
        try:
            #compute loss
            if args['create_summary']:
                summary, tmp_loss, _, _ = sess.run([ops["validation_summary"], ops["loss_eval"], ops["acc_update"], ops["auc_update"]],
                                                    feed_dict=feed_dict_validation)
            else:
                tmp_loss, _, _ = sess.run([ops["loss_eval"], ops["acc_update"], ops["auc_update"]], feed_dict=feed_dict_validation)
    
            #add loss
            validation_loss += tmp_loss
            validation_batches += 1
            
        except:
            break
    
    #report the results
    validation_accuracy, validation_auc = sess.run([ops["acc_eval"], ops["auc_eval"]])
    if args["is_chief"]:
        tstamp = time.time()
        print("%.2f EVALUATION %s: step %d (%d), average loss %.6f"%(tstamp, prefix, gstep, args["last_step"], validation_loss/float(validation_batches)))
        print("%.2f EVALUATION %s: step %d (%d), average accu %.6f"%(tstamp, prefix, gstep, args["last_step"], validation_accuracy))
        print("%.2f EVALUATION %s: step %d (%d), average auc %.6f"%(tstamp, prefix, gstep, args["last_step"], validation_auc))


def train_loop(sess, ops, args, iterator_train_init_op, feed_dict_train, iterator_validation_init_op, feed_dict_validation):
    
    #counters
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
    
    #init iterators
    sess.run(iterator_train_init_op, feed_dict=feed_dict_train)
    
    #do training
    while not sess.should_stop():
        
        #increment total batch counter
        total_batches+=1
                
        try:
            start_time = time.time()
            if args['create_summary']:
                _, gstep, summary, tmp_loss = sess.run([train_step, global_step, ops["train_summary"], loss_eval], feed_dict=feed_dict_train)
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
                if args["is_chief"]:
                    tstamp = time.time()
                    print("%.2f TRAINING REPORT: step %d (%d), average loss %.6f (%.3f sec/batch)"%(tstamp, gstep, args["last_step"],
                                                                                                        train_loss/float(train_batches),
                                                                                                        train_time/float(train_batches)))
                train_batches = 0
                train_loss = 0.
                train_time = 0.
                
            if gstep%args['validation_interval']==0:
                evaluate_loop(sess, ops, args, iterator_validation_init_op, feed_dict_validation, "REPORT")
    
        except:
            #get global step:
            gstep = sess.run(global_step)

            #reinit iterator for next round
            sess.run(iterator_train_init_op, feed_dict=feed_dict_train)
            
            #reset counters
            train_loss = 0.
            train_batches = 0
            train_time = 0.
            
            #run eval loop:
            evaluate_loop(sess, ops, args, iterator_validation_init_op, feed_dict_validation, "EPOCH SUMMARY")


def main():

    # Parse Parameters
    args = parse_arguments()
    
    # Multi-Node Stuff
    #decide who will be worker and who will be parameters server
    args['num_workers']=hvd.size()
    args['task_index']=hvd.rank()
    args["is_chief"]=True if args['task_index']==0 else False
    
    #general stuff
    if not args["batch_size_per_node"]:
        args["train_batch_size_per_node"]=int(args["train_batch_size"]/float(args["num_workers"]))
        args["validation_batch_size_per_node"]=int(args["validation_batch_size"]/float(args["num_workers"]))
    else:
        args["train_batch_size_per_node"]=args["train_batch_size"]
        args["validation_batch_size_per_node"]=args["validation_batch_size"]
    
    #check how many validation steps we will do
    if "validation_max_steps" not in args or args["validation_max_steps"] <= 0:
        args["validation_max_steps"] = np.inf
    
    # On-Node Stuff
    #common stuff
    os.environ["KMP_BLOCKTIME"] = "1"
    os.environ["KMP_SETTINGS"] = "1"
    os.environ["KMP_AFFINITY"]= "noverbose,granularity=fine,compact,1,0"
    
    #arch-specific stuff
    if args['arch']=='hsw':
        num_inter_threads = 2
        num_intra_threads = 16
    elif args['arch']=='ivb':
        num_inter_threads = 2
        num_intra_threads = 12
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
    sess_config.gpu_options.visible_device_list = str(hvd.local_rank())
    if args["is_chief"]:
        print("Using ",num_inter_threads,"-way task parallelism with ",num_intra_threads,"-way data parallelism.")
    
    
    # Build Network and Functions
    if args["is_chief"]:
        print("Building model")
    variables, network = bc.build_cnn_model(args)
    variables, pred_fn, loss_fn, accuracy_fn, auc_fn = bc.build_functions(args,variables,network)
    #rank averages
    loss_avg_fn = hvd.allreduce(tf.cast(loss_fn, tf.float32))
    accuracy_avg_fn = hvd.allreduce(tf.cast(accuracy_fn[0], tf.float32))
    auc_avg_fn = hvd.allreduce(tf.cast(auc_fn[0], tf.float32))
    if args["is_chief"]:
        print("Variables:",variables)
        print("Network:",network)
    
    # Setup Iterators
    if args["is_chief"]:
        print("Setting up iterators")
        
    #training files
    trainfiles = [args['inputpath']+'/'+x for x in os.listdir(args['inputpath']) if 'train' in x and (x.endswith('.h5') or x.endswith('.hdf5'))]
        
    #validation files
    validationfiles = [args['inputpath']+'/'+x for x in os.listdir(args['inputpath']) if 'val' in x and (x.endswith('.h5') or x.endswith('.hdf5'))]
    
    #create tensorflow datasets
    #training
    h5_train_gen = utils.hdf5_generator(shuffle=True, data_format=args["conv_params"]['data_format'])
    dataset_train = tf.data.Dataset.from_tensor_slices(trainfiles)
    dataset_train = dataset_train.shuffle(len(trainfiles))
    dataset_train = dataset_train.interleave(lambda filename: tf.data.Dataset.from_generator(h5_train_gen, \
                                                                        output_types = (tf.float32, tf.int32, tf.float32, tf.float32, tf.float32), \
                                                                        output_shapes = (args['input_shape'], (), (), (), ()), \
                                                                        args=[filename]), cycle_length = 4, block_length = 1)
    dataset_train = dataset_train.prefetch(args['train_batch_size_per_node'])
    dataset_train = dataset_train.apply(tf.contrib.data.batch_and_drop_remainder(args['train_batch_size_per_node']))
    dataset_train = dataset_train.repeat(1)
    #do some weight-preprocessing
    #dataset_train = dataset_train.map(lambda im,lb,wg,nw,ps: (im, lb, wg, nw, ps), num_parallel_calls=2)
    iterator_train = dataset_train.make_initializable_iterator()
    iterator_train_handle_string = iterator_train.string_handle()
    iterator_train_init_op = iterator_train.make_initializer(dataset_train)
    
    
    #validation
    h5_validation_gen = utils.hdf5_generator(shuffle=False, data_format=args["conv_params"]['data_format'])
    dataset_validation = tf.data.Dataset.from_tensor_slices(validationfiles)
    dataset_validation = dataset_validation.shuffle(len(validationfiles))
    dataset_validation = dataset_validation.interleave(lambda filename: tf.data.Dataset.from_generator(h5_validation_gen, \
                                                                                    output_types = (tf.float32, tf.int32, tf.float32, tf.float32, tf.float32), \
                                                                                    output_shapes = (args['input_shape'], (), (), (), ()), \
                                                                                    args=[filename]), cycle_length = 4, block_length = 1)
    dataset_validation = dataset_validation.prefetch(args['validation_batch_size_per_node'])
    dataset_validation = dataset_validation.apply(tf.contrib.data.batch_and_drop_remainder(args['validation_batch_size_per_node']))
    dataset_validation = dataset_validation.repeat(1)
    #do some weight-preprocessing
    #dataset_validation = dataset_validation.map(lambda im,lb,wg,nw,ps: (im, lb, wg, nw, ps), num_parallel_calls=2)
    iterator_validation = dataset_validation.make_initializable_iterator()
    iterator_validation_handle_string = iterator_validation.string_handle()
    iterator_validation_init_op = iterator_validation.make_initializer(dataset_validation)
    
    #Determine stopping point, i.e. compute last_step:
    args["steps_per_epoch"] = args["trainsamples"] // (args["train_batch_size_per_node"] * args["num_workers"])
    args["last_step"] = args["steps_per_epoch"] * args["num_epochs"]
    if args["is_chief"]:
        print("Stopping after %d global steps, doing %d steps per epoch"%(args["last_step"],args["steps_per_epoch"]))
        
        #set up file infrastructure
        if not os.path.isdir(args['logpath']):
            print("Creating log directory ",args['logpath'])
            os.makedirs(args['logpath'])
        if not os.path.isdir(args['modelpath']):
            print("Creating model directory ",args['modelpath'])
            os.makedirs(args['modelpath'])
        if not os.path.isdir(args['inputpath']) and not args['dummy_data']:
            raise ValueError("Please specify a valid path with input files in hdf5 format")

    # Train Model
    #determining which model to load:
    metafilelist = [args['modelpath']+'/'+x for x in os.listdir(args['modelpath']) if x.endswith('.meta')]
    if not metafilelist:
        #no model found, restart from scratch
        args['restart']=True
    
    #a hook that will stop training at a certain number of steps
    hooks=[tf.train.StopAtStepHook(last_step=args["last_step"])]
            
    #global step that either gets updated after any node processes a batch (async) or when all nodes process a batch for a given iteration (sync)
    global_step = tf.train.get_or_create_global_step()
    opt = args['opt_func'](**args['opt_args'])
                
    #only sync update supported
    opt = hvd.DistributedOptimizer(opt)
    
    #broadcasting model
    init_bcast = hvd.broadcast_global_variables(0)
                
    #create train step handle
    train_step = opt.minimize(loss_fn, global_step=global_step)
                
    #creating summary
    if args['create_summary']:
        if args["is_chief"]:
            summary_loss = tf.summary.scalar("loss",loss_fn)
            train_summary = tf.summary.merge([summary_loss])
            hooks.append(tf.train.StepCounterHook(every_n_steps=100,output_dir=args['logpath']))
            hooks.append(tf.train.SummarySaverHook(save_steps=100,output_dir=args['logpath'],summary_op=train_summary))
                
    # Add an op to initialize the variables.
    init_global_op = tf.global_variables_initializer()
    init_local_op = tf.local_variables_initializer()
    
    #checkpointing hook
    if args["is_chief"]:
        checkpoint_save_freq = np.min([args["steps_per_epoch"], 500])
        model_saver = tf.train.Saver(max_to_keep = 1000)
        hooks.append(tf.train.CheckpointSaverHook(checkpoint_dir=args['modelpath'], save_steps=checkpoint_save_freq, saver=model_saver))
    
    #print parameters
    if args["is_chief"]:
        for k,v in args.items():
            print("{k}: {v}".format(k=k,v=v))
        
        #print start command
        print("Starting training using "+args['optimizer']+" optimizer")
            
    with tf.train.MonitoredTrainingSession(config=sess_config, hooks=hooks) as sess:
        
        #initialize variables
        sess.run([init_global_op, init_local_op])
            
        #init iterator handle
        iterator_train_handle, iterator_validation_handle = sess.run([iterator_train_handle_string, iterator_validation_handle_string])
        
        #restore weights belonging to graph
        if not args['restart'] and args["is_chief"]:
            utils.load_model(sess, model_saver, args['modelpath'])
            
        #broadcast model
        sess.run(init_bcast)
        
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
        
        
        ##DEBUG
        #sess.run(iterator_train_init_op, feed_dict=feed_dict_train)
        #count = 0
        #while True:
        #    try:
        #        result = sess.run(variables["images_"], feed_dict=feed_dict_train)
        #        count+=1
        #        print(count)
        #    except:
        #        print("End of Epoch")
        #        sess.run(iterator_train_init_op, feed_dict=feed_dict_train)
        
        #do the training loop
        total_time = time.time()
        train_loop(sess, ops, args, iterator_train_init_op, feed_dict_train, iterator_validation_init_op, feed_dict_validation)
        total_time -= time.time()
        if args["is_chief"]:
            print("FINISHED Training. Total time %g"%(total_time))

#main
if "__main__" in __name__:
    main()
