from __future__ import print_function

import os
import argparse
import numpy as np
#import root_numpy as rnp

import glob

def parse_args():
    """Parse the command line arguments"""
    parser = argparse.ArgumentParser('split_root_dataset')
    add_arg = parser.add_argument
    add_arg('--inputdirs_signal', type=str, required=True, nargs='+',
            help='Directories to read signal data from')
    add_arg('--inputdirs_background', type=str, required=True, nargs='+',
            help='Directories to read background data from')
    add_arg('--split', type=float, nargs=3, default=[0.8,0.1,0.1],
            help='split fractions for train,validation,test')
    add_arg('--outputdir', type=str, required=True,
            help='Directories to write splitted data to')
    return parser.parse_args()

def main():
    
    #set up argparse
    args = parse_args()
    input_signal_files = []
    
    #init seed
    np.random.seed(12345)
    
    #parse input and shuffle
    #signal
    signal_file_list = []
    for directory in args.inputdirs_signal:
        filelist = [os.path.join(directory,x) for x in os.listdir(directory) if x.endswith(".root")]
        signal_file_list += filelist
    signal_files = np.asarray(signal_file_list)
    #shuffle
    perm = np.random.permutation(len(signal_files))
    signal_files = signal_files[perm]
    
    #background
    background_file_list = []
    for directory in args.inputdirs_background:
        filelist = [os.path.join(directory,x) for x in os.listdir(directory) if x.endswith(".root")]
        background_file_list += filelist
    background_files = np.asarray(background_file_list)
    #shuffle
    perm = np.random.permutation(len(background_files))
    background_files = background_files[perm]
    
    
    #determine splits
    #signal
    num_train_signal = int(signal_files.shape[0]*args.split[0])
    num_validation_signal = int(signal_files.shape[0]*args.split[1])
    #background
    num_train_background = int(background_files.shape[0]*args.split[0])
    num_validation_background = int(background_files.shape[0]*args.split[1])

    #create output directories
    if not os.path.isdir(args.outputdir):
        os.makedirs(args.outputdir)
    for phase in ["training", "test", "validation"]:
        for label in ["signal", "background"]:
            if not os.path.isdir(os.path.join(args.outputdir,phase,label)):
                os.makedirs(os.path.join(args.outputdir,phase,label))
    
    #do the split
    #signal
    #training
    for sourcename in signal_files[:num_train_signal]:
        destname = os.path.join(args.outputdir, "training", "signal", os.path.basename(sourcename))
        os.symlink(sourcename, destname)
    #validation
    for sourcename in signal_files[num_train_signal:num_train_signal+num_validation_signal]:
        destname = os.path.join(args.outputdir, "validation", "signal", os.path.basename(sourcename))
        os.symlink(sourcename, destname)
    #test
    for sourcename in signal_files[num_train_signal+num_validation_signal:]:
        destname = os.path.join(args.outputdir, "test", "signal", os.path.basename(sourcename))
        os.symlink(sourcename, destname)
    #background
    #training
    for sourcename in background_files[:num_train_background]:
        destname = os.path.join(args.outputdir, "training", "background", os.path.basename(sourcename))
        os.symlink(sourcename, destname)
    #validation
    for sourcename in background_files[num_train_background:num_train_background+num_validation_background]:
        destname = os.path.join(args.outputdir, "validation", "background", os.path.basename(sourcename))
        os.symlink(sourcename, destname)
    #test
    for sourcename in background_files[num_train_background+num_validation_background:]:
        destname = os.path.join(args.outputdir, "test", "background", os.path.basename(sourcename))
        os.symlink(sourcename, destname)
    
if __name__ == '__main__':
    main()