from __future__ import print_function

import os
import argparse
import numpy as np
import fnmatch as fn
import pandas as pd
import re
import json
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
    add_arg('--excludelist', type=str, default=None, required=False,
            help='Text file with list of files to exclude (wildcards supported). Takes precendence over include list.')
    add_arg('--includelist', type=str, default=None, required=False,
            help='Text file with list of files to include (wildcards supported). Superseded by include list.')
    add_arg('--weightfile', type=str, default=None, required=True,
            help='Text file containing a mapping of filenames to weights/cross sections')
    return parser.parse_args()

def main():
    
    #set up argparse
    args = parse_args()
    input_signal_files = []
    
    #init seed
    np.random.seed(12345)
    
    #parse excludelist
    excludelist = []
    if args.excludelist:
        with open(args.excludelist,'r') as f:
            excludelist = [x.replace('\n','') for x in f.readlines()]
        print("Excluding ",excludelist)
    
    #parse excludelist
    includelist = None
    if args.includelist:
        with open(args.includelist,'r') as f:
            includelist = [x.replace('\n','') for x in f.readlines()]
        print("Including ",includelist)
    
    #load weightfile
    weights={}
    with open(args.weightfile, 'r') as f:
        lines = f.readlines()
    for line in lines:
        if not line.startswith('#'):
            prefix, xsect = line.split()
            weights[prefix] = float(xsect)
    
    #parse input and shuffle
    #--------------------------------------
    #--------------- SIGNAL ---------------
    #--------------------------------------
    #initial file list
    signal_file_list = []
    for directory in args.inputdirs_signal:
        filelist = [os.path.join(directory,x) for x in os.listdir(directory) if x.endswith(".root")]
        signal_file_list += filelist
    
    #include list:
    if includelist:
        signal_file_list = list(filter(lambda x: any([fn.fnmatch(x,y) for y in includelist]), signal_file_list))
    
    #remove stuff
    if excludelist:
        signal_file_list = list(filter(lambda x: all([not fn.fnmatch(x,y) for y in excludelist]), signal_file_list))
    
    #add to dataframe
    signaldf = pd.DataFrame(signal_file_list, columns=['filename'])
    
    #extract RPV params
    tmpser = signaldf.apply(lambda x: re.findall(r'^RPV(\d{1,}?)_(\d{1,}?)_(\d{1,}?)-.*$', os.path.basename(x['filename'])), axis=1)
    signaldf['rpv'] = tmpser.apply(lambda x: int(x['filename'][0]), axis=1)
    signaldf['mglu'] = tmpser.apply(lambda x: int(x['filename'][1]), axis=1)
    signaldf['mneu'] = tmpser.apply(lambda x: int(x['filename'][2]), axis=1)
    
    #merge in the weights
    signaldf['weights'] = signaldf.apply(lambda x: weights[os.path.basename(x['filename']).split("-10k")[0]], axis=1)
    
    #sort by rpv params
    signaldf.sort_values(by=['rpv','mglu','mneu'], inplace=True)
    
    #group and split
    signaldf_train = pd.DataFrame(signaldf.groupby(['rpv','mglu','mneu']).apply(lambda x: x.iloc[0:int(x.shape[0]*args.split[0])]).reset_index(drop=True))
    signaldf_train["phase"] = "training"
    signaldf_validation = pd.DataFrame(signaldf.groupby(['rpv','mglu','mneu']).apply(lambda x: x.iloc[int(x.shape[0]*args.split[0]):int(x.shape[0]*args.split[0])+int(x.shape[0]*args.split[1])]).reset_index(drop=True))
    signaldf_validation["phase"] = "validation"
    signaldf_test = pd.DataFrame(signaldf.groupby(['rpv','mglu','mneu']).apply(lambda x: x.iloc[int(x.shape[0]*args.split[0])+int(x.shape[0]*args.split[1]):]).reset_index(drop=True))
    signaldf_test["phase"] = "test"
    #concatenate back
    signaldf = pd.concat([signaldf_train, signaldf_validation, signaldf_test])
    
    #compute counts for all of them:
    tmpc = pd.DataFrame(signaldf.groupby(['rpv','mglu','mneu', 'phase']).count().reset_index())
    tmpc.rename(columns={'filename':'count'},inplace=True)
    signaldf = signaldf.merge(tmpc[['rpv','mglu','mneu', 'phase', 'count']], on=['rpv','mglu','mneu', 'phase'], how='left')
    
    #--------------------------------------
    #------------- Background -------------
    #--------------------------------------
    #initial file list
    background_file_list = []
    for directory in args.inputdirs_background:
        filelist = [os.path.join(directory,x) for x in os.listdir(directory) if x.endswith(".root")]
        background_file_list += filelist
    
    #include stuff
    if includelist:
        background_file_list = list(filter(lambda x: any([fn.fnmatch(x,y) for y in includelist]), background_file_list))
    
    #remove stuff
    if excludelist:
        background_file_list = list(filter(lambda x: all([not fn.fnmatch(x,y) for y in excludelist]), background_file_list))
    
    #add to dataframe
    backgrounddf = pd.DataFrame(background_file_list, columns=['filename'])

    #extract JZ params
    tmpser = backgrounddf.apply(lambda x: re.findall(r'^QCDBkg_JZ(\d{1,}?)_(\d{1,}?)_(\d{1,}?)-.*$', os.path.basename(x['filename'])), axis=1)
    backgrounddf['jz'] = tmpser.apply(lambda x: int(x['filename'][0]), axis=1).astype(int)
    backgrounddf['pt_lo'] = tmpser.apply(lambda x: int(x['filename'][1]), axis=1).astype(int)
    backgrounddf['pt_hi'] = tmpser.apply(lambda x: int(x['filename'][2]), axis=1).astype(int)
    
    #merge in the weights
    backgrounddf['weights'] = backgrounddf.apply(lambda x: weights[os.path.basename(x['filename']).split("-10k")[0]], axis=1)
    
    #sort by those
    backgrounddf.sort_values(by=['jz','pt_lo','pt_hi'], inplace=True)
    
    #group and split
    backgrounddf_train = pd.DataFrame(backgrounddf.groupby(['jz','pt_lo','pt_hi']).apply(lambda x: x.iloc[0:int(x.shape[0]*args.split[0])]).reset_index(drop=True))
    backgrounddf_train['phase'] = 'training'
    backgrounddf_validation = pd.DataFrame(backgrounddf.groupby(['jz','pt_lo','pt_hi']).apply(lambda x: x.iloc[int(x.shape[0]*args.split[0]):int(x.shape[0]*args.split[0])+int(x.shape[0]*args.split[1])]).reset_index(drop=True))
    backgrounddf_validation['phase'] = 'validation'
    backgrounddf_test = pd.DataFrame(backgrounddf.groupby(['jz','pt_lo','pt_hi']).apply(lambda x: x.iloc[int(x.shape[0]*args.split[0])+int(x.shape[0]*args.split[1]):]).reset_index(drop=True))
    backgrounddf_test['phase'] = 'test'
    #concatenate back
    backgrounddf = pd.concat([backgrounddf_train, backgrounddf_validation, backgrounddf_test])
    
    #compute counts for all of them:
    tmpc = pd.DataFrame(backgrounddf.groupby(['jz','pt_lo','pt_hi', 'phase']).count().reset_index())
    tmpc.rename(columns={'filename':'count'},inplace=True)
    backgrounddf = backgrounddf.merge(tmpc[['jz','pt_lo','pt_hi', 'phase', 'count']], on=['jz', 'pt_lo', 'pt_hi', 'phase'], how='left')
    
    #--------------------------------------
    #-------------- Weights ---------------
    #--------------------------------------
    #we need to divide the weights by the counts to get weights corresponding to the sample populations
    signaldf['weights'] = signaldf['weights'] / signaldf['count'].astype(np.float32)
    backgrounddf['weights'] = backgrounddf['weights'] / backgrounddf['count'].astype(np.float32)
    
    #--------------------------------------
    #--------------- Output ---------------
    #--------------------------------------
    #create output directories
    if not os.path.isdir(args.outputdir):
        os.makedirs(args.outputdir)
    for phase in ["training", "test", "validation"]:
        for label in ["signal", "background"]:
            if not os.path.isdir(os.path.join(args.outputdir,phase,label)):
                os.makedirs(os.path.join(args.outputdir,phase,label))
    
    #save new weights dictionary
    #write out the files
    for phase in ['training', 'validation', 'test']:
        #signal
        outpath = os.path.join(args.outputdir, phase, "signal")
        selectdf = signaldf[ signaldf.phase == phase ]
        #data
        sourcefiles = selectdf['filename'].values
        #create symlinks
        for sourcename in sourcefiles:
            destname = os.path.join(outpath, os.path.basename(sourcename))
            os.symlink(sourcename, destname)
        #extract weights dict
        weightdf = pd.DataFrame(selectdf.groupby(['rpv', 'mglu', 'mneu']).apply(lambda x: x['weights'].iloc[0]).reset_index().rename(columns={0:'weights'}))
        weightdf['prefix'] = weightdf.apply(lambda x: 'RPV{rpv:d}_{mglu:d}_{mneu:d}'.format(rpv=int(x['rpv']), mglu=int(x['mglu']), mneu=int(x['mneu'])), axis=1)
        sweights = {x[0]: {"weight": x[1], "label": 1} for x in weightdf[['prefix', 'weights']].values}
            
        #background
        outpath = os.path.join(args.outputdir, phase, "background")
        selectdf = backgrounddf[ backgrounddf.phase == phase ]
        #data
        sourcefiles = backgrounddf['filename'].values
        #create symlinks
        for sourcename in sourcefiles:
            destname = os.path.join(outpath, os.path.basename(sourcename))
            os.symlink(sourcename, destname)
        
        #extract weights dict
        weightdf = pd.DataFrame(selectdf.groupby(['jz', 'pt_lo', 'pt_hi']).apply(lambda x: x['weights'].iloc[0]).reset_index().rename(columns={0:'weights'}))
        weightdf['prefix'] = weightdf.apply(lambda x: 'QCDBkg_JZ{jz:d}_{ptl:d}_{pth:d}'.format(jz=int(x['jz']), ptl=int(x['pt_lo']), pth=int(x['pt_hi'])), axis=1)
        bweights = {x[0]: {"weight": x[1], "label": 0} for x in weightdf[['prefix', 'weights']].values}
        
        #store weight dicts:
        outweights = {**sweights, **bweights}
        with open(os.path.join(args.outputdir, phase, 'metadata.json'), 'w') as f:
            json.dump(outweights, f)
    
if __name__ == '__main__':
    main()