import os
import tensorflow as tf
import h5py as h5
import itertools
import numpy as np

#generator for loading HDF5 data
class hdf5_generator():
    
    def __init__(self, shuffle=True, data_format="NCHW"):
        self._shuffle = shuffle
        self._data_format = data_format
    
    def __call__(self, filename):
        with h5.File(filename, 'r') as f:
            #load the chunk which is needed
            images = f['data'][...]
            labels = f['label'][...]
            normweights = f['normweight'][...]
            weights = f['weight'][...]
            psr = f['psr'][...]
        
        num_elems = images.shape[0]
        if self._shuffle:
            perm = np.random.permutation(num_elems)
            images = images[perm]
            labels = labels[perm]
            normweights = normweights[perm]
            weights = weights[perm]
            psr = psr[perm]
        
        if self._data_format == "NHWC":
            images = np.transpose(images, (0,2,3,1))
        
        for i in range(num_elems):
            yield images[i,...], labels[i,...], normweights[i,...], weights[i,...], psr[i,...]


#load model wrapper
def load_model(sess, saver, checkpoint_dir, checkpoint_index=None):
    print("Looking for model in {}".format(checkpoint_dir))
    #get list of checkpoints
    checkpoints = [x.replace(".index","") for x in os.listdir(checkpoint_dir) if x.startswith("model.ckpt") and x.endswith(".index")]
    checkpoints = sorted([(int(x.split("-")[1]),x) for x in checkpoints], key=lambda tup: tup[0])
    
    #select whioch checkpoint to restore
    if not checkpoint_index:
        restore_ckpt = os.path.join(checkpoint_dir,checkpoints[-1][1])
    else:
        restore_ckpt = None
        chk = {x[0]:x[1] for x in checkpoints}
        if checkpoint_index in chk:
            restore_ckpt = chk[restore_ckpt]
    
    #attempt to restore
    print("Restoring model {}".format(restore_ckpt))
    try:
        saver.restore(sess, restore_ckpt)
        print("Model restoration successful.")
    except:
        print("Loading model {rc} failed, exiting.".fomat(rc=restore_ckpt))
        os.sys.exit(1)
