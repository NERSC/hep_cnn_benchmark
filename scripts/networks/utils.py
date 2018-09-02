import os
import tensorflow as tf
import h5py as h5
import itertools
import numpy as np
try:
    import root_numpy as rnp
except:
    print("Warning, no RootPy detected, cannot use ROOT iterators")


#some helper for suppressing annoying output from subroutines
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


class root_generator():
    
    def transform_calohits_to_pointcloud(self, eta, phi, energy, emfrac):
        #perform sampling with replacement, only if there are fewer points than requested points
        choice = np.random.choice(eta.shape[0], self._num_calorimeter_hits, replace=(eta.shape[0] < self._num_calorimeter_hits))
        eta = eta[choice].astype(dtype=self._dtype)
        phi = phi[choice].astype(dtype=self._dtype)
        energy = energy[choice].astype(dtype=self._dtype)
        emfrac = emfrac[choice].astype(dtype=self._dtype)
        #pad with 0-class for emhits
        result=np.stack([eta, np.cos(phi), np.sin(phi), energy, emfrac, np.zeros(self._num_calorimeter_hits, dtype=self._dtype)], axis=1)
        return result
            
    def transform_tracks_to_pointcloud(self, eta, phi):
        #perform sampling with replacement
        choice = np.random.choice(eta.shape[0], self._num_tracks, replace=(eta.shape[0] < self._num_tracks))
        eta = eta[choice].astype(dtype=self._dtype)
        phi = phi[choice].astype(dtype=self._dtype)
        #pad by zero for EM and EMfrac and one for tracks
        result=np.stack([eta, np.cos(phi), np.sin(phi), \
                        np.zeros(self._num_tracks, dtype=self._dtype), \
                        np.zeros(self._num_tracks, dtype=self._dtype), \
                        np.ones(self._num_tracks, dtype=self._dtype)], axis=1)
        return result
    
    def __init__(self, num_calorimeter_hits, num_tracks, shuffle=True, dtype=np.float32):
        self._shuffle = shuffle
        self._branches = {
            'Tower.Eta',
            'Tower.Phi',
            'Tower.E'  ,
            'Tower.Eem',
            'Track.Eta',
            'Track.Phi'
        }
        self._num_calorimeter_hits = num_calorimeter_hits
        self._num_tracks = num_tracks
        self._dtype=dtype
    
    def __call__(self, filename, label):
        with suppress_stdout_stderr():
            self._tree = rnp.root2array(filename, treename='Delphes',
                                  branches=self._branches, stop=None,
                                  warn_missing_tree=True)
        
        num_examples = self._tree['Tower.Eta'].shape[0]
        
        #preprocess
        #em hits
        calohits = map(self.transform_calohits_to_pointcloud, self._tree['Tower.Eta'], self._tree['Tower.Phi'], self._tree['Tower.E'], self._tree['Tower.Eem'])
        calohits = np.stack(calohits, axis=0)
        #tracks
        tracks = map(self.transform_tracks_to_pointcloud, self._tree['Track.Eta'], self._tree['Track.Phi'])
        tracks = np.stack(tracks, axis=0)
        #stack all of it together
        data = np.concatenate([calohits,tracks],axis=1)
        
        if self._shuffle:
            perm = np.random.permutation(num_examples)
            data = data[perm]
            
        for i in range(num_examples):
            yield data[i,...], label


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