import os
import tensorflow as tf
import h5py as h5
import itertools
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
try:
    from rootpy.io import root_open
    import root_numpy as rnp
except:
    print("Warning, no RootPy detected, cannot use ROOT iterators")


# ROC curve
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
    
    #merge event classes together and pick a subset
    def transform_to_pointcloud(self, calo_eta, calo_phi, calo_energy, calo_emfrac, track_eta, track_phi):
        #calorimeter hits
        calo_eta = calo_eta.astype(dtype=self._dtype)
        calo_phi = calo_phi.astype(dtype=self._dtype)
        calo_energy = (calo_energy.astype(dtype=self._dtype) - self._metadata['calo_min']) / (self._metadata['calo_max'] - self._metadata['calo_min'])
        calo_emfrac = (calo_emfrac.astype(dtype=self._dtype) - self._metadata['em_min']) / (self._metadata['em_max'] - self._metadata['em_min'])
        calo_class = np.zeros(calo_eta.shape[0], dtype=self._dtype)
        #stack
        calo = np.stack([calo_eta, np.cos(calo_phi), np.sin(calo_phi), calo_energy, calo_emfrac, calo_class], axis=1)
        
        #tracks
        track_eta = track_eta.astype(dtype=self._dtype)
        track_phi = track_phi.astype(dtype=self._dtype)
        track_energy = (np.zeros(track_eta.shape[0], dtype=self._dtype) - self._metadata['calo_min']) / (self._metadata['calo_max'] - self._metadata['calo_min'])
        track_emfrac = (np.zeros(track_eta.shape[0], dtype=self._dtype) - self._metadata['em_min']) / (self._metadata['em_max'] - self._metadata['em_min'])
        track_class = np.ones(track_eta.shape[0], dtype=self._dtype)
        #stack
        track = np.stack([track_eta, np.cos(track_phi), np.sin(track_phi), track_energy, track_emfrac, track_class], axis=1)
        
        #concatenate
        result = np.concatenate([calo, track], axis=0)
        
        #pick random choice
        choice = np.random.choice(result.shape[0], self._num_points, replace=(result.shape[0] < self._num_points))
        result = result[choice]

        return result
    
    #def transform_calohits_to_pointcloud(self, eta, phi, energy, emfrac):
    #    #perform sampling with replacement, only if there are fewer points than requested points
    #    choice = np.random.choice(eta.shape[0], self._num_calorimeter_hits, replace=(eta.shape[0] < self._num_calorimeter_hits))
    #    eta = eta[choice].astype(dtype=self._dtype)
    #    phi = phi[choice].astype(dtype=self._dtype)
    #    energy = (energy[choice].astype(dtype=self._dtype) - self._metadata['calo_min']) / (self._metadata['calo_max'] - self._metadata['calo_min'])
    #    emfrac = (emfrac[choice].astype(dtype=self._dtype) - self._metadata['em_min']) / (self._metadata['em_max'] - self._metadata['em_min'])
    #    #pad with 0-class for emhits
    #    result=np.stack([eta, np.cos(phi), np.sin(phi), energy, emfrac, np.zeros(self._num_calorimeter_hits, dtype=self._dtype)], axis=1)
    #    return result
    #        
    #def transform_tracks_to_pointcloud(self, eta, phi):
    #    #perform sampling with replacement
    #    choice = np.random.choice(eta.shape[0], self._num_tracks, replace=(eta.shape[0] < self._num_tracks))
    #    eta = eta[choice].astype(dtype=self._dtype)
    #    phi = phi[choice].astype(dtype=self._dtype)
    #    energy = (np.zeros(self._num_tracks, dtype=self._dtype) - self._metadata['calo_min']) / (self._metadata['calo_max'] - self._metadata['calo_min'])
    #    emfrac = (np.zeros(self._num_tracks, dtype=self._dtype) - self._metadata['em_min']) / (self._metadata['em_max'] - self._metadata['em_min'])
    #    #pad by zero for EM and EMfrac and one for tracks
    #    result=np.stack([eta, np.cos(phi), np.sin(phi), energy, emfrac, np.ones(self._num_tracks, dtype=self._dtype)], axis=1)
    #    return result
    
    def __init__(self, num_points, metadata, shuffle=True, blocksize=10, dtype=np.float32):
        self._shuffle = shuffle
        self._branches = {
            'Tower.Eta',
            'Tower.Phi',
            'Tower.E'  ,
            'Tower.Eem',
            'Track.Eta',
            'Track.Phi'
        }
        self._num_points = num_points
        self._dtype = dtype
        self._blocksize = blocksize
        self._metadata = metadata
        
        #reset minimum for calo and em:
        self._metadata['em_min'] = np.min([self._metadata['em_min'], 0.])
        self._metadata['calo_min'] = np.min([self._metadata['calo_min'], 0.])
    
    def __call__(self, filename):
        try:
            #determine label and weights
            basename = os.path.basename(filename).split("-10k")[0]
            label = self._metadata[basename]['label']
            weight = self._metadata[basename]['weight']
        except Exception as error:
            print("Error, for looking up {basename} in metadata: {err}".format(basename=basename, err=error))
            return
        
        try:
            #with suppress_stdout_stderr():
            with root_open(filename) as f:
            
                #get tree
                mtree = f['Delphes']
                num_examples = len(mtree)
            
                #iterate over blocks
                for i in range(0, num_examples, self._blocksize):
                    #determine read range
                    start = i
                    end = np.min([i+self._blocksize, num_examples])
                    #read the tree
                    tree = rnp.tree2array(mtree, branches=self._branches, start=start, stop=end)
                
                    #preproces
                    data = map(self.transform_to_pointcloud, tree['Tower.Eta'], tree['Tower.Phi'], tree['Tower.E'], tree['Tower.Eem'], tree['Track.Eta'], tree['Track.Phi'])
                    data = np.stack(data, axis=0)
                    
                    ##em hits
                    #calohits = map(self.transform_calohits_to_pointcloud, tree['Tower.Eta'], tree['Tower.Phi'], tree['Tower.E'], tree['Tower.Eem'])
                    #calohits = np.stack(calohits, axis=0)
                    ##tracks
                    #tracks = map(self.transform_tracks_to_pointcloud, tree['Track.Eta'], tree['Track.Phi'])
                    #tracks = np.stack(tracks, axis=0)
                    ##stack all of it together
                    #data = np.concatenate([calohits,tracks],axis=1)
                    
                    if self._shuffle:
                        perm = np.random.permutation(self._blocksize)
                        data = data[perm]
                    
                    for i in range(data.shape[0]):
                        yield data[i,...], label, weight
                        
        except Exception as error:
            print("Error: cannot open file {fname}: {err}".format(fname=filename, err=error))
            return


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