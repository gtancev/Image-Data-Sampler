# Image Data Sampler
A class for image (sub)sampling for CNN derivates with the option to apply several kinds of augmentation methods.

Description
===========

What the class does is basically random sampling within a image volume with provided size and padding. To this volume subsample, Gaussian-filtering can be applied and the original images together with the Gaussian-filtered versions are stacked in the channel dimension. Next, transformation are applied, if requested. The output is a tensor of shape (batch_size, nx, ny, nz, n_channels). Same holds for the masks. In addition, it is possible to selectively sample, i.e. that each n-th sample includes labelled data (by which the selectively sampled class can be determined). Read more about the inputs below.

**Be aware that each class in the masks will correspong to one channel, i.e. for two classes there will be two channels (and not one).**


Mandatory Inputs
================



    w: subsample dimensions as list of type int and length ndims, e.g. [80, 80, 80]
    
    p: subsample paddings as list of type int and length ndims, e.g. [5, 5, 5]
    
    location: path to folders with data for training/testing of type str
    
    folder: folder name of type str
    
    featurefiles: filenames of featurefiles of tupe str as list
    
    maskfiles: filenames of mask file(s) to be used as reference as type str as list
    
    nclasses: number of classes of type int
    
    params: optional parameters for data augmentation
    
    
Example
-------

To extract single subvolumes, the method random_sample is used after initiation of a data collection instance.

    w = [80, 80, 80]
    p = [5, 5, 5]
    location = '/scicore/home/scicore/rumoke43/mdgru_experiments/files'
    folder = 'train'
    files = ['flair_pp.nii', 'mprage_pp.nii', 'pd_pp.nii', 't2_pp.nii']
    mask = ['mask.nii']
    nclasses = 2
    
    params = {}
    params['deform','deformSigma'] = [0], [0]
    
    threaded_data_instance = dsc.ThreadedDataSetCollection(w, p, location, folder, files, mask, nclasses, params)
    
    batch, batchlabs = threaded_data_instance.random_sample()
    
To sample a whole volume, a separate method that generates a generator object is available. This is useful for evaluation.

    batches = threaded_data_instance.get_volume_batch_generators()
    
    for batch, file, shape, w, p in batches:
        for subvol, subvol_mask, imin, imax in batch:
            ...
            
            
Optional Inputs
===============

Optional inputs can be provided as dict (see example above).
    
Sampling
--------

    whiten: perform Gaussian-filtering of images as type bool (default: True)
    
    subtractGaussSigma: standard deviation for Gaussian filtering as list of len 1 or ndims (default: [5])
    
    nooriginal: use only Gaussian-filtered images as type bool (default: False)
    
    each_with_labels: input of type int to fix the selective sampling interval, i.e. each n-th sample (default: 0, i.e. off)
    
    minlabel: input of type int to fix which label/class to selectively sample (default: 1)
    
Data Augmentation
------------------

    deform: deformation grid spacing in voxels as list of len 1 or ndims with types int (default: [0])
    
    deformSigma: given a deformation grid spacing, this determines the standard deviations for each dimension of the random deformation vectors as list with length 1 or ndims with types float (default: [0])
    
    mirror: list input of len 1 or ndims of type bool to activate random mirroring along the specified axes during training (default: [0])
    
    rotation: list input of len 1 or ndims of type float as amount in radians to randomly rotate the input around a randomly drawn vector (default: [0])
    
    scaling: list input of len 1 or ndims of type float as amount ot randomly scale images, per dimension, or for all dimensions, as a factor, e.g. 1.25 (default: [0])
    
    shift: list input of len 1 or ndims of type int in order to sample outside of discrete coordinates, this can be set to 1 on the relevant axes (default: [0])
    
    gaussiannoise: input of type bool or float to apply random multiplicative Gaussian noise on the input data with given std and mean 1 (default: False)
    
    vary_mean: input of type float to vary mean of images in a random manner (default: 0)
    
    vary_stddev: input of type float to vary standard deviation of images in a random manner (default: 0)
