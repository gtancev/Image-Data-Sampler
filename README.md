# Image-Data-Sampler
A class for image (sub)sampling for CNN derivates with the option to apply several kinds of augmentation methods.

Description
-----------
What the class does is basically random sampling within a image volume with provided size and padding. To this volume subsample, Gaussian-filtering can be applied and the original images together with the Gaussian-filtered versions are stacked in the channel dimension. Next, transformation are applied, if requested. The output is a tensor of shape (batch_size, nx, ny, nz, n_channels). Same holds for the masks. In addition, it is possible to selectively sample, i.e. that each n-th sample includes labelled data (by which the selectively sampled class can be determined). Read more about the inputs below.  


Mandatory Inputs
----------------



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
    

Optional Inputs
---------------
    
Sampling

    whiten: perform Gaussian-filtering of images as type bool (default: True)
    subtractGaussSigma: standard deviation for Gaussian filtering as list of len 1 or ndims (default: [5])
    nooriginal: use only Gaussian-filtered images as type bool (default: False)
    each_with_labels: input of type int to fix the selective sampling interval (each n-th sample)
    minlabel: input of type int to fix which label/class to selectively sample
    
Data Augmentation

