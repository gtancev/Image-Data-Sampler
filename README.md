# Image-Data-Sampler
A class for image (sub)sampling for CNN derivates with the option to apply several kinds of augmentation methods.

Arguments:

Mandatory
----------



    w: subsample dimensions of type int and length ndims, e.g. [80, 80, 80]
    p: subsample paddings of type int and length ndims, e.g. [5, 5, 5]
    location: path to folders with data for training/testing of type str
    folder: folder name of type str.
    featurefiles: filenames of featurefiles
    maskfiles: filenames of mask file(s) to be used as reference
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
