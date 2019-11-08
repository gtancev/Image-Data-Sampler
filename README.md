# Image-Data-Sampler
A class for image (sub)sampling for CNN derivates with the option to apply several kinds of augmentation methods.

Arguments:

Mandatory
----------

::

    w: subsample dimensions of type int and length ndims, e.g. [80, 80, 80]
    p: subsample paddings of type int and length ndims, e.g. [5, 5, 5]
    location: path to folders with data for training/testing of type str
    folder: folder name of type str.
    featurefiles: filenames of featurefiles
    maskfiles: filenames of mask file(s) to be used as reference
    nclasses: number of classes of type int
    params: optional parameters for data augmentation
