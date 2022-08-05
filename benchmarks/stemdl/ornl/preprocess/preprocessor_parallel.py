import multiprocessing
import numpy as np
from hdf5_helper import HDF5Dataset
import sys, os


def wrapper_function(parameters):
    dataset, index, dest, offset = parameters
    dataset.convert_to_npy(index, dest, offset)


def preprocess(source, destination, num_cores):
    subdirs = ["train", "test", "dev"]
    offsets = [0 for _ in range(3)]
    offset = 0

    # Parallel processing setup
    if num_cores == -1:
        num_cores = multiprocessing.cpu_count()

    for part, subdir in enumerate(subdirs):
        print("Processing " + subdir + " with offset " + str(offset))
        offsets[part] = offset
        dataset = HDF5Dataset(source + "/" + subdir, recursive=False, load_data=False,
                              data_cache_size=4, transform=None)

        print("Total images to preprocess in " + subdir + " : ", len(dataset.data_info))
        offset += len(dataset.data_info)

        pool = multiprocessing.Pool()
        pool = multiprocessing.Pool(processes=num_cores)
        ds_list = [dataset] * len(dataset.data_info)
        index_list = list(range(len(dataset.data_info)))
        dest_list = [os.path.join(destination, subdir)] * len(dataset.data_info)
        offset_list = [offsets[part]] * len(dataset.data_info)
        pool.map(wrapper_function, zip(ds_list, index_list, dest_list, offset_list))


def main(argv):
    source_dir = argv[0]
    destination_dir = argv[1]
    num_cores = int(argv[2])
    print("Source directory: ", source_dir)
    print("Destination directory: ", destination_dir)
    preprocess(source_dir, destination_dir, num_cores)


if __name__ == "__main__":
    main(sys.argv[1:])
