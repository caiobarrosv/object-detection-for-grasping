from __future__ import print_function
import os
import h5py


weights_path = os.path.abspath(os.path.join(os.path.dirname( __file__ ), 'weights'))

# TODO: Set the path for the source weights file you want to load.

weights_source_path = 'VGG_coco_SSD_300x300_iter_400000.h5'
weights_source_path = os.path.abspath(os.path.join(weights_path, weights_source_path))

def print_structure(weight_file_path):
    """
    Prints out the structure of HDF5 file.

    Args:
      weight_file_path (str) : Path to the file to analyze
    """
    f = h5py.File(weight_file_path)
    try:
        if len(f.attrs.items()):
            print("{} contains: ".format(weight_file_path))
            print("Root attributes:")
        for key, value in f.attrs.items():
            print("  {}: {}".format(key, value))

        if len(f.items())==0:
            return 

        for layer, g in f.items():
            print("  {}".format(layer))
            print("    Attributes:")
            for key, value in g.attrs.items():
                print("      {}: {}".format(key, value))

            print("    Dataset:")
            for p_name in g.keys():
                param = g[p_name]
                subkeys = param.keys()
                for k_name in param.keys():
                    print("      {}/{}".format(p_name, k_name))#, param.get(k_name)[:1]))
    finally:
        f.close()
        f.close()

def print_layers(weight_file_path, layers):
    """
    Prints out specific layers of the model

    Args:
      weight_file_path (str) : Path to the file to analyze
      layers (list): list of strings containing the layers to be printed. Ex: ['conv4_3_norm_mbox_conf', 'conv6_2_mbox_conf']
    """
    f = h5py.File(weight_file_path)
    try:
        for layer, g in f.items():
            if layer in layers:
                print("  {}".format(layer))
                print("    Attributes:")
                for key, value in g.attrs.items():
                    print("      {}: {}".format(key, value))

                print("    Dataset:")
                for p_name in g.keys():
                    param = g[p_name]
                    subkeys = param.keys()
                    for k_name in param.keys():
                        print("      {}/{}".format(p_name, k_name))#, param.get(k_name)[:1]))
    finally:
        f.close()
        f.close()