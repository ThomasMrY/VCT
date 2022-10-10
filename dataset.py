# Custum Dataset loader
from matplotlib import transforms
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
import torch
import h5py
from PIL import Image
from imageio import imread
from glob import glob
import PIL
import scipy.io as sio
from sklearn.utils.extmath import cartesian
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def features_to_index(features):
    """Returns the indices in the input space for given factor configurations.

    Args:
        features: Numpy matrix where each row contains a different factor
        configuration for which the indices in the input space should be
        returned.
    """
    factor_sizes = [4, 24, 183]
    num_total_atoms = np.prod(factor_sizes)
    lookup_table = np.zeros(num_total_atoms, dtype=np.int64)
    global_features = cartesian([np.array(list(range(i))) for i in factor_sizes])
    feature_state_space_index = _features_to_state_space_index(global_features)
    lookup_table[feature_state_space_index] = np.arange(num_total_atoms)
    state_space_to_save_space_index = lookup_table
    state_space_index = _features_to_state_space_index(features)
    return state_space_to_save_space_index[state_space_index]

def _features_to_state_space_index(features):
    """Returns the indices in the atom space for given factor configurations.

    Args:
    features: Numpy matrix where each row contains a different factor
        configuration for which the indices in the atom space should be
        returned.
    """
    factor_sizes = [4, 24, 183]
    num_total_atoms = np.prod(factor_sizes)
    factor_bases = num_total_atoms / np.cumprod(factor_sizes)
    if (np.any(features > np.expand_dims(factor_sizes, 0)) or
        np.any(features < 0)):
        raise ValueError("Feature indices have to be within [0, factor_size-1]!")
    return np.array(np.dot(features, factor_bases), dtype=np.int64)

def _load_mesh(filename):
    """Parses a single source file and rescales contained images."""
    with open(filename, "rb") as f:
        mesh = np.einsum("abcde->deabc", sio.loadmat(f)["im"])
    flattened_mesh = mesh.reshape((-1,) + mesh.shape[2:])
    rescaled_mesh = np.zeros((flattened_mesh.shape[0], 64, 64, 3))
    for i in range(flattened_mesh.shape[0]):
        pic = PIL.Image.fromarray(flattened_mesh[i, :, :, :])
        pic.thumbnail((64, 64), PIL.Image.ANTIALIAS)
        rescaled_mesh[i, :, :, :] = np.array(pic)
    return rescaled_mesh * 1. / 255

def _load_data(dset_folder):
    dataset = np.zeros((24 * 4 * 183, 64, 64, 3))
    all_files = [x for x in os.listdir(dset_folder) if ".mat" in x]
    for i, filename in enumerate(all_files):
        data_mesh = _load_mesh(os.path.join(dset_folder, filename))
        factor1 = np.array(list(range(4)))
        factor2 = np.array(list(range(24)))
        all_factors = np.transpose([
          np.tile(factor1, len(factor2)),
          np.repeat(factor2, len(factor1)),
          np.tile(i,
                  len(factor1) * len(factor2))
        ])
        indexes = features_to_index(all_factors)
        dataset[indexes] = data_mesh
    return dataset

class CustomTensorDataset(Dataset):
    def __init__(self, data_tensor, transform, eval_flag = False):
        self.data_tensor = data_tensor
        self.transform = transform
        self.eval_flag = eval_flag

    def __getitem__(self, index):
        img = Image.fromarray(self.data_tensor[index])
        return self.transform(img), torch.tensor(1.0)
        
    def __len__(self):
        return self.data_tensor.shape[0]


class ImagesFilesDataset(Dataset):
    def __init__(self, paths, transform, labels = None, eval_flag = False):
        self.paths = paths
        self.transform = transform
        self.lables = labels
        self.eval_flag = eval_flag

    def __getitem__(self, index):
        data_img = imread(self.paths[index])[:, :, :3]
        img = Image.fromarray(data_img)
        return self.transform(img), torch.tensor(1.0)
        
    def __len__(self):
        return len(self.paths)

class ImagesFilesDataset_eval(Dataset):
    def __init__(self, paths, transform, labels, eval_flag = True):
        self.paths = paths
        self.transform = transform
        self.labels = labels
        self.eval_flag = eval_flag

    def __getitem__(self, index):
        data_img = imread(self.paths[index])[:, :, :3]
        img = Image.fromarray(data_img)
        return self.transform(img), self.labels[index]
        
    def __len__(self):
        return len(self.paths)
    

class ClipDataset(Dataset):
    def __init__(self, data_tensor, transform, clip_transform):
        self.data_tensor = data_tensor
        self.transform = transform
        self.clip_transform = clip_transform

    def __getitem__(self, index):
        img = Image.fromarray(self.data_tensor[index])
        return self.transform(img), self.clip_transform(img)
        
    def __len__(self):
        return self.data_tensor.shape[0]

def custum_dataset(
    dset_dir = "./",
    Dataset_fc = CustomTensorDataset,
    transform = None,
    name = "shapes3d",
    eval_flag = False
):

    if name.lower() == 'dsprites':
        root = os.path.join(dset_dir, 'dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz')
        data = np.load(root, encoding='bytes')
        data = data['imgs']
        train_kwargs = {'data_tensor':data, 'transform':transform, 'eval_flag': True}
        dset = Dataset_fc
    
    elif name.lower() == 'shapes3d':
        data = h5py.File(os.path.join(dset_dir,'3dshapes.h5'), 'r')
        train_kwargs = {'data_tensor':data["images"][()],"transform":transform, 'eval_flag': True}
        dset = Dataset_fc
    
    elif name.lower() == "mpi_toy":
        data  = np.load(os.path.join(dset_dir,"mpi3d_toy.npz"), "r")
        images = data['images']
        train_kwargs = {'data_tensor':images,'transform':transform, 'eval_flag': True}
        dset = Dataset_fc
        

    elif name.lower() == "mpi_mid":
        data  = np.load(os.path.join(dset_dir,"mpi3d_realistic.npz"), "r")
        images = data['images']
        train_kwargs = {'data_tensor':images,'transform':transform, 'eval_flag': True}
        dset = Dataset_fc


    elif name.lower() == "mpi_real":
        data  = np.load(os.path.join(dset_dir,"mpi3d_real.npz"), "r")
        images = data['images']
        train_kwargs = {'data_tensor':images,'transform':transform, 'eval_flag': True}
        dset = Dataset_fc

    elif name.lower() == "clevr":
        if eval_flag:
            path = os.path.join(dset_dir.replace("clevr",""),"clevr_test/images/*.png")
            images = sorted(glob(path))
            masks = np.load(os.path.join(dset_dir.replace("clevr",""),"clevr_test/masks.npz"))['masks']
            train_kwargs = {'paths':images,'transform':transform, 'labels':masks, 'eval_flag': True}
            dset = ImagesFilesDataset_eval
        else:
            path = os.path.join(dset_dir.replace("clevr",""),"images_clevr/*.png")
            images = sorted(glob(path))
            train_kwargs = {'paths':images,'transform':transform}
            dset = ImagesFilesDataset
    
    elif name.lower() == "mnist":
        path = os.path.join(dset_dir,"mnist_5.npz")
        data = np.load(path,"r")['data']
        train_kwargs = {'data_tensor':data[:,0,:,:],"transform":transform}
        dset = Dataset_fc

    elif name.lower() == "celebanpz":
        path = os.path.join(dset_dir,"celeba.npz")
        data = np.load(path,"r")['imgs']
        train_kwargs = {'data_tensor':data,"transform":transform}
        dset = Dataset_fc
    
    elif name.lower() == "celeba":
        path = os.path.join(dset_dir,"data.npz")
        data = np.load(path,"r")['data']
        train_kwargs = {'data_tensor':np.uint8(data*255),'transform':transform}
        dset = Dataset_fc
    
    elif name.lower() == 'cars3d':
        dataset = _load_data(dset_dir.replace("cars3d","cars"))
        train_kwargs = {'data_tensor':np.uint8(dataset*255), 'transform':transform, 'eval_flag': True}
        dset = Dataset_fc
    
    elif name.lower() == 'tetrominoes_npz':
        data = np.load(os.path.join(dset_dir.replace("_npz",""), 'data.npz'),"r")['images']
        train_kwargs = {'data_tensor':np.uint8(data),'transform':transform}
        dset = Dataset_fc
    
    elif name.lower() == 'object_room_npz':
        data = np.load(os.path.join(dset_dir.replace("object_room_npz","objects_room"), 'data.npz'),"r")['images']
        train_kwargs = {'data_tensor':np.uint8(data),'transform':transform}
        dset = Dataset_fc
    elif name.lower() == 'coco':
        file_list = np.load(os.path.join(dset_dir,'file_np.npy')).tolist()
        train_kwargs = {'file_list':file_list, 'transform':transform}
        dset = ImagesFilesDataset
    
    elif name.lower() == "kitti":
        txt_file = open(os.path.join(dset_dir,'file_paths.txt'),"r")
        row_file_list = txt_file.readlines()
        file_list = []
        for line in row_file_list:
            file_list.append(os.path.join(dset_dir,line).replace("\n",""))
        train_kwargs = {'paths':file_list, 'transform':transform}
        dset = ImagesFilesDataset
    else:
        raise NotImplementedError


    return dset(**train_kwargs)


def custum_dataset_clip(
    dset_dir = "./",
    Dataset_fc = ClipDataset,
    transform = None,
    clip_transform = None,
    name = "shapes3d"
):

    if name.lower() == 'dsprites':
        root = os.path.join(dset_dir, 'dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz')
        data = np.load(root, encoding='bytes')
        data = data['imgs']
        train_kwargs = {'data_tensor':data, 'transform':transform, 'clip_transform': clip_transform}
        dset = Dataset_fc
    
    elif name.lower() == 'shapes3d':
        data = h5py.File(os.path.join(dset_dir,'3dshapes.h5'), 'r')
        images = data["images"][()]
        data = images
        train_kwargs = {'data_tensor':data,'transform':transform, "clip_transform":clip_transform}
        dset = Dataset_fc


    else:
        raise NotImplementedError


    return dset(**train_kwargs)


try:
    """Objects Room dataset reader."""
    import functools
    import tensorflow as tf

    COMPRESSION_TYPE = tf.io.TFRecordOptions.get_compression_type_string('GZIP')
    BYTE_FEATURES = ['mask', 'image']

    def clevr_dataset(tfrecords_path, read_buffer_size=None,
                    map_parallel_calls=None):
        IMAGE_SIZE = [240, 320]
        # The maximum number of foreground and background entities in the provided
        # dataset. This corresponds to the number of segmentation masks returned per
        # scene.
        MAX_NUM_ENTITIES = 11
        BYTE_FEATURES = ['mask', 'image', 'color', 'material', 'shape', 'size']

        # Create a dictionary mapping feature names to `tf.Example`-compatible
        # shape and data type descriptors.
        features = {
            'image': tf.FixedLenFeature(IMAGE_SIZE+[3], tf.string),
            'mask': tf.FixedLenFeature([MAX_NUM_ENTITIES]+IMAGE_SIZE+[1], tf.string),
            'x': tf.FixedLenFeature([MAX_NUM_ENTITIES], tf.float32),
            'y': tf.FixedLenFeature([MAX_NUM_ENTITIES], tf.float32),
            'z': tf.FixedLenFeature([MAX_NUM_ENTITIES], tf.float32),
            'pixel_coords': tf.FixedLenFeature([MAX_NUM_ENTITIES, 3], tf.float32),
            'rotation': tf.FixedLenFeature([MAX_NUM_ENTITIES], tf.float32),
            'size': tf.FixedLenFeature([MAX_NUM_ENTITIES], tf.string),
            'material': tf.FixedLenFeature([MAX_NUM_ENTITIES], tf.string),
            'shape': tf.FixedLenFeature([MAX_NUM_ENTITIES], tf.string),
            'color': tf.FixedLenFeature([MAX_NUM_ENTITIES], tf.string),
            'visibility': tf.FixedLenFeature([MAX_NUM_ENTITIES], tf.float32),
        }


        def _decode(example_proto):
            # Parse the input `tf.Example` proto using the feature description dict above.
            single_example = tf.parse_single_example(example_proto, features)
            for k in BYTE_FEATURES:
                single_example[k] = tf.squeeze(tf.decode_raw(single_example[k], tf.uint8),
                                            axis=-1)
            return single_example

        raw_dataset = tf.data.TFRecordDataset(
            tfrecords_path, compression_type=COMPRESSION_TYPE,
            buffer_size=read_buffer_size)
        return raw_dataset.map(_decode, num_parallel_calls=map_parallel_calls)

    def tetrom_dataset(tfrecords_path, read_buffer_size=None,
                map_parallel_calls=None):
        IMAGE_SIZE = [35, 35]
        # The maximum number of foreground and background entities in the provided
        # dataset. This corresponds to the number of segmentation masks returned per
        # scene.
        MAX_NUM_ENTITIES = 4

        # Create a dictionary mapping feature names to `tf.Example`-compatible
        # shape and data type descriptors.
        features = {
            'image': tf.FixedLenFeature(IMAGE_SIZE+[3], tf.string),
            'mask': tf.FixedLenFeature([MAX_NUM_ENTITIES]+IMAGE_SIZE+[1], tf.string),
            'x': tf.FixedLenFeature([MAX_NUM_ENTITIES], tf.float32),
            'y': tf.FixedLenFeature([MAX_NUM_ENTITIES], tf.float32),
            'shape': tf.FixedLenFeature([MAX_NUM_ENTITIES], tf.float32),
            'color': tf.FixedLenFeature([MAX_NUM_ENTITIES, 3], tf.float32),
            'visibility': tf.FixedLenFeature([MAX_NUM_ENTITIES], tf.float32),
        }


        def _decode(example_proto):
            # Parse the input `tf.Example` proto using the feature description dict above.
            single_example = tf.parse_single_example(example_proto, features)
            for k in BYTE_FEATURES:
                if k == 'image':
                    single_example[k] = tf.image.resize(tf.squeeze(tf.decode_raw(single_example[k], tf.uint8),
                                            axis=-1), (32,32))
                else:
                    single_example[k] = tf.squeeze(tf.decode_raw(single_example[k], tf.uint8),
                                            axis=-1)
            return single_example
        
        raw_dataset = tf.data.TFRecordDataset(
        tfrecords_path, compression_type=COMPRESSION_TYPE,
        buffer_size=read_buffer_size)
        return raw_dataset.map(_decode, num_parallel_calls=map_parallel_calls)


    def obects_dataset(tfrecords_path, dataset_variant, read_buffer_size=None,
                map_parallel_calls=None):
        """Read, decompress, and parse the TFRecords file.
        Args:
            tfrecords_path: str. Path to the dataset file.
            dataset_variant: str. One of ['train', 'six_objects', 'empty_room',
            'identical_color']. This is used to identify the maximum number of
            entities in each scene. If an incorrect identifier is passed in, the
            TFRecords file will not be read correctly.
            read_buffer_size: int. Number of bytes in the read buffer. See documentation
            for `tf.data.TFRecordDataset.__init__`.
            map_parallel_calls: int. Number of elements decoded asynchronously in
            parallel. See documentation for `tf.data.Dataset.map`.
        Returns:
            An unbatched `tf.data.TFRecordDataset`.
        """
        IMAGE_SIZE = [64, 64]
        # The maximum number of foreground and background entities in each variant
        # of the provided datasets. The values correspond to the number of
        # segmentation masks returned per scene.
        MAX_NUM_ENTITIES = {
            'train': 7,
            'six_objects': 10,
            'empty_room': 4,
            'identical_color': 10
        }
        def _decode(example_proto, features):
            # Parse the input `tf.Example` proto using a feature description dictionary.
            single_example = tf.parse_single_example(example_proto, features)
            for k in BYTE_FEATURES:
                single_example[k] = tf.squeeze(tf.decode_raw(single_example[k], tf.uint8),
                                            axis=-1)
            return single_example
        def feature_descriptions(max_num_entities):
            """Create a dictionary describing the dataset features.
            Args:
                max_num_entities: int. The maximum number of foreground and background
                entities in each image. This corresponds to the number of segmentation
                masks returned per scene.
            Returns:
                A dictionary which maps feature names to `tf.Example`-compatible shape and
                data type descriptors.
            """
            return {
                'image': tf.FixedLenFeature(IMAGE_SIZE+[3], tf.string),
                'mask': tf.FixedLenFeature([max_num_entities]+IMAGE_SIZE+[1], tf.string),
            }
        if dataset_variant not in MAX_NUM_ENTITIES:
            raise ValueError('Invalid `dataset_variant` provided. The supported values'
                            ' are: {}'.format(list(MAX_NUM_ENTITIES.keys())))
        max_num_entities = MAX_NUM_ENTITIES[dataset_variant]
        raw_dataset = tf.data.TFRecordDataset(
            tfrecords_path, compression_type=COMPRESSION_TYPE,
            buffer_size=read_buffer_size)
        features = feature_descriptions(max_num_entities)
        partial_decode_fn = functools.partial(_decode, features=features)
        return raw_dataset.map(partial_decode_fn,
                                num_parallel_calls=map_parallel_calls)


    class tf_record_data():
        def __init__(self, path, batch_size, shuffle, transform, name):
            # tfrecord_path = "../../guided-diffusion/datasets/objects_room/objects_room_train.tfrecords"
            self.name = name
            if name == "object_room":
                tfrecord_path = os.path.join(path.replace("object_room","objects_room"), "objects_room_train.tfrecords")
                row_dataset = obects_dataset(tfrecord_path, 'train')
                self.length = 1000000
            elif name == "tetrominoes":
                tfrecord_path = os.path.join(path, "tetrominoes_train.tfrecords")
                row_dataset = tetrom_dataset(tfrecord_path)
                self.length = 60000
            elif name == "clevr":
                tfrecord_path = os.path.join(path, "clevr_with_masks_train.tfrecords")
                row_dataset = clevr_dataset(tfrecord_path)
                self.length = 1000000
            else:
                raise NotImplementedError
            row_dataset = row_dataset.repeat(201)
            self.batch_size = batch_size
            if shuffle:
                row_dataset = row_dataset.shuffle(buffer_size = 100 * batch_size)
            row_dataset = row_dataset.prefetch(buffer_size = 10 * batch_size)
            batched_dataset = row_dataset.batch(batch_size)
            iterator = batched_dataset.make_one_shot_iterator()
            self.data = iterator.get_next()
            self.sess = tf.Session()
            self.transform = transform

        def __iter__(self):
            return self

        # def __next__(self):
        #     return self.transform(torch.from_numpy(self.sess.run(self.data)['image'])), None
        
        def __next__(self):
            data = self.sess.run(self.data)
            return data['image'], data['mask']
        
        def __len__(self):
            return int(self.length/self.batch_size)

except:
    pass

