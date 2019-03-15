import tensorflow as tf
import time
import socket
import sys
import src.util.codegenUtil as gen
import inspect
import importlib
import codecs
import json
from collections import OrderedDict
import threading
from src.util.ssh_handler import shell_handler, sftp_handler
import numpy as np
import os
from PIL import Image
import collections
import cPickle
import urllib
import tarfile
import zipfile
import pickle
import math
import glob
import re

from tensorflow.python.training.summary_io import SummaryWriterCache

class PTBreader(object):
    """
        Dataset reader class for PTB text dataset
        inputs:
            data_path: data directory
            batch_Size: batch size
            num_steps: the number of unrolled steps of recurrent cell
            istrain: whether train or test. Boolean
        """
    def __init__(self, data_path, batch_size, num_steps, istrain=True):
        self.index_in_epoch = 0
        self.epochs_completed = 0
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.istrain = istrain
        self.data, self.epoch_size = self._ptb_producer()
        self.label = []
        pass

    def _read_words(self, train_path):
        with tf.gfile.GFile(train_path, "r") as f:
            if sys.version_info[0] >= 3:
                return f.read().replace("\n", "<eos>").split()
            else:
                return f.read().decode("utf-8").replace("\n", "<eos>").split()

    def _build_vocab(self, train_path):
        data = self._read_words(train_path)

        counter = collections.Counter(data)
        count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

        words, _ = list(zip(*count_pairs))
        word_to_id = dict(zip(words, range(len(words))))

        return word_to_id

    def _file_to_word_ids(self, train_path, word_to_id):
        data = self._read_words(train_path)
        return [word_to_id[word] for word in data if word in word_to_id]

    def _ptb_raw_data(self):
        if self.istrain == True:
            train_path = os.path.join(self.data_path, "ptb.train.txt")
        else:
            train_path = os.path.join(self.data_path, "ptb.test.txt")
        word_to_id = self._build_vocab(train_path)
        raw_data = self._file_to_word_ids(train_path, word_to_id)

        vocabulary = len(word_to_id)
        return raw_data, vocabulary

    def _ptb_producer(self):
        raw_data, _ = self._ptb_raw_data()
        # TODO: need to check epoch size is positive
        # if epochs_size is negative, decrease batch_size or num_steps
        epoch_size = ((len(raw_data) // self.batch_size) - 1 ) // self.num_steps
        data_len = len(raw_data)
        batch_len = data_len // self.batch_size
        data = np.reshape(raw_data[0:self.batch_size * batch_len], [self.batch_size, batch_len])
        return data, epoch_size

    def get_epoch_size(self):
        return self.epoch_size

    def next_batch(self):
        # WARNING: PTB data inputs sequentially
        x = self.data[0:self.batch_size, self.index_in_epoch*self.num_steps:(self.index_in_epoch+1)*self.num_steps]
        y = np.roll(self.data, -1)[0:self.batch_size, self.index_in_epoch*self.num_steps:(self.index_in_epoch+1)*self.num_steps]
        self.index_in_epoch += 1
        if self.index_in_epoch == self.epoch_size - 1:
            self.index_in_epoch = 0
        return x, y

class JPEGreader(object):
    """
    Dataset reader class for JPEG files
    inputs:
        img_path: image data path
        label_path: labeling data path
        batch_size: batch size for reader
        image_size: image width/height
        shuffle: whether or not shuffling
        num_classes: number of class in dataset
        istrain: whether train or test.
    """
    def __init__(self, img_path, label_path, batch_size,image_size, shuffle, num_classes):
        self.index_in_epoch = 0
        self.epochs_completed = 0
        self.img_path = img_path
        self.label_path =  label_path
        self.batch_size =  batch_size
        self.image_size = image_size
        self.shuffle = shuffle
        self.num_classes = num_classes
        self.img_files, self._num_examples = self._set_img_files()

    def _is_grey_scale(self):
        """
        Check the image channels
        :return: whether geryscale or not
        """
        im_check = Image.open(self.img_path + '/' + self.img_files[0]).convert('RGB')
        w,h = im_check.size
        for i in range(w):
            for j in range(h):
                r,g,b = im_check.getpixel((i,j))
                if r != g != b: return False
        return True

    def _set_img_files(self):
        """
        set jpeg dataset in path
        :return: whole image filenames and the number of images
        """
        filenames = os.listdir(self.img_path)
        # remove label file if exist with dataset
        if filenames.count(os.path.basename(self.label_path)) != 0:
            filenames.remove(os.path.basename(self.label_path))
        for filename in filenames:
            full_filename = os.path.join(self.img_path, filename)
            ext = os.path.splitext(full_filename)[-1]
            if ext != '.jpg' or ext !='.jpeg':
                filenames.remove(filename)
        return filenames, len(filenames)

    def _img_read(self, batch_filenames):
        """
        read pixel data of batch images
        :param batch_filenames: image filenames in batch size
        :return: pixel data of batch_filenames
        """
        data = []
        is_grey = self._is_grey_scale()
        for file in batch_filenames:
            with Image.open(self.img_path + '/' + file) as im:
                im_resize = im.resize((self.image_size[0], self.image_size[1]))
                im_array = np.array(im_resize.getdata()).astype(np.uint8).reshape((im_resize.size[0], im_resize.size[1], 1 if is_grey == True else 3))
                data.append(im_array)
        return data

    def _label_read(self, batch_filenames):
        """
        read label of batch images using label file(.txt)
        :param batch_filenames: image filenames in batch size
        :return:
        """
        label = []
        for file in batch_filenames:
            with open(self.label_path) as fp:
                for line in fp:
                    if os.path.splitext(file)[0] in line:
                        # int convert is conflict?
                        label.append(line.split(' ')[-1].rstrip())
                        break
        return label

    def _dense_to_one_hot(self, labels):
        """
        make one-hot encoding. [N,1] to [N, NUM_CLASSES]
        :param labels: dense label [N, 1]
        :return: one-hot encoded [N, NUM_CLASSES]
        """
        num_labels = labels.shape[0]
        index_offset = np.arange(num_labels) * self.num_classes
        labels_one_hot = np.zeros((num_labels, self.num_classes))
        labels_one_hot.flat[index_offset + labels.ravel()] = 1
        return labels_one_hot

    def next_batch(self):
        """
        pop next batch among dataset
        :return: image batch, label batch
        """
        data = []
        label = []
        start = self.index_in_epoch
        if start == 0 and self.epochs_completed == 0:
            if self.shuffle:
                np.random.shuffle(self.img_files)  # shuffle indexs
            else: # not shuffle
                pass

        # go to the next batch
        if start + self.batch_size > self._num_examples:
            self.epochs_completed += 1
            rest_num_examples = self._num_examples - start
            files_rest_part = self.img_files[start:self._num_examples]
            np.random.shuffle(self.img_files)  # shuffle indexes

            start = 0
            self.index_in_epoch = self.batch_size - rest_num_examples #avoid the case where the #sample != integar times of batch_size
            end =  self.index_in_epoch
            files_new_part =  self.img_files[start:end]
            batch_filenames = np.concatenate((files_rest_part, files_new_part), axis=0)
        else:
            self.index_in_epoch += self.batch_size
            end = self.index_in_epoch
            batch_filenames = self.img_files[start:end]
        data = self._img_read(batch_filenames)
        label = self._label_read(batch_filenames)
        return np.asarray(data, dtype=np.float32)/255.0, self._dense_to_one_hot(np.asarray(label, dtype=np.int32))

class CIFAR10reader(object):
    """
        Dataset reader class for CIFAR10 daaset
        inputs:
            img_path: image data path
            batch_size: batch size for reader
            shuffle: whether or not shuffling
            istrain: whether train or test.
        """
    def __init__(self, img_path, batch_size, shuffle, istrain=1):
        self.index_in_epoch = 0
        self.epochs_completed = 0
        self.img_path = img_path
        self.batch_size = batch_size
        self.shuffle = shuffle
        self._num_examples = 50000 if istrain == 1 else 10000 # fixed
        self.option = istrain
        self.batch_idx = np.arange(0, self._num_examples)
        self.images, self.labels = self._get_cifar_data()

    def _get_cifar_data(self):
        """
        Generate cifar10 datasets
        :return
            x: images pixel. [ALLIMGAES, IMG_SIZE, IMG_SIZE, CHANNEL]
            dense_to_one_hot(y_fine, num_classes=num_fine_classes): fine labl. [ALLIMAGES, NUM_FINE_LABLES]
        """
        # prepare python dictionary to save datsets
        x = None
        y_fine = None

        # if datasets is not in the directory, download datasets first
        CIFAR10reader._maybe_download_and_extract(self.img_path)

        # variable setting
        meta_name = 'batches.meta'  # filename with information of datasets label name
        train_name = ['data_batch_' + str(num + 1) for num in
                      xrange(5)]  # filename with information of train images
        eval_name = ['test_batch']  # filename with information of test images
        num_fine_classes = 10  # number of fine labels
        fine_label_key = 'labels'  # python dictionary key to extract fine label

        folder_name = self.img_path
        metafile = os.path.join(folder_name, meta_name)

        # read meta file and extract label info.
        with open(metafile, 'rb') as f:
            datadict = cPickle.load(f)
        fine_label = datadict['label_names']

        # read train/test images and extract images pixel and labels
        if self.option == 1:
            for f_name in train_name:
                trainfile = os.path.join(folder_name, f_name)
                with open(trainfile, 'rb') as f:
                    datadict = cPickle.load(f)
                    _x = datadict.get("data")
                    _x = np.array(_x)
                    _x = _x.reshape([-1, 3, 32, 32])
                    _x = _x.transpose([0, 2, 3, 1])
                    _x = _x.reshape(-1, 32, 32, 3)

                    # prepare coarse/fine labels
                    _y_fine = np.array(datadict.get(fine_label_key))

                # combine all datasets in files
                if x is None:
                    x = _x
                    y_fine = _y_fine
                else:
                    x = np.concatenate((x, _x), axis=0)
                    y_fine = np.concatenate((y_fine, _y_fine), axis=0)

        elif self.option == 0:
            for f_name in eval_name:
                evalfile = os.path.join(folder_name, f_name)
                with open(evalfile, 'rb') as f:
                    # prepare images
                    datadict = cPickle.load(f)
                    _x = datadict.get("data")
                    _x = np.array(_x)
                    _x = _x.reshape([-1, 3, 32, 32])
                    _x = _x.transpose([0, 2, 3, 1])
                    x = _x.reshape(-1, 32, 32, 3)

                    # prepare coarse/fine labels
                    y_fine = np.array(datadict.get(fine_label_key))

        # make [ALLIMAGES,1] to [ALLIMAGES, NUM_CLASSES]
        def dense_to_one_hot(labels_dense, num_classes):
            if num_classes is 0:
                labels_one_hot = None
            else:
                num_labels = labels_dense.shape[0]
                index_offset = np.arange(num_labels) * num_classes
                labels_one_hot = np.zeros((num_labels, num_classes))
                labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
            return labels_one_hot

        return x, dense_to_one_hot(y_fine, num_classes=num_fine_classes)

    def next_batch(self):
        """
        pop next batch among dataset
        :return: image batch, label batch
        """
        start = self.index_in_epoch
        if start == 0 and self.epochs_completed == 0:
            if self.shuffle:
                np.random.shuffle(self.batch_idx)  # shuffle indexs
            else:  # not shuffle
                pass

        # go to the next batch
        if start + self.batch_size > self._num_examples:
            self.epochs_completed += 1
            rest_num_examples = self._num_examples - start
            rest_idx = self.batch_idx[start:self._num_examples]
            images_rest_part = self.images[rest_idx]
            labels_rest_part = self.labels[rest_idx]

            if self.shuffle:
                np.random.shuffle(self.batch_idx)  # shuffle index
            else:  # not shuffle
                pass

            start = 0
            self.index_in_epoch = self.batch_size - rest_num_examples  # avoid the case where the #sample != integer times of batch_size
            end = self.index_in_epoch
            idx = self.batch_idx[start:end]
            images_new_part = self.images[idx]
            labels_new_part = self.labels[idx]
            batched_x = np.concatenate((images_rest_part, images_new_part), axis=0)
            batched_y = np.concatenate((labels_rest_part, labels_new_part), axis=0)
        else:
            self.index_in_epoch += self.batch_size
            end = self.index_in_epoch
            idx = self.batch_idx[start:end]
            batched_x = self.images[idx]
            batched_y = self.labels[idx]
        return batched_x , batched_y

    # print download progress
    @staticmethod
    def _print_download_progress(count, block_size, total_size):
        pct_complete = float(count * block_size) / total_size
        msg = "\r- Download progress: {0:.1%}".format(pct_complete)
        sys.stdout.write(msg)
        sys.stdout.flush()

    # download cifar10/100 python dictionary and extract in directory
    @staticmethod
    def _maybe_download_and_extract(dir):
        main_directory = dir + '/'
        cifar_directory = dir + '/'
        if not os.path.exists(main_directory):
            os.makedirs(main_directory)

        if not os.path.exists(cifar_directory):
            os.makedirs(cifar_directory)
            url = "http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
            filename = url.split('/')[-1]
            file_path = os.path.join(main_directory, filename)
            zip_cifar = file_path
            file_path, _ = urllib.urlretrieve(url=url, filename=file_path,
                                              reporthook=CIFAR10reader._print_download_progress)

            print()
            print("Download finished. Extracting files.")
            if file_path.endswith(".zip"):
                zipfile.ZipFile(file=file_path, mode="r").extractall(main_directory)
            elif file_path.endswith((".tar.gz", ".tgz")):
                tarfile.open(name=file_path, mode="r:gz").extractall(main_directory)
            print("Done.")
            os.rename(main_directory + "./cifar-10-batches-py", cifar_directory)
            os.remove(zip_cifar)

class GLAUCOMAreader(object):
    #def __init__(self,data, labels):
    def __init__(self, data_path, batch_size, is_shuffle, images, labels, istrain=1):
        self.index_in_epoch = 0
        self.epochs_completed = 0
        self.data_path = data_path
        self.batch_size = batch_size
        self.istrain = istrain
        self.images =  images
        self.labels = labels
        self.num_examples = images.shape[0]
        self.shuffle = is_shuffle
        self.batch_idx = np.arange(self.num_examples)

    def _read_dataset(self):
        if self.istrain == 1:
            images = open(self.data_path + '/train_dataset.npy')
            labels = open(self.data_path + '/train_labels.npy')
        else:
            images = open(self.data_path + '/test_dataset.npy')
            labels = open(self.data_path + '/test_labels.npy')
        raw_images = np.load(images)
        raw_labels = np.load(labels)

        raw_labels_dense = np.empty((raw_labels.shape[0]), np.int32)
        for idx, col in enumerate(raw_labels):
                raw_labels_dense[idx] = 0 if col[0] == 1 else 1
        return raw_images, raw_labels_dense, raw_images.shape[0]

    def next_batch(self):
        start = self.index_in_epoch
        # go to the next batch
        if start == 0 and self.epochs_completed == 0:
            if self.shuffle:
                np.random.shuffle(self.batch_idx)  # shuffle indexs
            else:  # not shuffle
                pass

        if start + self.batch_size > self.num_examples:
            self.epochs_completed += 1
            rest_num_examples = self.num_examples - start
            rest_idx = self.batch_idx[start:self.num_examples]
            images_rest_part = self.images[rest_idx]
            labels_rest_part = self.labels[rest_idx]

            if self.shuffle:
                np.random.shuffle(self.batch_idx)  # shuffle indexs
            else:  # not shuffle
                pass
            start = 0
            self.index_in_epoch = self.batch_size - rest_num_examples  # avoid the case where the #sample != integar times of batch_size
            end = self.index_in_epoch
            idx = self.batch_idx[start:end]
            images_new_part = self.images[idx]
            labels_new_part = self.labels[idx]

            batched_x = np.concatenate((images_rest_part, images_new_part), axis=0)
            batched_y = np.concatenate((labels_rest_part, labels_new_part), axis=0)

        else:
            self.index_in_epoch += self.batch_size
            end = self.index_in_epoch
            idx = self.batch_idx[start:end]
            batched_x = self.images[idx]
            batched_y = self.labels[idx]

        return batched_x, batched_y

class BODYDATAreader(object):
    def __init__(self, pkl_path, is_shuffle):
        self.index_in_epoch = 0
        self.epochs_completed = 0
        self.data_path = pkl_path
        self.shuffle = is_shuffle
        self.train_image, self.train_type_label, self.valid_image, self.valid_type_label, self.test_image, self.test_type_label, self.input_avg, self.input_std, self.output_avg, self.output_std = self._get_data()

    def _get_data(self):
        # TODO: FIX RATIO
        test_ratio = 0.2
        valid_ratio = 0.1
        train_ratio = 0.7

        with open(self.data_path + '/data.pkl', 'rb') as f:
            data = pickle.load(f)
        im = data['image']
        im = im / 255.

        # normalize
        input_avg = np.mean(data['input'], 0)
        output_avg = np.mean(data['output'], 0)
        input_std = np.std(data['input'], 0)
        output_std = np.std(data['output'], 0)
        data['input'] = (data['input'] - input_avg) / input_std
        data['output'] = (data['output'] - output_avg) / output_std

        with open(self.data_path + '/normalize_value.pkl', 'wb') as f:
            pickle.dump([input_avg, output_avg, input_std, output_std], f)

        type_label = np.concatenate([data['input'], data['output'], data['class']], axis=1)
        # type_label = np.concatenate([data['input'],data['output'],data['class']],axis=0)
        total_num = im.shape[0]
        #print("total_num of im ", total_num)
        train_num = int(math.floor(total_num * train_ratio))
        valid_num = int(math.floor(total_num * valid_ratio))

        random_idx = np.random.permutation(total_num)
        train_idx = random_idx[:train_num]
        valid_idx = random_idx[train_num:train_num + valid_num]
        test_idx = random_idx[train_num + valid_num:]

        train_im = im[train_idx]
        train_type_label = type_label[train_idx]
        valid_im = im[valid_idx]
        valid_type_label = type_label[valid_idx]
        test_im = im[test_idx]
        test_type_label = type_label[test_idx]

        return train_im, train_type_label, valid_im, valid_type_label, test_im, test_type_label, input_avg, input_std, output_avg, output_std

    def get_output_std(self):
        return self.output_std

    def get_output_avg(self):
        return self.output_avg

    def next_batch(self, batch_size, is_train=True):

        im = self.train_image if is_train else self.test_image
        data = self.train_type_label if is_train else self.test_type_label

        N = data.shape[0]
        idx = np.random.choice(N, batch_size, replace=False)
        im_batch = im[idx, :, :, :]
        input_batch = data[idx, :14]
        output_batch = data[idx, 14:23]
        #  output_batch = data[idx,15:23]
        BT_batch = data[idx, -2]
        BS_batch = data[idx, -1]

        BT_batch = np.asarray(BT_batch, dtype=np.int32)
        BS_batch = np.asarray(BS_batch, dtype=np.int32)
        # make [ALLIMAGES,1] to [ALLIMAGES, NUM_CLASSES]
        def dense_to_one_hot(labels_dense, num_classes):
            if num_classes is 0:
                labels_one_hot = None
            else:
                num_labels = labels_dense.shape[0]
                index_offset = np.arange(num_labels) * num_classes
                labels_one_hot = np.zeros((num_labels, num_classes))
                labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
            return labels_one_hot

        return im_batch, input_batch, output_batch, dense_to_one_hot(BT_batch, 3), dense_to_one_hot(BS_batch, 11)

class MNISTreader(object):
    """
        Dataset reader class for MNIST daaset
        inputs:
            img_path: image data path
            batch_size: batch size for reader
            shuffle: whether or not shuffling
            istrain: whether train or test.
        """
    def __init__(self, img_path, batch_size, shuffle, istrain=1):
        self.index_in_epoch = 0
        self.epochs_completed = 0
        self.img_path = img_path
        self.batch_size = batch_size
        self.shuffle = shuffle
        self._num_examples = 60000 if istrain == 1 else 10000 # fixed
        self.option = istrain
        self.batch_idx = np.arange(0, self._num_examples)
        self.images, self.labels = self._get_mnist_data()

    def _get_mnist_data(self):
        """
        Generate mnist datasets
        :return
            x: images pixel. [ALLIMGAES, IMG_SIZE, IMG_SIZE]
            dense_to_one_hot(y_fine, num_classes=num_fine_classes): fine labl. [ALLIMAGES, NUM_FINE_LABLES]
        """
        # prepare python dictionary to save datsets
        x = None
        y_fine = None
        num_fine_classes = 10

        #WARNING: follow variables do not need - keras API only allow dataset file located in ~/.keras/dataset
        npz_name= 'mnist.npz'
        folder_name = self.img_path
        npzfile = os.path.join(folder_name, npz_name)

        (x_train, y_train), (x_test, y_test) = tf.contrib.keras.datasets.mnist.load_data()
        # read train/test images and extract images pixel and labels
        if self.option == 1:
            x = x_train / 255.0
            y_fine = y_train
        elif self.option == 0:
            x = x_test / 255.0
            y_fine = y_test

        # make [ALLIMAGES,1] to [ALLIMAGES, NUM_CLASSES]
        def dense_to_one_hot(labels_dense, num_classes):
            if num_classes is 0:
                labels_one_hot = None
            else:
                num_labels = labels_dense.shape[0]
                index_offset = np.arange(num_labels) * num_classes
                labels_one_hot = np.zeros((num_labels, num_classes))
                labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
            return labels_one_hot

        return x, dense_to_one_hot(y_fine, num_classes=num_fine_classes)

    def next_batch(self):
        """
        pop next batch among dataset
        :return: image batch, label batch
        """
        start = self.index_in_epoch
        if start == 0 and self.epochs_completed == 0:
            if self.shuffle:
                np.random.shuffle(self.batch_idx)  # shuffle indexs
            else:  # not shuffle
                pass

        # go to the next batch
        if start + self.batch_size > self._num_examples:
            self.epochs_completed += 1
            rest_num_examples = self._num_examples - start
            rest_idx = self.batch_idx[start:self._num_examples]
            images_rest_part = self.images[rest_idx]
            labels_rest_part = self.labels[rest_idx]

            if self.shuffle:
                np.random.shuffle(self.batch_idx)  # shuffle indexs
            else:  # not shuffle
                pass

            start = 0
            self.index_in_epoch = self.batch_size - rest_num_examples  # avoid the case where the #sample != integar times of batch_size
            end = self.index_in_epoch
            idx = self.batch_idx[start:end]
            images_new_part = self.images[idx]
            labels_new_part = self.labels[idx]
            batched_x = np.concatenate((images_rest_part, images_new_part), axis=0)
            batched_y = np.concatenate((labels_rest_part, labels_new_part), axis=0)
        else:
            self.index_in_epoch += self.batch_size
            end = self.index_in_epoch
            idx = self.batch_idx[start:end]
            batched_x = self.images[idx]
            batched_y = self.labels[idx]
        # make [batch_size, 28, 28] images to [batch_size, 28, 28, 1] dimensions
        batched_x = np.reshape(batched_x, [-1, 28, 28, 1])
        return batched_x , batched_y

    # print download progress
    @staticmethod
    def _print_download_progress(count, block_size, total_size):
        pct_complete = float(count * block_size) / total_size
        msg = "\r- Download progress: {0:.1%}".format(pct_complete)
        sys.stdout.write(msg)
        sys.stdout.flush()

    # download cifar10/100 python dictionary and extract in directory
    @staticmethod
    def _maybe_download_and_extract(dir):
        main_directory = dir + '/'
        cifar_directory = dir + '/'
        if not os.path.exists(main_directory):
            os.makedirs(main_directory)

        if not os.path.exists(cifar_directory):
            os.makedirs(cifar_directory)
            url = "http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
            filename = url.split('/')[-1]
            file_path = os.path.join(main_directory, filename)
            zip_cifar = file_path
            file_path, _ = urllib.urlretrieve(url=url, filename=file_path,
                                              reporthook=CIFAR10reader._print_download_progress)

            print()
            print("Download finished. Extracting files.")
            if file_path.endswith(".zip"):
                zipfile.ZipFile(file=file_path, mode="r").extractall(main_directory)
            elif file_path.endswith((".tar.gz", ".tgz")):
                tarfile.open(name=file_path, mode="r:gz").extractall(main_directory)
            print("Done.")
            os.rename(main_directory + "./cifar-10-batches-py", cifar_directory)
            os.remove(zip_cifar)

class IMAGENETreader(object):
    """
    Dataset reader class for imagenet files with tf.Record
    inputs:

    """
    def __init__(self, data_dir, batch_size, shuffle, is_train):
        self.index_in_epoch = 0
        self.epochs_completed = 0
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.is_train = is_train
        self.num_preprocess_threads = 4 # Number of preprocessing threads per tower. Please make this a multiple of 4
        self.num_readers = 4 # Number of parallel readers during train
        self.input_queue_memory_factor = 16 # Size of the queue of preprocessed images.
        self.height = self.width = 224
        self.depth = 3

    def eval_image(self, image):
        """Prepare one image for evaluation.
          Args:
            image: 3-D float Tensor
            height: integer
            width: integer
            scope: Optional scope for name_scope.
          Returns:
            3-D float Tensor of prepared image.
          """
        with tf.name_scope(values=[image, self.height, self.width], name='eval_image'):
            # Crop the central region of the image with an area containing 87.5% of
            # the original image.
            image = tf.image.central_crop(image, central_fraction=0.875)

            # Resize the image to the original height and width.
            image = tf.expand_dims(image, 0)
            image = tf.image.resize_bilinear(image, [self.height, self.width],
                                             align_corners=False)
            image = tf.squeeze(image, [0])
            return image

    def data_files(self):
        """Returns a python list of all (sharded) data subset files.
            Returns:
              python list of all (sharded) data set files.
            Raises:
              ValueError: if there are not data_files matching the subset.
        """
        if self.is_train == 1:
            str_file = 'train'
        else:
            str_file = 'validation'
        tf_record_pattern = os.path.join(self.data_dir, '%s-*' % str_file)
        data_files = tf.gfile.Glob(tf_record_pattern)
        if not data_files:
            raise ValueError('No files found for dataset in %s' % self.data_dir)

        return data_files

    def decode_jpeg(self, image_buffer):
        """Decode a JPEG string into one 3-D float image Tensor.
          Args:
            image_buffer: scalar string Tensor.
          Returns:
            3-D float Tensor with values ranging from [0, 1).
          """
        with tf.name_scope(values=[image_buffer], name='decode_jpeg'):
            # Decode the string as an RGB JPEG.
            # Note that the resulting image contains an unknown height and width
            # that is set dynamically by decode_jpeg. In other words, the height
            # and width of image is unknown at compile-time.
            image = tf.image.decode_jpeg(image_buffer, channels=3)

            # After this point, all image pixels reside in [0,1)
            # until the very end, when they're rescaled to (-1, 1).  The various
            # adjust_* ops all require this range for dtype float.
            image = tf.image.convert_image_dtype(image, dtype=tf.float32)
            return image

    def distorted_image(self, image, bbox, thread_id=0):
        '''
        WARNING: current version only resize image, not augmentation.
        TODO: image augmentation func.
        :param image:
        :param bbox:
        :param thread_id:
        :return:
        '''
        # A large fraction of image datasets contain a human-annotated bounding
        # box delineating the region of the image containing the object of interest.
        # We choose to create a new bounding box for the object which is a randomly
        # distorted version of the human-annotated bounding box that obeys an allowed
        # range of aspect ratios, sizes and overlap with the human-annotated
        # bounding box. If no box is supplied, then we assume the bounding box is
        # the entire image.
        sample_distorted_bounding_box = tf.image.sample_distorted_bounding_box(
            tf.shape(image),
            bounding_boxes=bbox,
            min_object_covered=0.1,
            aspect_ratio_range=[0.75, 1.33],
            area_range=[0.05, 1.0],
            max_attempts=100,
            use_image_if_no_bounding_boxes=True)
        bbox_begin, bbox_size, distort_bbox = sample_distorted_bounding_box

        # Crop the image to the specified bounding box.
        distorted_image = tf.slice(image, bbox_begin, bbox_size)

        # This resizing operation may distort the images because the aspect
        # ratio is not respected. We select a resize method in a round robin
        # fashion based on the thread number.
        # Note that ResizeMethod contains 4 enumerated resizing methods.
        resize_method = thread_id % 4
        distorted_image = tf.image.resize_images(distorted_image, [self.height, self.width],
                                                 method=resize_method)
        # Restore the shape since the dynamic slice based upon the bbox_size loses
        # the third dimension.
        distorted_image.set_shape([self.height, self.width, 3])
        return distorted_image

    def image_preprocessing(self, image_buffer, bbox, thread_id=0):
        if bbox is None:
            raise ValueError('Please supply a bounding box.')

        image = self.decode_jpeg(image_buffer)
        if self.is_train:
            image = self.distorted_image(image, bbox, thread_id)
        else:
            image = self.eval_image(image)

        # Finally, rescale to [-1,1] instead of [0, 1)
        image = tf.subtract(image, 0.5)
        image = tf.multiply(image, 2.0)

        return image

    def parse_example_proto(self, example_serialized):
        '''Parses an Example proto containing a training example of an image.
          The output of the build_image_data.py image preprocessing script is a dataset
          containing serialized Example protocol buffers. Each Example proto contains
          the following fields:
            image/height: 462
            image/width: 581
            image/colorspace: 'RGB'
            image/channels: 3
            image/class/label: 615
            image/class/synset: 'n03623198'
            image/class/text: 'knee pad'
            image/object/bbox/xmin: 0.1
            image/object/bbox/xmax: 0.9
            image/object/bbox/ymin: 0.2
            image/object/bbox/ymax: 0.6
            image/object/bbox/label: 615
            image/format: 'JPEG'
            image/filename: 'ILSVRC2012_val_00041207.JPEG'
            image/encoded: <JPEG encoded string>
        '''
        # Dense features in Example proto.
        feature_map = {
            'image/encoded': tf.FixedLenFeature([], dtype=tf.string,
                                                default_value=''),
            'image/class/label': tf.FixedLenFeature([1], dtype=tf.int64,
                                                    default_value=-1),
            'image/class/text': tf.FixedLenFeature([], dtype=tf.string,
                                                   default_value=''),
        }
        sparse_float32 = tf.VarLenFeature(dtype=tf.float32)
        # Sparse features in Example proto.
        feature_map.update(
            {k: sparse_float32 for k in ['image/object/bbox/xmin',
                                         'image/object/bbox/ymin',
                                         'image/object/bbox/xmax',
                                         'image/object/bbox/ymax']})

        features = tf.parse_single_example(example_serialized, feature_map)
        label = tf.cast(features['image/class/label'], dtype=tf.int32)

        xmin = tf.expand_dims(features['image/object/bbox/xmin'].values, 0)
        ymin = tf.expand_dims(features['image/object/bbox/ymin'].values, 0)
        xmax = tf.expand_dims(features['image/object/bbox/xmax'].values, 0)
        ymax = tf.expand_dims(features['image/object/bbox/ymax'].values, 0)

        # Note that we impose an ordering of (y, x) just to make life difficult.
        bbox = tf.concat(axis=0, values=[ymin, xmin, ymax, xmax])

        # Force the variable number of bounding boxes into the shape
        # [1, num_boxes, coords].
        bbox = tf.expand_dims(bbox, 0)
        bbox = tf.transpose(bbox, [0, 2, 1])

        return features['image/encoded'], label, bbox, features['image/class/text']

    def batch_inputs(self):
        """
        pop next batch among dataset
        :return: image batch, label batch
        """

        with tf.name_scope('batch_processing'):
            data_files = self.data_files()
            if data_files is None:
                raise ValueError('No data files found for imagenet dataset')

        # create filename_queue
        if self.is_train:
            filename_queue = tf.train.string_input_producer(data_files,
                                                            shuffle=self.shuffle,
                                                            capacity=16)
        else:
            filename_queue = tf.train.string_input_producer(data_files,
                                                            shuffle=False,
                                                            capacity=1)

        examples_per_shard = 1024
        min_queue_examples = examples_per_shard * self.input_queue_memory_factor
        if self.is_train:
            examples_queue = tf.RandomShuffleQueue(
                capacity=min_queue_examples + 3 * self.batch_size,
                min_after_dequeue=min_queue_examples,
                dtypes=[tf.string])
        else:
            examples_queue = tf.FIFOQueue(
                capacity=examples_per_shard + 3 * self.batch_size,
                dtypes=[tf.string])

        # Create multiple readers to populate the queue of examples.
        if self.num_readers > 1:
            enqueue_ops = []
            for _ in range(self.num_readers):
                reader = tf.TFRecordReader()
                _, value = reader.read(filename_queue)
                enqueue_ops.append(examples_queue.enqueue([value]))
            tf.train.queue_runner.add_queue_runner(
                tf.train.queue_runner.QueueRunner(examples_queue, enqueue_ops))
            example_serialized = examples_queue.dequeue()
        else:
            reader = tf.TFRecordReader()
            _, example_serialized = reader.read(filename_queue)

        images_and_labels = []
        for thread_id in range(self.num_preprocess_threads):
            # Parse a serialized Example proto to extract the image and metadata.
            image_buffer, label_index, bbox, _ = self.parse_example_proto(example_serialized)
            image = self.image_preprocessing(image_buffer, bbox, thread_id)
            images_and_labels.append([image, label_index])

        images, label_index_batch = tf.train.batch_join(images_and_labels,
                                                        batch_size=self.batch_size,
                                                        capacity=2 * self.num_preprocess_threads * self.batch_size)

        images = tf.cast(images, tf.float32)
        images = tf.reshape(images, shape=[self.batch_size, self.height, self.width, self.depth])

        return images, tf.reshape(label_index_batch, [self.batch_size])

    def next_batch(self):
        """Generate batches of distorted versions of ImageNet images.
          Use this function as the inputs for training a network.
          Distorting images provides a useful technique for augmenting the data
          set during training in order to make the network invariant to aspects
          of the image that do not effect the label.
          Args:
            dataset: instance of Dataset class specifying the dataset.
            batch_size: integer, number of examples in batch
            num_preprocess_threads: integer, total number of preprocessing threads but
              None defaults to FLAGS.num_preprocess_threads.
          Returns:
            images: Images. 4D tensor of size [batch_size, FLAGS.image_size,
                                               FLAGS.image_size, 3].
            labels: 1-D integer Tensor of [batch_size].
          """
        with tf.device('/cpu:0'):
            images, labels = self.batch_inputs()
        return images, labels - 1

# TODO: implement
class TEXTreader(object):
    def __init__(self):
        pass
    def next_batch(self):
        pass

# TODO: implement
class CSVreader(object):
    def __init__(self):
        pass
    def next_batch(self):
        pass

# TODO: implement
class TFRECORDreader(object):
    def __init__(self):
        pass
    def next_batch(self):
        pass

class tf_adaptor:
    #TODO: reader class input dynamically in script
    @staticmethod
    def compile(UDLI):
        network = ''

        for l in UDLI.network.parser.get_network():
            template = """self.Layer(type={type}, name={name}, inputs={inputs}, outputs={outputs}, attributes={attributes}, device={device})"""
            network += gen.indentNL(template.format(type=tf_adaptor.t(l['type']), name=tf_adaptor.t(l['name']),
                                                    inputs=tf_adaptor.t(l['inputs']),
                                                    outputs=tf_adaptor.t(l['outputs']),
                                                    attributes=tf_adaptor.t(l['attributes']),
                                                    device=tf_adaptor.t(l['device'])), 2)

        learning_option = ''
        for key, value in UDLI.learning_option.iteritems():
            template = """self.Learning_Option("{key}", {value})"""
            learning_option += gen.indentNL(template.format(key=key, value=tf_adaptor.t(value)), 2)

        cluster = ''
        for c in UDLI.cluster.cluster:
            template = """self.Cluster(name={name}, ip={ip}, type={type}, task={task})"""
            cluster += gen.indentNL(template.format(name=tf_adaptor.t(c['name']), ip=tf_adaptor.t(c['ip']),
                                                    type=tf_adaptor.t(c['type']), task=tf_adaptor.t(c['task'])), 2)

        src = {
            'src.DLMDL': {
                'cluster_manager': ['cluster_manager'],
                'DLNetwork': ['DLNetwork'],
                'inout': ['InOut', 'In', 'Out'],
                'layer': ['Layer'],
                'tf_adaptor': ['tf_adaptor', 'JPEGreader', 'PTBreader', 'CIFAR10reader', 'GLAUCOMAreader', 'BODYDATAreader', 'MNISTreader', 'IMAGENETreader']
            },
            'src.DLMDL.LayerOperation': {
                'layer_operation': ['LayerOperationAdaptor', 'LayerOperation']
            },
            'src.DLMDL.LayerOperation.tf': {
                'accuracy': ['op_tf_accuracy'],
                'adadelta': ['op_tf_adadelta'],
                'adagrad': ['op_tf_adagrad'],
                'adam': ['op_tf_adam'],
                'batchnorm': ['op_tf_batchnorm'],
                'bodydata_input': ['op_tf_bodydata_input'],
                'cifar10_input': ['op_tf_cifar10_input'],
                'imagenet_input': ['op_tf_imagenet_input'],
                'concat': ['op_tf_concat'],
                'conv': ['op_tf_conv'],
                'csv_input': ['op_tf_csv_input'],
                'database_input': ['op_tf_database_input'],
                'dropout': ['op_tf_dropout'],
                'dropout_wrapper': ['op_tf_dropout_wrapper'],
                'eltwise': ['op_tf_eltwise'],
                'elu': ['op_tf_elu'],
                'fc': ['op_tf_fc'],
                'glaucoma_input': ['op_tf_glaucoma_input'],
                'gru': ['op_tf_gru'],
                'initializer': ['get_initializer'],
                'jpeg_input': ['op_tf_jpeg_input'],
                'lr_scheduler': ['get_lr_scheduler'],
                'lrn': ['op_tf_lrn'],
                'lstm': ['op_tf_lstm'],
                'meansquareloss': ['op_tf_meansquareloss'],
                'mnist_input': ['op_tf_mnist_input'],
                'momentum': ['op_tf_momentum'],
                'multi_cells': ['op_tf_multi_cells'],
                'nesterov': ['op_tf_nesterov'],
                'perplexity': ['op_tf_perplexity'],
                'pooling': ['op_tf_pooling'],
                'prelu': ['op_tf_prelu'],
                'ptb_input': ['op_tf_ptb_input'],
                'reduction': ['op_tf_reduction'],
                'regularizer': ['get_regularizer'],
                'relu': ['op_tf_relu'],
                'reshape': ['op_tf_reshape'],
                'rmsprop': ['op_tf_rmsprop'],
                'rnn': ['op_tf_rnn'],
                'sequenceloss': ['op_tf_sequenceloss'],
                'sgd': ['op_tf_sgd'],
                'sigmoid': ['op_tf_sigmoid'],
                'sigmoidcrossentropyloss': ['op_tf_sigmoidcrossentropyloss'],
                'slice': ['op_tf_slice'],
                'softmaxwithloss': ['op_tf_softmaxwithloss'],
                'split': ['op_tf_split'],
                'static_rnn': ['op_tf_static_rnn'],
                'text_input': ['op_tf_text_input']
            },
            'src.Serializer': {
                'serializer': ['Serializer', 'LayerResolver', 'JsonLayerResolver']
            },
            'src.util': {
                'ssh_handler': ['shell_handler', 'sftp_handler']
            }
        }

        src = OrderedDict(sorted(src.items(), key=lambda t: t[0]))

        lib = ''
        for package, pys in src.iteritems():
            for py, modules in pys.iteritems():
                for module in modules:
                    print package + ' ' + py + ' ' + module
                    if py in ['initializer', 'regularizer', 'lr_scheduler']:
                        lib += inspect.getsource(getattr(importlib.import_module(
                            '{package}.{py}'.format(package=package, py=py)), module)) + '\n\n'
                    else:
                        lib += inspect.getsource(getattr(importlib.import_module(
                            '{package}.{py}'.format(package=package, py=py),
                            package=__package__), module)) + '\n\n'
        lib = lib.decode('utf-8')

        with codecs.open(UDLI.getFullPath(UDLI.getArgs('compile_out'), 'py'), 'w', encoding='utf-8') as fo:
            with codecs.open('src/util/UDLI_Script.template', 'r', encoding='utf-8') as ft:
                str = ft.read()
                str = str.replace('{network}', network)
                str = str.replace('{learning_option}', learning_option)
                str = str.replace('{cluster}', cluster)
                str = str.replace('{lib}', lib).replace('{framework}', 'tf')
                str = str.replace('{args}', tf_adaptor.t(
                    {'input': UDLI.getArgs('input'), 'log_out': UDLI.getArgs('log_out'),
                     'parameter_out': UDLI.getArgs('parameter_out')}))

                fo.write(str)

    @staticmethod
    def run(UDLI):
        print(sys.path)

        cluster = UDLI.cluster  # DLMDL cluster info.
        learning_option = UDLI.learning_option  # DLMDL learning option info.
        network = UDLI.network # DLMDL NN info.
        # TODO: multi parameter server will be considered.
        if learning_option.get('parallel') == 'DP': # data parallelism TF
            # divide cluster info. into parameter server and workers
            # WARNING: there is only one parameter server
            ps_hosts = [cluster.get('hosts').pop(cluster.getIndex('tasks', 'ps'))]
            worker_hosts = cluster.get('hosts')
            tf_cluster = tf.train.ClusterSpec({'ps':ps_hosts, 'worker':worker_hosts})
            current_address = tf_adaptor.resolve_address() # get address of this nodes
            if current_address == ps_hosts[0].split(':')[0]: # if this node is ps, running parameter server
                ps_idx = 0 # assume only one parameter server
                server = tf.train.Server(tf_cluster, job_name='ps', task_index=ps_idx) # in-process TF server
                ckpt_dir = learning_option.get('checkpoint_path')
                # create checkpoint directory if there isn't 'ckpt_dir'
                if ckpt_dir is not None:
                    if not os.path.exists(ckpt_dir):
                        os.makedirs(ckpt_dir)
                print('[DLMDL] grpc_server running')
                server.join()

            else: # if this node is worker
                worker_idx = tf_adaptor.get_node_index(worker_hosts)  # get worker index
                server = tf.train.Server(tf_cluster, job_name='worker',
                                         task_index=worker_idx)  # in-process TF server
                with tf.device(tf.train.replica_device_setter(worker_device='/job:worker/task:{0}'.format(worker_idx),
                                                              cluster=tf_cluster)):
                    tf_adaptor.worker_executor(UDLI, cluster, server.target, learning_option, network, worker_idx)
        elif learning_option.get('parallel') == 'MP' or learning_option.get('parallel') is None: # model parallelism or single node execution
            # divide cluster info. into workers
            worker_hosts = cluster.get('hosts')
            worker_idx = tf_adaptor.get_node_index(worker_hosts)  # get worker index
            tf_cluster = tf.train.ClusterSpec({'worker': worker_hosts})
            server = tf.train.Server(tf_cluster, job_name='worker',
                                     task_index=worker_idx)  # in-process TF server
            # test
            if worker_idx == 0:
                grpc_host = 'grpc://' + cluster.get('hosts')[0]
                tf_adaptor.worker_executor(UDLI, cluster, grpc_host, learning_option, network, worker_idx)
            else:
                tf_adaptor.grpc_server(server)

        elif learning_option.get('parallel') == 'DP_mb': # use memory box - ETRI
            # divide cluster info. into workers
            worker_hosts = cluster.get('hosts')
            worker_idx = tf_adaptor.get_node_index(worker_hosts)  # get worker index
            tf_cluster = tf.train.ClusterSpec({'worker': worker_hosts})
            server = tf.train.Server(tf_cluster, job_name='worker',
                                     task_index=worker_idx)  # in-process TF server
            tf_adaptor.worker_executor_mb(UDLI, cluster, server.target, learning_option, network, worker_idx)
        print("[DLMDL] Training Finished!")

    @staticmethod
    def worker_executor(UDLI, cluster, server, learning_option, network, worker_idx):
        current_address = tf_adaptor.resolve_address()  # get address of this node
        print('[DLMDL] worker ip: {0}, worker index: {1}'.format(current_address, worker_idx))


        #network.run(learning_option, cluster)  # bulid TF model

        # WARNING: parameter server must be last number which is bottom worker in .dlmdl cluster
        #          In release version, ssh handler don't include
        #conns = tf_adaptor.launch_slaves(UDLI, cluster.get('ips')[1:])

        # get layers: assume input/loss/optimizer must exist, but accuracy layer is optional
        input_layer = network.get_layer_by_type('input$').pop()  # get input layer
        print('[DLMDL] datasets({0}) training'.format(input_layer.type))
        loss_layer = network.get_layer_by_type('loss$').pop()  # get loss layer
        print('[DLMDL] using {0} as cost of model'.format(loss_layer.type))
        opt_layer = network.get_layer_by_type('adam|sgd|momentum|rmsprop|nesterov|adadelta|adagrad').pop()  # get optimizer layer
        print('[DLMDL] using {0} as optimizer of model'.format(opt_layer.type))
        global_step = tf.train.get_or_create_global_step()  # get global step

        # TODO: error handling in learning option
        print('[DLMDL] learning option parsing...')
        # options related to training
        train_dir = learning_option.get('data_path')
        train_iter = learning_option.get('iteration')
        train_bs = learning_option.get('batch_size')
        train_dp = learning_option.get('train_display')

        # options related to test
        test_interval = learning_option.get('test_interval')
        test_dir = learning_option.get('test_data_path')
        test_iter = learning_option.get('test_iteration')
        test_bs = learning_option.get('test_batch_size')

        # options related to checkpoint
        ckpt_interval = learning_option.get('checkpoint_interval')
        ckpt_dir = learning_option.get('checkpoint_path')

        # if option is restore, check the existing checkpoint file
        if learning_option.get('option') == 'RETRAIN':
            if len(glob.glob(ckpt_dir + '/*.meta')) == 0:
                raise Exception('[DLMDL ERROR] checkpoint file(.meta) do not exist in {0}'.format(ckpt_dir))

        # TF summary merge op. and session configuration
        config = tf.ConfigProto()  # ocnfiguration object for session
        config.gpu_options.allow_growth = True  # allocate minimum GPU memory for running
        config.log_device_placement = False  # not loggging ops/vars-devices mapping
        config.allow_soft_placement = True  # automatically unallocated ops/vars to device

        # step, summary, checkpoint hook setting
        summarydir = UDLI.getArgs('log_out')
        stophooks = tf.train.StopAtStepHook(last_step=train_iter)

        if input_layer.type == 'jpeg_input':  # for jpeg_input
            network.run(learning_option, cluster)  # bulid TF model
            # get operations
            is_train = learning_option.get('is_train')  # placeholder to set procedure
            # get loss op.
            loss_op = loss_layer.get_op_output('output')  # output op of loss layer is only 'loss'
            # get opt. op.
            train_op = opt_layer.get_op_output('output')  # output op of opt. layer is only 'output'

            # get accuracy layer and operation
            # WARNING: current accuracy related layers -> accuracy/top-k layer, perplexity layer
            acc_layer = network.get_layer_by_type('accuracy|perplexity')  # get acc layer
            if acc_layer == []:  # no acc op
                print('[DLMDL] no accuracy related layer in model')
                acc_op = None
            else:
                acc_layer = network.get_layer_by_type('accuracy|perplexity').pop()
                print('[DLMDL] using {0} as accuracy of model'.format(acc_layer.type))
                acc_op = acc_layer.get_op_output('output')  # output op of accuracy layer is only 'output'

            # get input placeholders.
            batch_x = input_layer.get_op_output('image')  # image/data/text/... (data info.)
            batch_y = input_layer.get_op_output('label')  # labels/targets/... (label info.)

            # print format
            format_str = '[{0:>6} Steps] Train Loss:{1:>8.3f}({2:>8.3f} sec/batch)'
            format_str_acc = '[{0:>6} Steps] Train Loss:{1:>8.3f}, Train {2}:{3:>8.3%}({4:>8.3f} sec/batch)'
            format_str_test = '[{0:>6} Steps] Average Test {1}:{2:>8.3%}'

            # get option attributes
            image_size = learning_option.get('image_size')
            is_shuffle = learning_option.get('shuffle')
            num_classes = learning_option.get('num_class')

            # get reader class for training data
            train_label_dir = learning_option.get('label_path')
            train_reader = JPEGreader(train_dir, train_label_dir, train_bs,
                                      image_size, is_shuffle, num_classes)

            if test_dir is None:  # if no test dataset
                test_reader = None
            else:  # get reader class for testing data
                test_label_dir = learning_option.get('test_label_path')
                test_reader = JPEGreader(test_dir, test_label_dir, test_bs,
                                         image_size, is_shuffle, num_classes)

            # if option is restore, check the existing checkpoint file
            if learning_option.get('option') == 'RETRAIN':
                if len(glob.glob(ckpt_dir + '/*.meta')) == 0:
                    raise Exception('[DLMDL ERROR] checkpoint file(.meta) do not exist in {0}'.format(ckpt_dir))

            # open TF session
            with tf.train.MonitoredTrainingSession(master=server, is_chief=(worker_idx == 0),
                                                   hooks=[stophooks],
                                                   save_checkpoint_steps=ckpt_interval,
                                                   checkpoint_dir=ckpt_dir,
                                                   save_summaries_steps=train_dp,
                                                   log_step_count_steps=0,
                                                   summary_dir=summarydir,
                                                   config=config) as sess:
                summary_writer = SummaryWriterCache.get(summarydir)
                while not sess.should_stop():
                    start_time = time.time()
                    # one step training
                    batched_x, batched_y = train_reader.next_batch()
                    _, step = sess.run([train_op, global_step],
                                       feed_dict={batch_x: batched_x, batch_y: batched_y})
                    if step % train_dp == 0:  # display training result
                        if acc_op is not None:  # exist accuracy op. in graph
                            # calculate batch loss and accuracy
                            train_loss, train_acc = sess.run([loss_op, acc_op],
                                                             feed_dict={batch_x: batched_x, batch_y: batched_y})
                            print(
                                format_str_acc.format(step, train_loss, acc_layer.type, train_acc, time.time() - start_time))
                        else:  # no acc op.
                            # calculate batch loss
                            train_loss = sess.run(loss_op,
                                                  feed_dict={batch_x: batched_x, batch_y: batched_y})
                            print(format_str.format(step, train_loss, time.time() - start_time))
                    if test_interval is not None:
                        if acc_op is not None and step % test_interval == 0:  # display testing result
                            print('Testing......')
                            test_acc_mean = 0.0
                            test_loss_mean = 0.0
                            for i in xrange(1, test_iter):
                                batched_x, batched_y = test_reader.next_batch()
                                test_acc, test_loss = sess.run([acc_op, loss_op],
                                                    feed_dict={batch_x: batched_x, batch_y: batched_y,
                                                               is_train: False})
                                test_acc_mean += test_acc
                                test_loss_mean += test_loss
                            test_acc_mean = test_acc_mean / test_iter
                            test_loss_mean = test_loss_mean / test_iter
                            summary = tf.Summary(value=[tf.Summary.Value(tag='test_accuracy', simple_value=test_acc_mean),
                                                        tf.Summary.Value(tag='test_loss', simple_value=test_loss_mean)
                                                        ])
                            summary_writer.add_summary(summary, step)
                            print(format_str_test.format(step, acc_layer.type, test_acc_mean))
        elif input_layer.type == 'database_input': # not yet used
            pass
        elif input_layer.type == 'mnist_input':
            network.run(learning_option, cluster)  # bulid TF model
            # get operations
            is_train = learning_option.get('is_train')  # placeholder to set procedure
            # get loss op.
            loss_op = loss_layer.get_op_output('output')  # output op of loss layer is only 'loss'
            # get opt. op.
            train_op = opt_layer.get_op_output('output')  # output op of opt. layer is only 'output'

            # get accuracy layer and operation
            # WARNING: current accuracy related layers -> accuracy/top-k layer, perplexity layer
            acc_layer = network.get_layer_by_type('accuracy|perplexity')  # get acc layer
            if acc_layer == []:  # no acc op
                print('[DLMDL] no accuracy related layer in model')
                acc_op = None
            else:
                acc_layer = network.get_layer_by_type('accuracy|perplexity').pop()
                print('[DLMDL] using {0} as accuracy of model'.format(acc_layer.type))
                acc_op = acc_layer.get_op_output('output')  # output op of accuracy layer is only 'output'

            # get input placeholders.
            batch_x = input_layer.get_op_output('image')  # image/data/text/... (data info.)
            batch_y = input_layer.get_op_output('label')  # labels/targets/... (label info.)

            # print format
            format_str = '[{0:>6} Steps] Train Loss:{1:>8.3f}({2:>8.3f} sec/batch)'
            format_str_acc = '[{0:>6} Steps] Train Loss:{1:>8.3f}, Train {2}:{3:>8.3%}({4:>8.3f} sec/batch)'
            format_str_test = '[{0:>6} Steps] Average Test {1}:{2:>8.3%}'

            # get option attributes
            is_shuffle = True  # TODO: always true

            # get reader class for training data
            train_reader = MNISTreader(train_dir, train_bs, is_shuffle, istrain=1)
            if test_dir is None:  # if no test dataset
                test_reader = None
            else:  # get reader class for testing data
                test_reader = MNISTreader(test_dir, test_bs, is_shuffle, istrain=0)

            if input_layer.get_attr('num_steps') is not None: # only for training RNN with MNIST
                num_units = 28 # fixed
                num_steps = input_layer.get_attr('num_steps')

            # open TF session
            with tf.train.MonitoredTrainingSession(master=server, is_chief=(worker_idx == 0),
                                                   hooks=[stophooks],
                                                   save_checkpoint_steps=ckpt_interval,
                                                   checkpoint_dir=ckpt_dir,
                                                   save_summaries_steps=train_dp,
                                                   summary_dir=summarydir,
                                                   log_step_count_steps=0,
                                                   config=config) as sess:
                summary_writer = SummaryWriterCache.get(summarydir)
                while not sess.should_stop():
                    start_time = time.time()
                    # one step training
                    batched_x, batched_y = train_reader.next_batch()
                    if input_layer.get_attr('num_steps') is not None: # only for training RNN with MNIST
                        batched_x = batched_x.reshape((train_bs, num_steps, num_units))
                    _, step = sess.run([train_op, global_step],
                                       feed_dict={batch_x: batched_x, batch_y: batched_y})
                    if step % train_dp == 0:  # display training result
                        if acc_op is not None:  # exist accuracy op. in graph
                            # calculate batch loss and accuracy
                            train_loss, train_acc = sess.run([loss_op, acc_op],
                                                             feed_dict={batch_x: batched_x, batch_y: batched_y})
                            print(
                                format_str_acc.format(step, train_loss, acc_layer.type, train_acc,
                                                      time.time() - start_time))
                        else:  # no acc op.
                            # calculate batch loss
                            train_loss = sess.run(loss_op,
                                                  feed_dict={batch_x: batched_x, batch_y: batched_y})
                            print(format_str.format(step, train_loss, time.time() - start_time))
                    if test_interval is not None:
                        if acc_op is not None and step % test_interval == 0:  # display testing result
                            print('Testing......')
                            test_acc_mean = 0.0
                            test_loss_mean = 0.0
                            for i in xrange(1, test_iter):
                                batched_x, batched_y = test_reader.next_batch()
                                if learning_option.get('num_steps') is not None:  # only for training RNN with MNIST
                                    batched_x = batched_x.reshape((train_bs, num_steps, num_units))
                                test_acc, test_loss = sess.run([acc_op, loss_op],
                                                    feed_dict={batch_x: batched_x, batch_y: batched_y,
                                                               is_train: False})
                                test_acc_mean += test_acc
                                test_loss_mean += test_loss
                            test_acc_mean = test_acc_mean / test_iter
                            test_loss_mean = test_loss_mean / test_iter
                            summary = tf.Summary(value=[tf.Summary.Value(tag='test_accuracy', simple_value=test_acc_mean),
                                                        tf.Summary.Value(tag='test_loss', simple_value=test_loss_mean)
                                                        ])
                            summary_writer.add_summary(summary, step)
                            print(format_str_test.format(step, acc_layer.type, test_acc_mean))
        elif input_layer.type == 'cifar10_input':
            network.run(learning_option, cluster)  # bulid TF model
            # get operations
            is_train = learning_option.get('is_train')  # placeholder to set procedure
            # get loss op.
            loss_op = loss_layer.get_op_output('output')  # output op of loss layer is only 'loss'
            # get opt. op.
            train_op = opt_layer.get_op_output('output')  # output op of opt. layer is only 'output'

            # get accuracy layer and operation
            # WARNING: current accuracy related layers -> accuracy/top-k layer, perplexity layer
            acc_layer = network.get_layer_by_type('accuracy|perplexity')  # get acc layer
            if acc_layer == []:  # no acc op
                print('[DLMDL] no accuracy related layer in model')
                acc_op = None
            else:
                acc_layer = network.get_layer_by_type('accuracy|perplexity').pop()
                print('[DLMDL] using {0} as accuracy of model'.format(acc_layer.type))
                acc_op = acc_layer.get_op_output('output')  # output op of accuracy layer is only 'output'

            # get input placeholders.
            batch_x = input_layer.get_op_output('image')  # image/data/text/... (data info.)
            batch_y = input_layer.get_op_output('label')  # labels/targets/... (label info.)

            # print format
            format_str = '[{0:>6} Steps] Train Loss:{1:>8.3f}({2:>8.3f} sec/batch)'
            format_str_acc = '[{0:>6} Steps] Train Loss:{1:>8.3f}, Train {2}:{3:>8.3%}({4:>8.3f} sec/batch)'
            format_str_test = '[{0:>6} Steps] Average Test {1}:{2:>8.3%}'

            # get option attributes
            is_shuffle = True # TODO: always true

            # get reader class for training data
            train_reader = CIFAR10reader(train_dir, train_bs, is_shuffle, istrain=1)
            if test_dir is None:  # if no test dataset
                test_reader = None
            else:  # get reader class for testing data
                test_reader = CIFAR10reader(test_dir, test_bs, is_shuffle, istrain=0)

            # open TF session
            with tf.train.MonitoredTrainingSession(master=server, is_chief=(worker_idx == 0),
                                                   hooks=[stophooks],
                                                   save_checkpoint_steps=ckpt_interval,
                                                   checkpoint_dir=ckpt_dir,
                                                   summary_dir=summarydir,
                                                   save_summaries_steps=train_dp,
                                                   log_step_count_steps=0,
                                                   config=config) as sess:
                summary_writer = SummaryWriterCache.get(summarydir)
                while not sess.should_stop():
                    start_time = time.time()
                    # one step training
                    batched_x, batched_y = train_reader.next_batch()
                    _, step = sess.run([train_op, global_step],
                                       feed_dict={batch_x: batched_x, batch_y: batched_y})
                    if step % train_dp == 0:  # display training result
                        if acc_op is not None:  # exist accuracy op. in graph
                            # calculate batch loss and accuracy
                            train_loss, train_acc = sess.run([loss_op, acc_op],
                                                             feed_dict={batch_x: batched_x, batch_y: batched_y})
                            print(
                                format_str_acc.format(step, train_loss, acc_layer.type, train_acc,
                                                      time.time() - start_time))
                        else:  # no acc op.
                            # calculate batch loss
                            train_loss = sess.run(loss_op,
                                                  feed_dict={batch_x: batched_x, batch_y: batched_y})
                            print(format_str.format(step, train_loss, time.time() - start_time))
                    if test_interval is not None:
                        if acc_op is not None and step % test_interval == 0:  # display testing result
                            print('Testing......')
                            test_acc_mean = 0.0
                            test_loss_mean = 0.0
                            for i in xrange(1, test_iter):
                                batched_x, batched_y = test_reader.next_batch()
                                test_acc, test_loss = sess.run([acc_op, loss_op],
                                                    feed_dict={batch_x: batched_x, batch_y: batched_y,
                                                               is_train: False})
                                test_acc_mean += test_acc
                                test_loss_mean += test_loss
                            test_acc_mean = test_acc_mean / test_iter
                            test_loss_mean = test_loss_mean / test_iter
                            summary = tf.Summary(value=[tf.Summary.Value(tag='test_accuracy', simple_value=test_acc_mean),
                                                        tf.Summary.Value(tag='test_loss', simple_value=test_loss_mean)
                                                        ])
                            summary_writer.add_summary(summary, step)
                            print(format_str_test.format(step, acc_layer.type, test_acc_mean))
        elif input_layer.type == 'glaucoma_input':
            network.run(learning_option, cluster)  # bulid TF model
            # get operations
            is_train = learning_option.get('is_train')  # placeholder to set procedure
            # get loss op.
            loss_op = loss_layer.get_op_output('output')  # output op of loss layer is only 'loss'
            # get opt. op.
            train_op = opt_layer.get_op_output('output')  # output op of opt. layer is only 'output'

            # get accuracy layer and operation
            # WARNING: current accuracy related layers -> accuracy/top-k layer, perplexity layer
            acc_layer = network.get_layer_by_type('accuracy|perplexity')  # get acc layer
            if acc_layer == []:  # no acc op
                print('[DLMDL] no accuracy related layer in model')
                acc_op = None
            else:
                acc_layer = network.get_layer_by_type('accuracy|perplexity').pop()
                print('[DLMDL] using {0} as accuracy of model'.format(acc_layer.type))
                acc_op = acc_layer.get_op_output('output')  # output op of accuracy layer is only 'output'

            # data augumentation for glaucoma images - refered from original source code -twkim
            def data_aug(data, labels):
                aug_data1 = tf.map_fn(lambda img: tf.image.crop_to_bounding_box(img, 0, 0, 224, 224), data)
                aug_data2 = tf.map_fn(lambda img: tf.image.crop_to_bounding_box(img, 0, 16, 224, 224), data)
                aug_data3 = tf.map_fn(lambda img: tf.image.crop_to_bounding_box(img, 16, 16, 224, 224), data)
                aug_data4 = tf.map_fn(lambda img: tf.image.crop_to_bounding_box(img, 16, 10, 224, 224), data)
                aug_data5 = tf.map_fn(lambda img: tf.image.central_crop(img, 0.93), data)
                flips = tf.map_fn(lambda img: tf.image.random_flip_left_right(img), data)
                aug_data6 = tf.map_fn(lambda img: tf.image.crop_to_bounding_box(img, 0, 0, 224, 224), flips)
                aug_data7 = tf.map_fn(lambda img: tf.image.crop_to_bounding_box(img, 0, 16, 224, 224), flips)
                aug_data8 = tf.map_fn(lambda img: tf.image.crop_to_bounding_box(img, 16, 16, 224, 224), flips)
                aug_data9 = tf.map_fn(lambda img: tf.image.crop_to_bounding_box(img, 16, 10, 224, 224), flips)
                aug_data10 = tf.map_fn(lambda img: tf.image.central_crop(img, 0.93), data)
                # concat_data = data
                # concat_labels = labels
                concat_data = tf.concat((aug_data1, aug_data2, aug_data3, aug_data4, aug_data5,
                                         aug_data6, aug_data7, aug_data8, aug_data9,
                                         aug_data10), axis=0)
                concat_labels = tf.concat((labels, labels, labels, labels, labels,
                                           labels, labels, labels, labels, labels), axis=0)
                return (concat_data, concat_labels)

            # get input placeholders.
            batch_x = input_layer.get_op_output('image')  # image/data/text/... (data info.)
            batch_y = input_layer.get_op_output('label')  # labels/targets/... (label info.)

            # define placeholder for preprocessing raw images
            images_raw = tf.placeholder(tf.float32, shape=(None, 240, 240, 3), name='glaucoma_raw_images')
            labels_raw = tf.placeholder(tf.int32, shape=(None, 2), name='glaucoma_raw_labels')

            with tf.device('/cpu:0'):
                augmented_train_images, augmented_train_labels = data_aug(images_raw, labels_raw)
                augmented_test_images, augmented_test_labels = data_aug(images_raw, labels_raw)

            # print format
            format_str = '[{0:>6} Steps] Train Loss:{1:>8.3f}({2:>8.3f} sec/batch)'
            format_str_acc = '[{0:>6} Steps] Train Loss:{1:>8.3f}, Train {2}:{3:>8.3%}({4:>8.3f} sec/batch)'
            format_str_test = '[{0:>6} Steps] Average Test {1}:{2:>8.3%}'

            # get option attributes
            is_shuffle = learning_option.get('shuffle')

            # get reader class for training data
            train_images = open(train_dir + '/train_dataset.npy')
            train_labels = open(train_dir + '/train_labels.npy')
            raw_train_images = np.load(train_images)
            raw_train_labels = np.load(train_labels)
            if test_dir is None:  # if no test dataset
                test_reader = None
            else:  # get reader class for testing data
                test_images = open(train_dir + '/test_dataset.npy')
                test_labels = open(train_dir + '/test_labels.npy')
                raw_test_images = np.load(test_images)
                raw_test_labels = np.load(test_labels)

            # open TF session
            with tf.train.MonitoredTrainingSession(master=server, is_chief=(worker_idx == 0),
                                                   hooks=[stophooks],
                                                   save_checkpoint_steps=ckpt_interval,
                                                   checkpoint_dir=ckpt_dir,
                                                   summary_dir=summarydir,
                                                   save_summaries_steps=train_dp,
                                                   log_step_count_steps=0,
                                                   config=config) as sess:
                summary_writer = SummaryWriterCache.get(summarydir)
                # load arugmented data w.r.t training/test dataset
                tmp_x = np.zeros([1, 224, 224, 3], dtype=np.float32)
                tmp_y = np.zeros([1, 2], dtype=np.float32)


                train_images_aug, train_labels_aug = sess.run([augmented_train_images, augmented_train_labels],
                                                              feed_dict={images_raw: raw_train_images,
                                                                         labels_raw: raw_train_labels,
                                                                         batch_x: tmp_x, batch_y: tmp_y})
                train_reader = GLAUCOMAreader(train_dir, train_bs, is_shuffle,
                                              train_images_aug, train_labels_aug)
                if acc_op is not None:
                    test_images_aug, test_labels_aug = sess.run([augmented_test_images, augmented_test_labels],
                                                                  feed_dict={images_raw: raw_test_images,
                                                                             labels_raw: raw_test_labels,
                                                                             batch_x: tmp_x, batch_y: tmp_y})
                    test_reader = GLAUCOMAreader(test_dir, test_bs, is_shuffle,
                                                 test_images_aug, test_labels_aug)
                while not sess.should_stop():
                    start_time = time.time()
                    # one step training
                    batched_x, batched_y = train_reader.next_batch()
                    _, step = sess.run([train_op, global_step],
                                       feed_dict={batch_x: batched_x, batch_y: batched_y})
                    if step % train_dp == 0:  # display training result
                        if acc_op is not None:  # exist accuracy op. in graph
                            # calculate batch loss and accuracy
                            train_loss, train_acc = sess.run([loss_op, acc_op],
                                                             feed_dict={batch_x: batched_x, batch_y: batched_y})
                            print(
                                format_str_acc.format(step, train_loss, acc_layer.type, train_acc,
                                                      time.time() - start_time))
                        else:  # no acc op.
                            # calculate batch loss
                            train_loss = sess.run(loss_op,
                                                  feed_dict={batch_x: batched_x, batch_y: batched_y})
                            print(format_str.format(step, train_loss, time.time() - start_time))

                    if test_interval is not None:
                        if acc_op is not None and step % test_interval == 0:  # display testing result
                            print('Testing......')
                            test_acc_mean = 0.0
                            test_loss_mean = 0.0
                            for i in xrange(1, test_iter):
                                batched_x, batched_y = test_reader.next_batch()
                                test_acc, test_loss = sess.run([acc_op, loss_op],
                                                    feed_dict={batch_x: batched_x, batch_y: batched_y,
                                                               is_train: False})
                                test_acc_mean += test_acc
                                test_loss_mean += test_loss
                            test_acc_mean = test_acc_mean / test_iter
                            test_loss_mean = test_loss_mean / test_iter
                            summary = tf.Summary(value=[tf.Summary.Value(tag='test_accuracy', simple_value=test_acc_mean),
                                                        tf.Summary.Value(tag='test_loss', simple_value=test_loss_mean)
                                                        ])
                            summary_writer.add_summary(summary, step)
                            print(format_str_test.format(step, acc_layer.type, test_acc_mean))
        elif input_layer.type == 'ptb_input':
            network.run(learning_option, cluster)  # bulid TF model
            # get operations
            is_train = learning_option.get('is_train')  # placeholder to set procedure
            # get loss op.
            loss_op = loss_layer.get_op_output('output')  # output op of loss layer is only 'loss'
            # get opt. op.
            train_op = opt_layer.get_op_output('output')  # output op of opt. layer is only 'output'

            # get accuracy layer and operation
            # WARNING: current accuracy related layers -> accuracy/top-k layer, perplexity layer
            acc_layer = network.get_layer_by_type('accuracy|perplexity')  # get acc layer
            if acc_layer == []:  # no acc op
                print('[DLMDL] no accuracy related layer in model')
                acc_op = None
            else:
                acc_layer = network.get_layer_by_type('accuracy|perplexity').pop()
                print('[DLMDL] using {0} as accuracy of model'.format(acc_layer.type))
                acc_op = acc_layer.get_op_output('output')  # output op of accuracy layer is only 'output'

            # get input placeholders.
            batch_x = learning_option.get('data_placeholder')  # image/data/text/... (data info.)
            batch_y = input_layer.get_op_output('targets')  # labels/targets/... (label info.)

            # print format
            format_str = '[{0:>3.2f} Epochs, {1:>6} Steps] Train Loss:{2:>8.3f}({2:>8.3f} sec/batch)'
            format_str_acc = '[{0:>3.2f} Epochs, {1:>6} Steps] Train Loss:{2:>8.3f}, Train {3}:{4:>8.3f}({5:>8.3f} sec/batch)'
            format_str_test = '[{0:>3.2f} Epochs, {1:>6} Steps] Average Test {2}:{3:>8.3f}'

            initial_state = learning_option.get('initial_state')  # get TF initial state op.
            # WARNINIG: if there is other op. added to static_rnn, use meta character '|'
            # e.g 'static_rnn|static_bidirectional_rnn' --> detect static_rnn or static_bidirectional_rnn op.
            final_state = network.get_layer_by_type('static_rnn').pop().get_op_output(
                'state')  # get TF output state op.

            # get option attributes
            num_steps = learning_option.get('num_steps')

            # get reader class for training data
            train_reader = PTBreader(train_dir, train_bs, num_steps, istrain=True)
            if test_dir is None:  # if no test dataset
                test_reader = None
            else:  # get reader class for testing data
                test_reader = PTBreader(test_dir, test_bs, num_steps, istrain=False)


            # open TF session
            with tf.train.MonitoredTrainingSession(master=server, is_chief=(worker_idx == 0),
                                                   hooks=[stophooks],
                                                   save_checkpoint_steps=ckpt_interval,
                                                   checkpoint_dir=ckpt_dir,
                                                   save_summaries_steps=train_dp,
                                                   summary_dir=summarydir,
                                                   log_step_count_steps=0,
                                                   config=config) as sess:
                summary_writer = SummaryWriterCache.get(summarydir)
                current_epoch = 0
                while not sess.should_stop():
                    if initial_state is not None:
                        batched_x, batched_y = train_reader.next_batch()
                        state = sess.run(initial_state,
                                         feed_dict={batch_x: batched_x,
                                                    batch_y: batched_y})
                    else:  # TODO: unable condition
                        state = None
                    train_epoch_size = train_reader.get_epoch_size()
                    for step_per_epoch in range(train_epoch_size):
                        # run one epoch
                        start_time = time.time()
                        feed_dict = {}
                        # one step training
                        batched_x, batched_y = train_reader.next_batch()
                        feed_dict[batch_x] = batched_x
                        feed_dict[batch_y] = batched_y
                        feed_dict[is_train] = True
                        for i, (c, h) in enumerate(initial_state):
                            feed_dict[c] = state[i].c
                            feed_dict[h] = state[i].h
                        _, fin_stat, step = sess.run([train_op, final_state, global_step], feed_dict)
                        state = fin_stat
                        if step % train_dp == 0:  # display training result
                            if acc_op is not None:  # exist accuracy op. in graph
                                # calculate batch loss and accuracy
                                train_loss, train_acc = sess.run([loss_op, acc_op],
                                                                 feed_dict)
                                print(format_str_acc.format(current_epoch + step_per_epoch * 1.0 / train_epoch_size,
                                                            step, train_loss, acc_layer.type, train_acc,
                                                            time.time() - start_time))
                            else:  # no acc op.
                                # calculate batch loss
                                train_loss = sess.run(loss_op, feed_dict)
                                print(format_str.format(
                                    current_epoch + step_per_epoch * 1.0 / train_epoch_size,
                                    step, train_loss, time.time() - start_time))

                    if test_interval is not None:
                        if acc_op is not None and step % test_interval == 0:  # display testing result
                            print('Testing......')
                            # calculate average accuracy during test iterations
                            test_iters = 0
                            test_losses = 0.0
                            if initial_state is not None:
                                batched_x, batched_y = test_reader.next_batch()
                                test_state = sess.run(initial_state,
                                                      feed_dict={batch_x: batched_x, batch_y: batched_y})
                            else:  # TODO: unable condition
                                test_state = None
                            test_epoch_size = test_reader.get_epoch_size()
                            for test_step in range(test_iter):  # during test iterations
                                batched_x, batched_y = test_reader.next_batch()  # get test batch
                                feed_dict = {}
                                feed_dict[batch_x] = batched_x
                                feed_dict[batch_y] = batched_y
                                feed_dict[is_train] = False
                                for i, (c, h) in enumerate(initial_state):
                                    feed_dict[c] = test_state[i].c
                                    feed_dict[h] = test_state[i].h

                                test_loss, test_fin_stat = sess.run([loss_op, final_state],
                                                                    feed_dict)
                                test_state = test_fin_stat

                                test_losses += test_loss
                                test_iters += num_steps
                            summary = tf.Summary(
                                value=[tf.Summary.Value(tag=acc_layer.type, simple_value=np.exp(test_losses / test_iters))
                                       ])
                            summary_writer.add_summary(summary, 0.0 + test_iter * 1.0 / test_epoch_size)
                            print(format_str_test.format(0.0 + test_iter * 1.0 / test_epoch_size,
                                                         step, acc_layer.type,
                                                         np.exp(test_losses / test_iters)))
                    current_epoch += 1
        elif input_layer.type == 'csv_input': # not yet used
            pass
        elif input_layer.type == 'bodydata_input':
            network.run(learning_option, cluster)  # bulid TF model

            # get all loss, train_op, acc layers
            opt_layer = network.get_layer_by_type('adam|sgd|momentum|rmsprop|nesterov|adadelta|adagrad')  # get optimizer layer
            convnet_train_op = None
            others_train_op = None
            whole_train_op = None
            for layer in opt_layer:
                if layer.get_attr('scope') == 'convnet':
                    convnet_train_op = layer.get_op_output('output')
                elif layer.get_attr('scope') == 'others':
                    others_train_op = layer.get_op_output('output')
                else:
                    whole_train_op = layer.get_op_output('output')

            convnet_loss_op = network.get_layer('convnet_loss').get_op_output('output')
            total_loss_op = network.get_layer('total_loss').get_op_output('output')
            others_op = network.get_layer('above_net_fc1').get_op_output('output')

            # WARNING: current accuracy related layers -> accuracy/top-k layer, perplexity layer
            acc_layer = network.get_layer_by_type('accuracy|perplexity')  # get acc layer
            bt_acc_op = None
            bs_acc_op = None
            full_bt_acc_op = None
            full_bs_acc_op = None
            for layer in acc_layer:
                if layer.name == 'bt_acc':
                    bt_acc_op = layer.get_op_output('output')
                elif layer.name == 'bs_acc':
                    bs_acc_op = layer.get_op_output('output')
                elif layer.name == 'full_bt_acc':
                    full_bt_acc_op = layer.get_op_output('output')
                elif layer.name == 'full_bs_acc':
                    full_bs_acc_op = layer.get_op_output('output')

            batch_image_x = input_layer.get_op_output('image')
            batch_bt_label_y = input_layer.get_op_output('bt_label')
            batch_bs_label_y = input_layer.get_op_output('bs_label')
            batch_numeric_x = input_layer.get_op_output('numeric')
            batch_numeric_label_y = input_layer.get_op_output('numeric_label')

            # print format
            format_str_convnet = '[{0:>6} Steps] Convnet Train Loss:{1:>8.3f}({2:>8.3f} sec/batch)'
            format_str_others = '[{0:>6} Steps] OthersNet Train Loss:{1:>8.3f}({2:>8.3f} sec/batch)'
            format_str_full = '[{0:>6} Steps] Full Train Loss:{1:>8.3f}({2:>8.3f} sec/batch)'

            # load input reader
            in_reader = BODYDATAreader(train_dir, False)

            # open TF session
            with tf.train.MonitoredTrainingSession(master=server, is_chief=(worker_idx == 0),
                                                   save_checkpoint_steps=ckpt_interval,
                                                   save_summaries_steps=train_dp,
                                                   summary_dir=summarydir,
                                                   checkpoint_dir=ckpt_dir,
                                                   log_step_count_steps=0,
                                                   config=config) as sess:
                summary_writer = SummaryWriterCache.get(summarydir)
                print('(1) ConvNet training w/ BT and BS image -------------------------')
                for step in range(train_iter):
                    batched_image, batched_numeric, batched_numeric_label, batched_bt_label, batched_bs_label= in_reader.next_batch(train_bs, is_train=True)
                    start_time = time.time()
                    convnet_train_loss, _ = sess.run([convnet_loss_op, convnet_train_op],
                                                     feed_dict={batch_image_x: batched_image,
                                                                batch_bt_label_y: batched_bt_label,
                                                                batch_bs_label_y: batched_bs_label,
                                                                batch_numeric_x: batched_numeric,
                                                                batch_numeric_label_y: batched_numeric_label})

                    if step % train_dp == 0:
                        print(format_str_convnet.format(step, convnet_train_loss, time.time()-start_time))
                        #train_writer.add_summary(summ, step)

                print('(2) on the mid of train--------------------------------')
                for _ in range(30):
                    batched_image, batched_numeric, batched_numeric_label, batched_bt_label, batched_bs_label = in_reader.next_batch(train_bs, is_train=False)
                    reg_output, bt_acc, bs_acc= sess.run([others_op, bt_acc_op, bs_acc_op],
                                                         feed_dict={batch_image_x: batched_image,
                                                                    batch_bt_label_y: batched_bt_label,
                                                                    batch_bs_label_y: batched_bs_label,
                                                                    batch_numeric_x: batched_numeric,
                                                                    batch_numeric_label_y: batched_numeric_label})
                print ('(2) Body Type Accuracy: {0:>5.3%}, Body Shape Accuracy: {1:>5.3%}'.format(bt_acc, bs_acc))
                print('(3)--------------------------------------------------')
                for step in range(train_iter):
                    batched_image, batched_numeric, batched_numeric_label, batched_bt_label, batched_bs_label= in_reader.next_batch(train_bs, is_train=True)
                    start_time = time.time()
                    total_train_loss, _ = sess.run([total_loss_op, others_train_op],
                                                   feed_dict={batch_image_x: batched_image,
                                                              batch_bt_label_y: batched_bt_label,
                                                              batch_bs_label_y: batched_bs_label,
                                                              batch_numeric_x: batched_numeric,
                                                              batch_numeric_label_y: batched_numeric_label})
                    if step % train_dp == 0:
                        print(format_str_others.format(step, total_train_loss, time.time()-start_time))
                print('(4)--------------------------------------------------')
                for step in range(train_iter):
                    batched_image, batched_numeric, batched_numeric_label, batched_bt_label, batched_bs_label = in_reader.next_batch(
                        train_bs, is_train=True)
                    start_time = time.time()
                    total_train_loss, _ = sess.run([total_loss_op, whole_train_op],
                                                   feed_dict={batch_image_x: batched_image,
                                                              batch_bt_label_y: batched_bt_label,
                                                              batch_bs_label_y: batched_bs_label,
                                                              batch_numeric_x: batched_numeric,
                                                              batch_numeric_label_y: batched_numeric_label})
                    if step % train_dp == 0:
                        print(format_str_full.format(step, total_train_loss, time.time() - start_time))
                print('TRAIN COMPLETE')
                print('(5)--------------------------------------------------')
                bt_acc_list = []
                bs_acc_list = []
                reg_error_rate_list = []
                for _ in range(test_iter):
                    batched_image, batched_numeric, batched_numeric_label, batched_bt_label, batched_bs_label = in_reader.next_batch(
                        train_bs, is_train=False)
                    reg_output, bt_acc, bs_acc = sess.run([others_op, bt_acc_op, bs_acc_op],
                                                          feed_dict={batch_image_x: batched_image,
                                                                     batch_bt_label_y: batched_bt_label,
                                                                     batch_bs_label_y: batched_bs_label,
                                                                     batch_numeric_x: batched_numeric,
                                                                     batch_numeric_label_y: batched_numeric_label})
                    bt_acc_list.append(bt_acc)
                    bs_acc_list.append(bs_acc)
                    output_std = in_reader.get_output_std()
                    output_avg = in_reader.get_output_avg()

                    reg_pred = reg_output * output_std + output_avg
                    reg_label = batched_numeric_label * output_std + output_avg
                    error_rate = np.abs((reg_pred - reg_label) / reg_label * 100)
                    reg_error_rate_list.append(error_rate)

            print('(5) Body Type Accuracy: {0:>5.3%}, Body Shape Accuracy: {1:>5.3%}'.format(np.mean(bt_acc_list), np.mean(bs_acc_list)))
            reg_error_rate_list = np.concatenate(reg_error_rate_list, 0)
            msg = '(5) Avg. Error Rate for Each Item: {0}, Median Error Rate for Each Item: {1}, Third Quarter of Error Rate for Each Item: {2}'
            print(msg.format(np.mean(reg_error_rate_list, 0), np.median(reg_error_rate_list, 0), np.percentile(reg_error_rate_list, 75, axis=0)))

        elif input_layer.type == 'imagenet_input':
            # get option attributes
            is_shuffle = True  # TODO: always true
            # get reader class for training data
            train_reader = IMAGENETreader(train_dir, train_bs, is_shuffle, is_train=1)
            if test_dir is None:  # if no test dataset
                test_reader = None
            else:  # get reader class for testing data
                test_reader = IMAGENETreader(test_dir, test_bs, is_shuffle, is_train=0)
                learning_option['test_imagenet'] = test_reader.next_batch()

            learning_option['train_imagenet'] = train_reader.next_batch()

            network.run(learning_option, cluster)  # bulid TF model
            # get operations
            is_train = learning_option.get('is_train')  # placeholder to set procedure
            # get loss op.
            loss_op = loss_layer.get_op_output('output')  # output op of loss layer is only 'loss'
            # get opt. op.
            train_op = opt_layer.get_op_output('output')  # output op of opt. layer is only 'output'

            # get accuracy layer and operation
            # WARNING: current accuracy related layers -> accuracy/top-k layer, perplexity layer
            acc_layer = network.get_layer_by_type('accuracy|perplexity')  # get acc layer
            if acc_layer == []:  # no acc op
                print('[DLMDL] no accuracy related layer in model')
                acc_op = None
            else:
                acc_layer = network.get_layer_by_type('accuracy|perplexity').pop()
                print('[DLMDL] using {0} as accuracy of model'.format(acc_layer.type))
                acc_op = acc_layer.get_op_output('output')  # output op of accuracy layer is only 'output'

            # print format
            format_str = '[{0:>6} Steps] Train Loss:{1:>8.3f}({2:>8.3f} sec/batch)'
            format_str_acc = '[{0:>6} Steps] Train Loss:{1:>8.3f}, Train {2}:{3:>8.3%}({4:>8.3f} sec/batch)'
            format_str_test = '[{0:>6} Steps] Average Test {1}:{2:>8.3%}'

            # open TF session
            with tf.train.MonitoredTrainingSession(master=server, is_chief=(worker_idx == 0),
                                                   hooks=[stophooks],
                                                   save_checkpoint_steps=ckpt_interval,
                                                   checkpoint_dir=ckpt_dir,
                                                   summary_dir=summarydir,
                                                   save_summaries_steps=train_dp,
                                                   log_step_count_steps=0,
                                                   config=config) as sess:
                summary_writer = SummaryWriterCache.get(summarydir)
                while not sess.should_stop():
                    start_time = time.time()
                    # one step training
                    _, step = sess.run([train_op, global_step])
                    if step % train_dp == 0:  # display training result
                        if acc_op is not None:  # exist accuracy op. in graph
                            # calculate batch loss and accuracy
                            train_loss, train_acc = sess.run([loss_op, acc_op])
                            print(
                                format_str_acc.format(step, train_loss, acc_layer.type, train_acc,
                                                      time.time() - start_time))
                        else:  # no acc op.
                            # calculate batch loss
                            train_loss = sess.run(loss_op)
                            print(format_str.format(step, train_loss, time.time() - start_time))
                    if test_interval is not None:
                        if acc_op is not None and step % test_interval == 0:  # display testing result
                            print('Testing......')
                            test_acc_mean = 0.0
                            test_loss_mean = 0.0
                            for i in xrange(1, test_iter):
                                test_acc, test_loss = sess.run([acc_op, loss_op], feed_dict={is_train: False})
                                test_acc_mean += test_acc
                                test_loss_mean += test_loss
                            test_acc_mean = test_acc_mean / test_iter
                            test_loss_mean = test_loss_mean / test_iter
                            summary = tf.Summary(
                                value=[tf.Summary.Value(tag='test_accuracy', simple_value=test_acc_mean),
                                       tf.Summary.Value(tag='test_loss', simple_value=test_loss_mean)
                                       ])
                            summary_writer.add_summary(summary, step)
                            print(format_str_test.format(step, acc_layer.type, test_acc_mean))
        # close processes
        # WARNINNG: In release, ssh handler don't include
        #for conn in conns:
        #    del conn

    @staticmethod
    def worker_executor_mb(UDLI, cluster, server, learning_option, network, worker_idx):
        current_address = tf_adaptor.resolve_address()  # get address of this node
        print('[DLMDL] worker ip: {0}, worker index: {1}'.format(current_address, worker_idx))
        network.run(learning_option, cluster)  # bulid TF model

        # WARNING: parameter server must be last number which is bottom worker in .dlmdl cluster
        #          In release version, ssh handler don't include
        # conns = tf_adaptor.launch_slaves(UDLI, cluster.get('ips')[1:])

        # get layers: assume input/loss/optimizer must exist, but accuracy layer is optional
        input_layer = network.get_layer_by_type('input$').pop()  # get input layer
        print('[DLMDL] datasets({0}) training'.format(input_layer.type))
        loss_layer = network.get_layer_by_type('loss$').pop()  # get loss layer
        print('[DLMDL] using {0} as cost of model'.format(loss_layer.type))
        opt_layer = network.get_layer_by_type(
            'adam|sgd|momentum|rmsprop|nesterov|adadelta|adagrad').pop()  # get optimizer layer
        print('[DLMDL] using {0} as optimizer of model'.format(opt_layer.type))
        global_step = tf.train.get_or_create_global_step()  # get global step

        # get operations
        is_train = learning_option.get('is_train')  # placeholder to set procedure
        # get loss op.
        loss_op = loss_layer.get_op_output('output')  # output op of loss layer is only 'loss'
        # get opt. op.
        train_op = opt_layer.get_op_output('output')  # output op of opt. layer is only 'output'

        # get accuracy layer and operation
        # WARNING: current accuracy related layers -> accuracy/top-k layer, perplexity layer
        acc_layer = network.get_layer_by_type('accuracy|perplexity')  # get acc layer
        if acc_layer == []:  # no acc op
            print('[DLMDL] no accuracy related layer in model')
            acc_op = None
        else:
            acc_layer = network.get_layer_by_type('accuracy|perplexity').pop()
            print('[DLMDL] using {0} as accuracy of model'.format(acc_layer.type))
            acc_op = acc_layer.get_op_output('output')  # output op of accuracy layer is only 'output'

        # TODO: error handling in learning option
        print('[DLMDL] learning option parsing...')
        # options related to training
        train_dir = learning_option.get('data_path')
        train_iter = learning_option.get('iteration')
        train_bs = learning_option.get('batch_size')
        train_dp = learning_option.get('train_display')

        # options related to test
        test_interval = learning_option.get('test_interval')
        test_dir = learning_option.get('test_data_path')
        test_iter = learning_option.get('test_iteration')
        test_bs = learning_option.get('test_batch_size')

        # options related to checkpoint
        ckpt_interval = learning_option.get('checkpoint_interval')
        if ckpt_interval is None:
            ckpt_interval = train_iter + 1
        ckpt_dir = learning_option.get('checkpoint_path')

        if input_layer.type == 'jpeg_input':  # for jpeg_input
            # get input placeholders.
            batch_x = input_layer.get_op_output('image')  # image/data/text/... (data info.)
            batch_y = input_layer.get_op_output('label')  # labels/targets/... (label info.)

            # print format
            format_str = '[{0:>6} Steps] Train Loss:{1:>8.3f}({2:>8.3f} sec/batch)'
            format_str_acc = '[{0:>6} Steps] Train Loss:{1:>8.3f}, Train {2}:{3:>8.3%}({4:>8.3f} sec/batch)'
            format_str_test = '[{0:>6} Steps] Average Test {1}:{2:>8.3%}'

            # get option attributes
            image_size = learning_option.get('image_size')
            is_shuffle = learning_option.get('shuffle')
            num_classes = learning_option.get('num_class')

            # get reader class for training data
            train_label_dir = learning_option.get('label_path')
            train_reader = JPEGreader(train_dir, train_label_dir, train_bs,
                                      image_size, is_shuffle, num_classes)
            if test_dir is None:  # if no test dataset
                test_reader = None
            else:  # get reader class for testing data
                test_label_dir = learning_option.get('test_label_path')
                test_reader = JPEGreader(test_dir, test_label_dir, test_bs,
                                         image_size, is_shuffle, num_classes)

            # TF summary merge op. and session configuration
            config = tf.ConfigProto()  # ocnfiguration object for session
            config.gpu_options.allow_growth = True  # allocate minimum GPU memory for running
            config.log_device_placement = False  # not loggging ops/vars-devices mapping
            config.allow_soft_placement = True  # automatically unallocated ops/vars to device

            # step, summary, checkpoint hook setting
            summarydir = UDLI.getArgs('log_out')

            # TODO: not implement
            # if option is restore, check the existing checkpoint file
            '''
            if learning_option.get('option') == 'RETRAIN':
                if len(glob.glob(ckpt_dir + '/*.meta')) == 0:
                    raise Exception('[DLMDL ERROR] checkpoint file(.meta) do not exist in {0}'.format(ckpt_dir))
                last_ckpt_filename = os.path.basename(glob.glob(ckpt_dir + '/*.meta')[-1])
                prev_iter = int(re.findall('\d+', last_ckpt_filename)[0])
                train_iter = train_iter - prev_iter
            '''

            merged = tf.summary.merge_all()
            train_writer = tf.summary.FileWriter(summarydir)

            init_op = tf.global_variables_initializer()
            sv = tf.train.Supervisor(is_chief=(worker_idx == 0),
                                     global_step=global_step,
                                     init_op=init_op,
                                     local_init_op=init_op)
            with sv.managed_session(server, config=config) as sess:
                step = 0
                while step < train_iter:
                    start_time = time.time()
                    # one step training
                    batched_x, batched_y = train_reader.next_batch()
                    _, step = sess.run([train_op, global_step],
                                       feed_dict={batch_x: batched_x, batch_y: batched_y})
                    if step % train_dp == 0:  # display training result
                        if acc_op is not None:  # exist accuracy op. in graph
                            # calculate batch loss and accuracy
                            train_loss, train_acc = sess.run([loss_op, acc_op],
                                                             feed_dict={batch_x: batched_x, batch_y: batched_y})
                            print(
                                format_str_acc.format(step, train_loss, acc_layer.type, train_acc,
                                                      time.time() - start_time))
                        else:  # no acc op.
                            # calculate batch loss
                            train_loss = sess.run(loss_op,
                                                  feed_dict={batch_x: batched_x, batch_y: batched_y})
                            print(format_str.format(step, train_loss, time.time() - start_time))
                        # write summary
                        summary = sess.run(merged, feed_dict={batch_x: batched_x, batch_y: batched_y})
                        train_writer.add_summary(summary, step)
                    if test_interval is not None:
                        if acc_op is not None and step % test_interval == 0:  # display testing result
                            print('Testing......')
                            test_acc_mean = 0.0
                            for i in xrange(1, test_iter):
                                batched_x, batched_y = test_reader.next_batch()
                                test_acc = sess.run(acc_op,
                                                    feed_dict={batch_x: batched_x, batch_y: batched_y,
                                                               is_train: False})
                                test_acc_mean += test_acc
                            test_acc_mean = test_acc_mean / test_iter
                            print(format_str_test.format(step, acc_layer.type, test_acc_mean))
        elif input_layer.type == 'database_input':  # not yet used
            pass
        elif input_layer.type == 'mnist_input':
            # get input placeholders.
            batch_x = input_layer.get_op_output('image')  # image/data/text/... (data info.)
            batch_y = input_layer.get_op_output('label')  # labels/targets/... (label info.)

            # print format
            format_str = '[{0:>6} Steps] Train Loss:{1:>8.3f}({2:>8.3f} sec/batch)'
            format_str_acc = '[{0:>6} Steps] Train Loss:{1:>8.3f}, Train {2}:{3:>8.3%}({4:>8.3f} sec/batch)'
            format_str_test = '[{0:>6} Steps] Average Test {1}:{2:>8.3%}'

            # get option attributes
            is_shuffle = True  # TODO: always true

            # get reader class for training data
            train_reader = MNISTreader(train_dir, train_bs, is_shuffle, istrain=1)
            if test_dir is None:  # if no test dataset
                test_reader = None
            else:  # get reader class for testing data
                test_reader = MNISTreader(test_dir, test_bs, is_shuffle, istrain=0)

            if input_layer.get_attr('num_steps') is not None:  # only for training RNN with MNIST
                num_units = 28  # fixed
                num_steps = input_layer.get_attr('num_steps')

            # TF summary merge op. and session configuration
            config = tf.ConfigProto()  # ocnfiguration object for session
            config.gpu_options.allow_growth = True  # allocate minimum GPU memory for running
            config.log_device_placement = False  # not loggging ops/vars-devices mapping
            config.allow_soft_placement = True  # automatically unallocated ops/vars to device

            # step, summary, checkpoint hook setting
            summarydir = UDLI.getArgs('log_out')

            # TODO: not implement
            # if option is restore, check the existing checkpoint file
            '''
            if learning_option.get('option') == 'RETRAIN':
                if len(glob.glob(ckpt_dir + '/*.meta')) == 0:
                    raise Exception('[DLMDL ERROR] checkpoint file(.meta) do not exist in {0}'.format(ckpt_dir))
                last_ckpt_filename = os.path.basename(glob.glob(ckpt_dir + '/*.meta')[-1])
                prev_iter = int(re.findall('\d+', last_ckpt_filename)[0])
                train_iter = train_iter - prev_iter
            '''

            merged = tf.summary.merge_all()
            train_writer = tf.summary.FileWriter(summarydir)

            init_op = tf.global_variables_initializer()
            sv = tf.train.Supervisor(is_chief=(worker_idx == 0),
                                     global_step=global_step,
                                     init_op=init_op,
                                     local_init_op=init_op)
            with sv.managed_session(server, config=config) as sess:
                step = 0
                while step < train_iter:
                    start_time = time.time()
                    # one step training
                    batched_x, batched_y = train_reader.next_batch()
                    if input_layer.get_attr('num_steps') is not None:  # only for training RNN with MNIST
                        batched_x = batched_x.reshape((train_bs, num_steps, num_units))
                    _, step = sess.run([train_op, global_step],
                                       feed_dict={batch_x: batched_x, batch_y: batched_y})
                    if step % train_dp == 0:  # display training result
                        if acc_op is not None:  # exist accuracy op. in graph
                            # calculate batch loss and accuracy
                            train_loss, train_acc = sess.run([loss_op, acc_op],
                                                             feed_dict={batch_x: batched_x, batch_y: batched_y})
                            print(
                                format_str_acc.format(step, train_loss, acc_layer.type, train_acc,
                                                      time.time() - start_time))
                        else:  # no acc op.
                            # calculate batch loss
                            train_loss = sess.run(loss_op,
                                                  feed_dict={batch_x: batched_x, batch_y: batched_y})
                            print(format_str.format(step, train_loss, time.time() - start_time))
                        # write summary
                        summary = sess.run(merged, feed_dict={batch_x: batched_x, batch_y: batched_y})
                        train_writer.add_summary(summary, step)
                    if test_interval is not None:
                        if acc_op is not None and step % test_interval == 0:  # display testing result
                            print('Testing......')
                            test_acc_mean = 0.0
                            for i in xrange(1, test_iter):
                                batched_x, batched_y = test_reader.next_batch()
                                if learning_option.get('num_steps') is not None:  # only for training RNN with MNIST
                                    batched_x = batched_x.reshape((train_bs, num_steps, num_units))
                                test_acc = sess.run(acc_op,
                                                    feed_dict={batch_x: batched_x, batch_y: batched_y,
                                                               is_train: False})
                                test_acc_mean += test_acc
                            test_acc_mean = test_acc_mean / test_iter
                            print(format_str_test.format(step, acc_layer.type, test_acc_mean))
        elif input_layer.type == 'cifar10_input':
            # get input placeholders.
            batch_x = input_layer.get_op_output('image')  # image/data/text/... (data info.)
            batch_y = input_layer.get_op_output('label')  # labels/targets/... (label info.)

            # print format
            format_str = '[{0:>6} Steps] Train Loss:{1:>8.3f}({2:>8.3f} sec/batch)'
            format_str_acc = '[{0:>6} Steps] Train Loss:{1:>8.3f}, Train {2}:{3:>8.3%}({4:>8.3f} sec/batch)'
            format_str_test = '[{0:>6} Steps] Average Test {1}:{2:>8.3%}'

            # get option attributes
            is_shuffle = True  # TODO: always true

            # get reader class for training data
            train_reader = CIFAR10reader(train_dir, train_bs, is_shuffle, istrain=1)
            if test_dir is None:  # if no test dataset
                test_reader = None
            else:  # get reader class for testing data
                test_reader = CIFAR10reader(test_dir, test_bs, is_shuffle, istrain=0)

            # TF summary merge op. and session configuration
            config = tf.ConfigProto()  # ocnfiguration object for session
            config.gpu_options.allow_growth = True  # allocate minimum GPU memory for running
            config.log_device_placement = False  # not loggging ops/vars-devices mapping
            config.allow_soft_placement = True  # automatically unallocated ops/vars to device

            # step, summary, checkpoint hook setting
            summarydir = UDLI.getArgs('log_out')
            # TODO: not implement
            # if option is restore, check the existing checkpoint file
            '''
            if learning_option.get('option') == 'RETRAIN':
                if len(glob.glob(ckpt_dir + '/*.meta')) == 0:
                    raise Exception('[DLMDL ERROR] checkpoint file(.meta) do not exist in {0}'.format(ckpt_dir))
                last_ckpt_filename = os.path.basename(glob.glob(ckpt_dir + '/*.meta')[-1])
                prev_iter = int(re.findall('\d+', last_ckpt_filename)[0])
                train_iter = train_iter - prev_iter
            '''

            merged = tf.summary.merge_all()
            train_writer = tf.summary.FileWriter(summarydir)

            init_op = tf.global_variables_initializer()
            sv = tf.train.Supervisor(is_chief=(worker_idx == 0),
                                     global_step=global_step,
                                     init_op=init_op,
                                     local_init_op=init_op)
            with sv.managed_session(server, config=config) as sess:
                step = 0
                while step < train_iter:
                    start_time = time.time()
                    # one step training
                    batched_x, batched_y = train_reader.next_batch()
                    _, step = sess.run([train_op, global_step],
                                       feed_dict={batch_x: batched_x, batch_y: batched_y})
                    if step % train_dp == 0:  # display training result
                        if acc_op is not None:  # exist accuracy op. in graph
                            # calculate batch loss and accuracy
                            train_loss, train_acc = sess.run([loss_op, acc_op],
                                                             feed_dict={batch_x: batched_x, batch_y: batched_y})
                            print(
                                format_str_acc.format(step, train_loss, acc_layer.type, train_acc,
                                                      time.time() - start_time))
                        else:  # no acc op.
                            # calculate batch loss
                            train_loss = sess.run(loss_op,
                                                  feed_dict={batch_x: batched_x, batch_y: batched_y})
                            print(format_str.format(step, train_loss, time.time() - start_time))

                        # write summary
                        summary = sess.run(merged, feed_dict={batch_x: batched_x, batch_y: batched_y})
                        train_writer.add_summary(summary, step)
                    if test_interval is not None:
                        if acc_op is not None and step % test_interval == 0:  # display testing result
                            print('Testing......')
                            test_acc_mean = 0.0
                            for i in xrange(1, test_iter):
                                batched_x, batched_y = test_reader.next_batch()
                                test_acc = sess.run(acc_op,
                                                    feed_dict={batch_x: batched_x, batch_y: batched_y,
                                                               is_train: False})
                                test_acc_mean += test_acc
                            test_acc_mean = test_acc_mean / test_iter
                            print(format_str_test.format(step, acc_layer.type, test_acc_mean))
        elif input_layer.type == 'glaucoma_input':
            # data augumentation for glaucoma images - refered from original source code -twkim
            def data_aug(data, labels):
                aug_data1 = tf.map_fn(lambda img: tf.image.crop_to_bounding_box(img, 0, 0, 224, 224), data)
                aug_data2 = tf.map_fn(lambda img: tf.image.crop_to_bounding_box(img, 0, 16, 224, 224), data)
                aug_data3 = tf.map_fn(lambda img: tf.image.crop_to_bounding_box(img, 16, 16, 224, 224), data)
                aug_data4 = tf.map_fn(lambda img: tf.image.crop_to_bounding_box(img, 16, 10, 224, 224), data)
                aug_data5 = tf.map_fn(lambda img: tf.image.central_crop(img, 0.93), data)
                flips = tf.map_fn(lambda img: tf.image.random_flip_left_right(img), data)
                aug_data6 = tf.map_fn(lambda img: tf.image.crop_to_bounding_box(img, 0, 0, 224, 224), flips)
                aug_data7 = tf.map_fn(lambda img: tf.image.crop_to_bounding_box(img, 0, 16, 224, 224), flips)
                aug_data8 = tf.map_fn(lambda img: tf.image.crop_to_bounding_box(img, 16, 16, 224, 224), flips)
                aug_data9 = tf.map_fn(lambda img: tf.image.crop_to_bounding_box(img, 16, 10, 224, 224), flips)
                aug_data10 = tf.map_fn(lambda img: tf.image.central_crop(img, 0.93), data)
                # concat_data = data
                # concat_labels = labels
                concat_data = tf.concat((aug_data1, aug_data2, aug_data3, aug_data4, aug_data5,
                                         aug_data6, aug_data7, aug_data8, aug_data9,
                                         aug_data10), axis=0)
                concat_labels = tf.concat((labels, labels, labels, labels, labels,
                                           labels, labels, labels, labels, labels), axis=0)
                return (concat_data, concat_labels)

            # get input placeholders.
            batch_x = input_layer.get_op_output('image')  # image/data/text/... (data info.)
            batch_y = input_layer.get_op_output('label')  # labels/targets/... (label info.)

            # define placeholder for preprocessing raw images
            images_raw = tf.placeholder(tf.float32, shape=(None, 240, 240, 3), name='glaucoma_raw_images')
            labels_raw = tf.placeholder(tf.int32, shape=(None, 2), name='glaucoma_raw_labels')

            with tf.device('/cpu:0'):
                augmented_train_images, augmented_train_labels = data_aug(images_raw, labels_raw)
                augmented_test_images, augmented_test_labels = data_aug(images_raw, labels_raw)

            # print format
            format_str = '[{0:>6} Steps] Train Loss:{1:>8.3f}({2:>8.3f} sec/batch)'
            format_str_acc = '[{0:>6} Steps] Train Loss:{1:>8.3f}, Train {2}:{3:>8.3%}({4:>8.3f} sec/batch)'
            format_str_test = '[{0:>6} Steps] Average Test {1}:{2:>8.3%}'

            # get option attributes
            is_shuffle = learning_option.get('shuffle')

            # get reader class for training data
            # train_images = open(train_dir + '/train_dataset.npy')
            train_images = open(train_dir + '/train_dataset.npy')
            train_labels = open(train_dir + '/train_labels.npy')
            raw_train_images = np.load(train_images)
            raw_train_labels = np.load(train_labels)
            if test_dir is None:  # if no test dataset
                test_reader = None
            else:  # get reader class for testing data
                test_images = open(train_dir + '/test_dataset.npy')
                test_labels = open(train_dir + '/test_labels.npy')
                raw_test_images = np.load(test_images)
                raw_test_labels = np.load(test_labels)

            # TF summary merge op. and session configuration
            merged = tf.summary.merge_all()  # merge summary data
            config = tf.ConfigProto()  # ocnfiguration object for session
            config.gpu_options.allow_growth = True  # allocate minimum GPU memory for running
            config.log_device_placement = False  # not loggging ops/vars-devices mapping
            config.allow_soft_placement = True  # automatically unallocated ops/vars to device

            # step, summary, checkpoint hook setting
            summarydir = UDLI.getArgs('log_out')
            # TODO: not implement
            # if option is restore, check the existing checkpoint file
            '''
            if learning_option.get('option') == 'RETRAIN':
                if len(glob.glob(ckpt_dir + '/*.meta')) == 0:
                    raise Exception('[DLMDL ERROR] checkpoint file(.meta) do not exist in {0}'.format(ckpt_dir))
                last_ckpt_filename = os.path.basename(glob.glob(ckpt_dir + '/*.meta')[-1])
                prev_iter = int(re.findall('\d+', last_ckpt_filename)[0])
                train_iter = train_iter - prev_iter
            '''

            merged = tf.summary.merge_all()
            train_writer = tf.summary.FileWriter(summarydir)

            init_op = tf.global_variables_initializer()
            sv = tf.train.Supervisor(is_chief=(worker_idx == 0),
                                     global_step=global_step,
                                     init_op=init_op,
                                     local_init_op=init_op)
            with sv.managed_session(server, config=config) as sess:

                # load arugmented data w.r.t training/test dataset
                tmp_x = np.zeros([1, 224, 224, 3], dtype=np.float32)
                tmp_y = np.zeros([1, 2], dtype=np.float32)

                train_images_aug, train_labels_aug = sess.run([augmented_train_images, augmented_train_labels],
                                                              feed_dict={images_raw: raw_train_images,
                                                                         labels_raw: raw_train_labels,
                                                                         batch_x: tmp_x, batch_y: tmp_y})
                train_reader = GLAUCOMAreader(train_dir, train_bs, is_shuffle,
                                              train_images_aug, train_labels_aug)
                if acc_op is not None:
                    test_images_aug, test_labels_aug = sess.run([augmented_test_images, augmented_test_labels],
                                                                feed_dict={images_raw: raw_test_images,
                                                                           labels_raw: raw_test_labels,
                                                                           batch_x: tmp_x, batch_y: tmp_y})
                    test_reader = GLAUCOMAreader(test_dir, test_bs, is_shuffle,
                                                 test_images_aug, test_labels_aug)
                step = 0
                while step < train_iter:
                    start_time = time.time()
                    # one step training
                    batched_x, batched_y = train_reader.next_batch()
                    _, step = sess.run([train_op, global_step],
                                       feed_dict={batch_x: batched_x, batch_y: batched_y})
                    if step % train_dp == 0:  # display training result
                        if acc_op is not None:  # exist accuracy op. in graph
                            # calculate batch loss and accuracy
                            train_loss, train_acc = sess.run([loss_op, acc_op],
                                                             feed_dict={batch_x: batched_x, batch_y: batched_y})
                            print(
                                format_str_acc.format(step, train_loss, acc_layer.type, train_acc,
                                                      time.time() - start_time))
                        else:  # no acc op.
                            # calculate batch loss
                            train_loss = sess.run(loss_op,
                                                  feed_dict={batch_x: batched_x, batch_y: batched_y})
                            print(format_str.format(step, train_loss, time.time() - start_time))
                        # write summary
                        summary = sess.run(merged, feed_dict={batch_x: batched_x, batch_y: batched_y})
                        train_writer.add_summary(summary, step)
                    if test_interval is not None:
                        if acc_op is not None and step % test_interval == 0:  # display testing result
                            print('Testing......')
                            test_acc_mean = 0.0
                            for i in xrange(1, test_iter):
                                batched_x, batched_y = test_reader.next_batch()
                                test_acc = sess.run(acc_op,
                                                    feed_dict={batch_x: batched_x, batch_y: batched_y,
                                                               is_train: False})
                                test_acc_mean += test_acc
                            test_acc_mean = test_acc_mean / test_iter
                            print(format_str_test.format(step, acc_layer.type, test_acc_mean))
        elif input_layer.type == 'ptb_input':

            # get input placeholders.
            batch_x = learning_option.get('data_placeholder')  # image/data/text/... (data info.)
            batch_y = input_layer.get_op_output('targets')  # labels/targets/... (label info.)

            # print format
            format_str = '[{0:>3.2f} Epochs, {1:>6} Steps] Train Loss:{2:>8.3f}({2:>8.3f} sec/batch)'
            format_str_acc = '[{0:>3.2f} Epochs, {1:>6} Steps] Train Loss:{2:>8.3f}, Train {3}:{4:>8.3f}({5:>8.3f} sec/batch)'
            format_str_test = '[{0:>3.2f} Epochs, {1:>6} Steps] Average Test {2}:{3:>8.3f}'

            initial_state = learning_option.get('initial_state')  # get TF initial state op.
            # WARNINIG: if there is other op. added to static_rnn, use meta character '|'
            # e.g 'static_rnn|static_bidirectional_rnn' --> detect static_rnn or static_bidirectional_rnn op.
            final_state = network.get_layer_by_type('static_rnn').pop().get_op_output(
                'state')  # get TF output state op.

            # get option attributes
            num_steps = learning_option.get('num_steps')
            hidden_size = learning_option.get('hidden_size')
            num_class = learning_option.get('num_class')  # 10000: fixed classes

            # get reader class for training data
            train_reader = PTBreader(train_dir, train_bs, num_steps, istrain=True)
            if test_dir is None:  # if no test dataset
                test_reader = None
            else:  # get reader class for testing data
                test_reader = PTBreader(test_dir, test_bs, num_steps, istrain=False)

            # TF summary merge op. and session configuration
            config = tf.ConfigProto()  # ocnfiguration object for session
            config.gpu_options.allow_growth = True  # allocate minimum GPU memory for running
            config.log_device_placement = False  # not loggging ops/vars-devices mapping
            config.allow_soft_placement = True  # automatically unallocated ops/vars to device

            # step, summary, checkpoint hook setting
            summarydir = UDLI.getArgs("log_out")
            # TODO: not implement
            # if option is restore, check the existing checkpoint file
            '''
            if learning_option.get('option') == 'RETRAIN':
                if len(glob.glob(ckpt_dir + '/*.meta')) == 0:
                    raise Exception('[DLMDL ERROR] checkpoint file(.meta) do not exist in {0}'.format(ckpt_dir))
                last_ckpt_filename = os.path.basename(glob.glob(ckpt_dir + '/*.meta')[-1])
                prev_iter = int(re.findall('\d+', last_ckpt_filename)[0])
                train_iter = train_iter - prev_iter
            '''

            merged = tf.summary.merge_all()
            train_writer = tf.summary.FileWriter(summarydir)

            init_op = tf.global_variables_initializer()
            sv = tf.train.Supervisor(is_chief=(worker_idx == 0),
                                     global_step=global_step,
                                     init_op=init_op,
                                     local_init_op=init_op)
            with sv.managed_session(server, config=config) as sess:
                step = 0
                current_epoch = 0
                while step < train_iter:
                    if initial_state is not None:
                        batched_x, batched_y = train_reader.next_batch()
                        state = sess.run(initial_state,
                                         feed_dict={batch_x: batched_x,
                                                    batch_y: batched_y})
                    else:  # TODO: unable condition
                        state = None
                    train_epoch_size = train_reader.get_epoch_size()
                    for step_per_epoch in range(train_epoch_size):
                        # run one epoch
                        start_time = time.time()
                        feed_dict = {}
                        # one step training
                        batched_x, batched_y = train_reader.next_batch()
                        feed_dict[batch_x] = batched_x
                        feed_dict[batch_y] = batched_y
                        feed_dict[is_train] = True
                        for i, (c, h) in enumerate(initial_state):
                            feed_dict[c] = state[i].c
                            feed_dict[h] = state[i].h
                        _, fin_stat, step = sess.run([train_op, final_state, global_step], feed_dict)
                        state = fin_stat
                        if step % train_dp == 0:  # display training result
                            if acc_op is not None:  # exist accuracy op. in graph
                                # calculate batch loss and accuracy
                                train_loss, train_acc = sess.run([loss_op, acc_op],
                                                                 feed_dict)
                                print(format_str_acc.format(current_epoch + step_per_epoch * 1.0 / train_epoch_size,
                                                            step, train_loss, acc_layer.type, train_acc,
                                                            time.time() - start_time))
                            else:  # no acc op.
                                # calculate batch loss
                                train_loss = sess.run(loss_op, feed_dict)
                                print(format_str.format(
                                    current_epoch + step_per_epoch * 1.0 / train_epoch_size,
                                    step, train_loss, time.time() - start_time))
                            # write summary
                            summary = sess.run(merged, feed_dict={batch_x: batched_x, batch_y: batched_y})
                            train_writer.add_summary(summary, step)
                    if test_interval is not None:
                        if acc_op is not None and step % test_interval == 0:  # display testing result
                            print('Testing......')
                            # calculate average accuracy during test iterations
                            test_iters = 0
                            test_losses = 0.0
                            if initial_state is not None:
                                batched_x, batched_y = test_reader.next_batch()
                                test_state = sess.run(initial_state,
                                                      feed_dict={batch_x: batched_x, batch_y: batched_y})
                            else:  # TODO: unable condition
                                test_state = None
                            test_epoch_size = test_reader.get_epoch_size()
                            for test_step in range(test_iter):  # during test iterations
                                batched_x, batched_y = test_reader.next_batch()  # get test batch
                                feed_dict = {}
                                feed_dict[batch_x] = batched_x
                                feed_dict[batch_y] = batched_y
                                feed_dict[is_train] = False
                                for i, (c, h) in enumerate(initial_state):
                                    feed_dict[c] = test_state[i].c
                                    feed_dict[h] = test_state[i].h

                                test_loss, test_fin_stat = sess.run([loss_op, final_state],
                                                                    feed_dict)
                                test_state = test_fin_stat

                                test_losses += test_loss
                                test_iters += num_steps
                            print(format_str_test.format(0.0 + test_iter * 1.0 / test_epoch_size,
                                                         step, acc_layer.type,
                                                         np.exp(test_losses / test_iters)))
                    current_epoch += 1
        elif input_layer.type == 'csv_input':  # not yet used
            pass
        elif input_layer.type == 'bodydata_input':
            # get all loss, train_op, acc layers
            opt_layer = network.get_layer_by_type(
                'adam|sgd|momentum|rmsprop|nesterov|adadelta|adagrad')  # get optimizer layer
            convnet_train_op = None
            others_train_op = None
            whole_train_op = None
            for layer in opt_layer:
                if layer.get_attr('scope') == 'convnet':
                    convnet_train_op = layer.get_op_output('output')
                elif layer.get_attr('scope') == 'others':
                    others_train_op = layer.get_op_output('output')
                else:
                    whole_train_op = layer.get_op_output('output')

            convnet_loss_op = network.get_layer('convnet_loss').get_op_output('output')
            total_loss_op = network.get_layer('total_loss').get_op_output('output')
            others_op = network.get_layer('above_net_fc1').get_op_output('output')

            # WARNING: current accuracy related layers -> accuracy/top-k layer, perplexity layer
            acc_layer = network.get_layer_by_type('accuracy|perplexity')  # get acc layer
            bt_acc_op = None
            bs_acc_op = None
            full_bt_acc_op = None
            full_bs_acc_op = None
            for layer in acc_layer:
                if layer.name == 'bt_acc':
                    bt_acc_op = layer.get_op_output('output')
                elif layer.name == 'bs_acc':
                    bs_acc_op = layer.get_op_output('output')
                elif layer.name == 'full_bt_acc':
                    full_bt_acc_op = layer.get_op_output('output')
                elif layer.name == 'full_bs_acc':
                    full_bs_acc_op = layer.get_op_output('output')

            batch_image_x = input_layer.get_op_output('image')
            batch_bt_label_y = input_layer.get_op_output('bt_label')
            batch_bs_label_y = input_layer.get_op_output('bs_label')
            batch_numeric_x = input_layer.get_op_output('numeric')
            batch_numeric_label_y = input_layer.get_op_output('numeric_label')

            # print format
            format_str_convnet = '[{0:>6} Steps] Convnet Train Loss:{1:>8.3f}({2:>8.3f} sec/batch)'
            format_str_others = '[{0:>6} Steps] OthersNet Train Loss:{1:>8.3f}({2:>8.3f} sec/batch)'
            format_str_full = '[{0:>6} Steps] Full Train Loss:{1:>8.3f}({2:>8.3f} sec/batch)'

            # load input reader
            in_reader = BODYDATAreader(train_dir, False)

            # TF summary merge op. and session configuration
            config = tf.ConfigProto()  # ocnfiguration object for session
            config.gpu_options.allow_growth = True  # allocate minimum GPU memory for running
            config.log_device_placement = False  # not loggging ops/vars-devices mapping
            config.allow_soft_placement = True  # automatically unallocated ops/vars to device

            # not used in this model
            # stophooks = tf.train.StopAtStepHook(last_step=train_iter)
            summarydir = UDLI.getArgs('log_out')
            ckptdir = UDLI.getArgs('parameter_out')

            # merged
            # TODO: THIS VERSION IS MANUAL SEARCH
            # merged = tf.summary.merge_all()

            train_writer = tf.summary.FileWriter(summarydir)

            # open TF session
            with tf.train.MonitoredTrainingSession(master=server, is_chief=(worker_idx == 0),
                                                   save_checkpoint_steps=ckpt_interval,
                                                   save_summaries_steps=train_dp,
                                                   summary_dir=summarydir,
                                                   checkpoint_dir=ckpt_dir,
                                                   log_step_count_steps=0,
                                                   config=config) as sess:

                print('(1) ConvNet training w/ BT and BS image -------------------------')
                for step in range(train_iter):
                    batched_image, batched_numeric, batched_numeric_label, batched_bt_label, batched_bs_label = in_reader.next_batch(
                        train_bs, is_train=True)
                    start_time = time.time()
                    convnet_train_loss, _ = sess.run([convnet_loss_op, convnet_train_op],
                                                     feed_dict={batch_image_x: batched_image,
                                                                batch_bt_label_y: batched_bt_label,
                                                                batch_bs_label_y: batched_bs_label,
                                                                batch_numeric_x: batched_numeric,
                                                                batch_numeric_label_y: batched_numeric_label})

                    if step % train_dp == 0:
                        print(format_str_convnet.format(step, convnet_train_loss, time.time() - start_time))
                        # train_writer.add_summary(summ, step)

                print('(2) on the mid of train--------------------------------')
                for _ in range(30):
                    batched_image, batched_numeric, batched_numeric_label, batched_bt_label, batched_bs_label = in_reader.next_batch(
                        train_bs, is_train=False)
                    reg_output, bt_acc, bs_acc = sess.run([others_op, bt_acc_op, bs_acc_op],
                                                          feed_dict={batch_image_x: batched_image,
                                                                     batch_bt_label_y: batched_bt_label,
                                                                     batch_bs_label_y: batched_bs_label,
                                                                     batch_numeric_x: batched_numeric,
                                                                     batch_numeric_label_y: batched_numeric_label})
                print ('(2) Body Type Accuracy: {0:>5.3%}, Body Shape Accuracy: {1:>5.3%}'.format(bt_acc, bs_acc))
                print('(3)--------------------------------------------------')
                for step in range(train_iter):
                    batched_image, batched_numeric, batched_numeric_label, batched_bt_label, batched_bs_label = in_reader.next_batch(
                        train_bs, is_train=True)
                    start_time = time.time()
                    total_train_loss, _ = sess.run([total_loss_op, others_train_op],
                                                   feed_dict={batch_image_x: batched_image,
                                                              batch_bt_label_y: batched_bt_label,
                                                              batch_bs_label_y: batched_bs_label,
                                                              batch_numeric_x: batched_numeric,
                                                              batch_numeric_label_y: batched_numeric_label})
                    if step % train_dp == 0:
                        print(format_str_others.format(step, total_train_loss, time.time() - start_time))
                print('(4)--------------------------------------------------')
                for step in range(train_iter):
                    batched_image, batched_numeric, batched_numeric_label, batched_bt_label, batched_bs_label = in_reader.next_batch(
                        train_bs, is_train=True)
                    start_time = time.time()
                    total_train_loss, _ = sess.run([total_loss_op, whole_train_op],
                                                   feed_dict={batch_image_x: batched_image,
                                                              batch_bt_label_y: batched_bt_label,
                                                              batch_bs_label_y: batched_bs_label,
                                                              batch_numeric_x: batched_numeric,
                                                              batch_numeric_label_y: batched_numeric_label})
                    if step % train_dp == 0:
                        print(format_str_full.format(step, total_train_loss, time.time() - start_time))
                print('TRAIN COMPLETE')
                print('(5)--------------------------------------------------')
                bt_acc_list = []
                bs_acc_list = []
                reg_error_rate_list = []
                for _ in range(test_iter):
                    batched_image, batched_numeric, batched_numeric_label, batched_bt_label, batched_bs_label = in_reader.next_batch(
                        train_bs, is_train=False)
                    reg_output, bt_acc, bs_acc = sess.run([others_op, bt_acc_op, bs_acc_op],
                                                          feed_dict={batch_image_x: batched_image,
                                                                     batch_bt_label_y: batched_bt_label,
                                                                     batch_bs_label_y: batched_bs_label,
                                                                     batch_numeric_x: batched_numeric,
                                                                     batch_numeric_label_y: batched_numeric_label})
                    bt_acc_list.append(bt_acc)
                    bs_acc_list.append(bs_acc)
                    output_std = in_reader.get_output_std()
                    output_avg = in_reader.get_output_avg()

                    reg_pred = reg_output * output_std + output_avg
                    reg_label = batched_numeric_label * output_std + output_avg
                    error_rate = np.abs((reg_pred - reg_label) / reg_label * 100)
                    reg_error_rate_list.append(error_rate)

            print('(5) Body Type Accuracy: {0:>5.3%}, Body Shape Accuracy: {1:>5.3%}'.format(np.mean(bt_acc_list),
                                                                                             np.mean(bs_acc_list)))
            reg_error_rate_list = np.concatenate(reg_error_rate_list, 0)
            msg = '(5) Avg. Error Rate for Each Item: {0}, Median Error Rate for Each Item: {1}, Third Quarter of Error Rate for Each Item: {2}'
            print(msg.format(np.mean(reg_error_rate_list, 0), np.median(reg_error_rate_list, 0),
                             np.percentile(reg_error_rate_list, 75, axis=0)))
        # close processes
        # WARNINNG: In release, ssh handler don't include
        # for conn in conns:
        #    del conn

    @staticmethod
    def get_node_index(cluster):
        try:
            ip = tf_adaptor.resolve_address() # get current ip address
            idx = None
            for i in range(len(cluster)):
                if ip in cluster[i]:
                    idx = i # get worker index of this ip
                    break
        except ValueError:
            raise Exception('[DLMDL ERROR]: ip address {0} not in cluster information.'
                            .format(ip))
        if idx is None:
            raise Exception('[DLMDL ERROR]: ip address {0} not in cluster information.'
                            .format(ip))
        return idx

    @staticmethod
    def resolve_address():
        hostName = socket.gethostname()
        return socket.gethostbyname(hostName)

    @staticmethod
    def grpc_server(server):
        print 'grpc_server'
        server.join()

    @staticmethod
    def launch_slaves(UDLI, ips):
        conns = []
        for ip in ips:
            file_path = UDLI.getFullPath(UDLI.getArgs('input'), 'dlmdl')
            # TODO: sftp will be DEPRECATED.
            '''
            sftp = sftp_handler(ip)
            sftp.upload(UDLI.script_name(), UDLI.script_name())
            sftp.upload(file_path, file_path)
            sftp.uploadRecursive(UDLI.getCWD(), UDLI.getCWD())
            del sftp
            '''
            conn = shell_handler(ip)
            conns.append(conn)
            threading.Thread(target=tf_adaptor.launch_slave,
                             args=(UDLI, conn, UDLI.script_name(), file_path.replace('.dlmdl', ''),)).start()
        print 'request done'
        return conns

    @staticmethod
    def launch_slave(UDLI, conn, exec_path, f):
        command = 'cd {wd} ; export PYTHONPATH={wd}:{wd}/src:$PYTHONPATH ; python {exec_path} -r --input={f}'.format(
            wd=UDLI.getCWD(), exec_path=exec_path, f=f)
        print command
        conn.execute(command)

    @staticmethod
    def t(obj):
        return json.dumps(obj).replace('true', 'True').replace('false', 'False').replace('null', 'None')