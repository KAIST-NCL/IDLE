# -*- coding: utf-8 -*-

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
import numpy as np
import os
from PIL import Image
import collections
from tensorflow.examples.tutorials.mnist import input_data


# PTBReader - tmp, not used - twkim
class PTBreader(object):
    # def __init__(self, data_path, batch_size, num_steps, istrain=1):
    def __init__(self, learning_option, istrain=1):
        self.index_in_epoch = 0
        self.epochs_completed = 0
        self.data_path = learning_option.get('train_path') if istrain == 1 else learning_option.get('test_data_path')
        self.batch_size = learning_option.get('batch_size') if istrain == 1 else learning_option.get('test_batch_size')
        self.num_steps = learning_option.get('num_steps ')
        self.istrain = istrain
        self.data, self.epoch_size = self._ptb_producer()
        self.label = []

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
        if self.istrain == 1:
            train_path = os.path.join(self.data_path, "ptb.datasets.txt")
        else:
            train_path = os.path.join(self.data_path, "ptb.test.txt")
        word_to_id = self._build_vocab(train_path)
        raw_data = self._file_to_word_ids(train_path, word_to_id)

        vocabulary = len(word_to_id)
        return raw_data, vocabulary

    def _ptb_producer(self):
        raw_data, _ = self._ptb_raw_data()
        epoch_size = ((len(raw_data) // self.batch_size) - 1) // self.num_steps
        data_len = len(raw_data)
        batch_len = data_len // self.batch_size
        data = np.reshape(raw_data[0:self.batch_size * batch_len], [self.batch_size, batch_len])
        return data, epoch_size

    def get_epoch_size(self):
        return self.epoch_size

    def next_batch(self, idx):
        x = self.data[0:self.batch_size, idx * self.num_steps:(idx + 1) * self.num_steps]
        y = np.roll(self.data, -1)[0:self.batch_size, idx * self.num_steps:(idx + 1) * self.num_steps]
        return x, y


class JPEGreader(object):
    # def __init__(self, img_path, file_format, image_size, label_path, batch_size):
    def __init__(self, learning_option, istrain=1):
        self.index_in_epoch = 0
        self.epochs_completed = 0
        self.file_format = learning_option.get('file_format')
        self.img_path = learning_option.get('train_path') if istrain == 1 else learning_option.get('test_data_path')
        self.label_path = learning_option.get('label_path') if istrain == 1 else learning_option.get('test_label_path')
        self.batch_size = learning_option.get('batch_size') if istrain == 1 else learning_option.get('test_batch_size')
        self.image_size = learning_option.get('image_size')
        self.img_files, self._num_examples = self._set_img_files()
        self.data = []
        self.label = []

    def _is_grey_scale(self):
        im_check = Image.open(self.img_path + '/' + self.img_files[0]).convert('RGB')
        w, h = im_check.size
        for i in range(w):
            for j in range(h):
                r, g, b = im_check.getpixel((i, j))
                if r != g != b: return False
        return True

    def _set_img_files(self):
        filenames = os.listdir(self.img_path)
        for filename in filenames:
            full_filename = os.path.join(self.img_path, filename)
            ext = os.path.splitext(full_filename)[-1]
            if ext != '.' + self.file_format:
                filenames.remove(filename)
        return filenames, len(filenames)

    def img_read(self, batch_filenames):
        is_grey = self._is_grey_scale()
        for file in batch_filenames:
            with Image.open(self.img_path + '/' + file) as im:
                im_resize = im.resize((self.image_size[0], self.image_size[1]))
                im_array = np.array(im_resize.getdata()).astype(np.uint8).reshape(
                    (im_resize.size[0], im_resize.size[1], 1 if is_grey == True else 3))
                self.data.append(im_array)

    def label_read(self, batch_filenames):
        for file in batch_filenames:
            with open(self.label_path) as fp:
                for line in fp:
                    if os.path.splitext(file)[0] in line:
                        # int convert is conflict?
                        self.label.append(line.split(' ')[-1].rstrip())
                        break

    def next_batch(self, shuffle=True):
        self.data = []
        self.label = []
        start = self.index_in_epoch
        if start == 0 and self.epochs_completed == 0:
            np.random.shuffle(self.img_files)  # shuffle indexs

        # go to the next batch
        if start + self.batch_size > self._num_examples:
            self.epochs_completed += 1
            rest_num_examples = self._num_examples - start
            files_rest_part = self.img_files[start:self._num_examples]
            np.random.shuffle(self.img_files)  # shuffle indexes

            start = 0
            self.index_in_epoch = self.batch_size - rest_num_examples  # avoid the case where the #sample != integar times of batch_size
            end = self.index_in_epoch
            files_new_part = self.img_files[start:end]
            batch_filenames = np.concatenate((files_rest_part, files_new_part), axis=0)
        else:
            self.index_in_epoch += self.batch_size
            end = self.index_in_epoch
            batch_filenames = self.img_files[start:end]
        self.img_read(batch_filenames)
        self.label_read(batch_filenames)
        return np.asarray(self.data, dtype=np.uint8), np.asarray(self.label, dtype=np.int64)


# Galucoma image reader class -twkim
class GALUCOMAReader(object):
    # def __init__(self,data, labels):
    def __init__(self, learning_option, istrain=1):
        self.index_in_epoch = 0
        self.epochs_completed = 0
        self.data_path = learning_option.get('train_path') if istrain == 1 else learning_option.get('test_data_path')
        self.batch_size = learning_option.get('batch_size') if istrain == 1 else learning_option.get('test_batch_size')
        self.istrain = istrain
        self.data, self.labels, self.num_examples = self._read_dataset()
        pass

    def _read_dataset(self):
        if self.istrain == 1:
            imgaes = open(self.data_path + '/train_dataset.npy')
            labels = open(self.data_path + '/train_labels.npy')
        else:
            imgaes = open(self.data_path + '/test_dataset.npy')
            labels = open(self.data_path + '/test_labels.npy')
        raw_images = np.load(images)
        raw_labels = np.load(labels)

        raw_labels_dense = np.empty((raw_labels.shape[0]), np.int32)
        for idx, col in enumerate(raw_labels):
            raw_labels_dense[idx] = 0 if col[0] == 1 else 1
        return raw_images, raw_labels_dense, raw_images.shape[0]

    @property
    def data(self):
        return self.data

    @property
    def labels(self):
        return self.labels

    def next_batch(self):
        start = self.index_in_epoch
        # go to the next batch
        if start + self.batch_size > self.num_examples:
            self.epochs_completed += 1
            rest_num_examples = self.num_examples - start
            data_rest_part = self.data[start:self.num_examples]
            labels_rest_part = self.labels[start:self.num_examples]
            idx0 = np.arange(0, self.num_examples)  # get all possible indexes

            # shuffle for new epoch
            np.random.shuffle(idx0)  # shuffle indexes
            self.data = self.data[idx0]  # get list of `num` random samples
            self.labels = self.labels[idx0]

            start = 0
            self.index_in_epoch = batch_size - rest_num_examples  # avoid the case where the #sample != integar times of batch_size
            end = self.index_in_epoch
            data_new_part = self.data[start:end]
            labels_new_part = self.labels[start:end]
            return np.concatenate((data_rest_part, data_new_part), axis=0), np.concatenate(
                (labels_rest_part, labels_new_part), axis=0)
        else:
            self.index_in_epoch += batch_size
            end = self.index_in_epoch
            return self.data[start:end], self.labels[start:end]


class tf_adaptor:
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
                'tf_adaptor': ['tf_adaptor']
            },
            'src.DLMDL.LayerOperation': {
                'layer_operation': ['LayerOperationAdaptor', 'LayerOperation']
            },
            'src.DLMDL.LayerOperation.tf': {
                'c_accuracy': ['op_tf_c_accuracy'],
                'c_adagradoptimizer': ['op_tf_c_adagradoptimizer'],
                'c_adamoptimizer': ['op_tf_c_adamoptimizer'],
                'c_avg_pool': ['op_tf_c_avg_pool'],
                'c_concat': ['op_tf_c_concat'],
                'c_conv': ['op_tf_c_conv'],
                'c_dropout': ['op_tf_c_dropout'],
                'c_eltwise': ['op_tf_c_eltwise'],
                'c_fc': ['op_tf_c_fc'],
                'c_gradientdescentoptimizer': ['op_tf_c_gradientdescentoptimizer'],
                'c_loss': ['op_tf_c_loss'],
                'c_max_pool': ['op_tf_c_max_pool'],
                'c_momentumoptimizer': ['op_tf_c_momentumoptimizer'],
                'c_nesterovoptimizer': ['op_tf_c_nesterovoptimizer'],
                'c_rmspropoptimizer': ['op_tf_c_rmspropoptimizer'],
                'r_accuracy': ['op_tf_r_accuracy'],
                'r_multicell': ['op_tf_r_multicell'],
                'r_fc': ['op_tf_r_fc'],
                'r_gradientdescentoptimizer': ['op_tf_r_gradientdescentoptimizer'],
                'r_loss': ['op_tf_r_loss'],
                'r_staticrnn': ['op_tf_r_staticrnn'],
                'r_sequenceloss': ['op_tf_r_sequenceloss'],
                'mnist_input': ['op_tf_mnist_input'],
                'jpeg_input': ['op_tf_jpeg_input'],
                'ptb_input': ['op_tf_ptb_input'],
                'galucoma_input': ['op_tf_galucoma_input']
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
                # print(str) # unicode error occured, because of korean commment - twkim

    @staticmethod
    def run(UDLI):
        print sys.path
        cluster = UDLI.cluster
        learning_option = UDLI.learning_option

        # Data parallelism
        if UDLI.learning_option.get('parallel') == "DP":
            ps_hosts = [cluster.get('hosts')[cluster.getIndex('tasks', 'ps')]]
            cluster.get('hosts').pop(cluster.getIndex('tasks', 'ps'))
            worker_hosts = cluster.get('hosts')
            tf_cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})

            current_address = tf_adaptor.resolve_address()

            if current_address == ps_hosts[0].split(':')[0]:
                # isps = True
                ps_taskIdx = 0  # 0
                server = tf.train.Server(tf_cluster, job_name='ps', task_index=ps_taskIdx)
                tf_adaptor.grpc_server(server)

            # worker
            else:
                worker_taskIdx = tf_adaptor.get_node_index(cluster)
                server = tf.train.Server(tf_cluster, job_name='worker', task_index=worker_taskIdx)
                # graph def
                with tf.device(tf.train.replica_device_setter(worker_device="/job:worker/task:%d" % worker_taskIdx,
                                                              cluster=tf_cluster)):
                    tf_adaptor.session_executor(UDLI, server.target, worker_taskIdx)

        # Model parallelism, Single node
        else:
            taskIdx = tf_adaptor.get_node_index(cluster)
            tf_cluster = tf.train.ClusterSpec({"worker": cluster.get('hosts')})
            server = tf.train.Server(tf_cluster, job_name='worker', task_index=taskIdx)

            # master
            if taskIdx == 0:
                grpcHost = 'grpc://' + cluster.get('hosts')[0]
                tf_adaptor.session_executor(UDLI, grpcHost, taskIdx)

            # slave
            else:
                tf_adaptor.grpc_server(server)

    @staticmethod
    def resolve_address():
        hostName = socket.gethostname()
        return socket.gethostbyname(hostName)

    @staticmethod
    def get_node_index(cluster):
        try:
            ipAddress = tf_adaptor.resolve_address()
            taskIdx = cluster.getIndex('ips', ipAddress)
        except ValueError:
            print('DL-MDL[ERROR]: NodeIP %s is not found in cluster information.' % ipAddress)
            raise ValueError
        print ipAddress + ", %d" % taskIdx
        return taskIdx

    @staticmethod
    def session_executor(UDLI, creator, taskIdx):
        # temp

        # temp seperation
        # TODO: dynamic executor seperation
        print(learning_option.get('file_format'))  # twkim-test
        if learning_option.get('file_format') == "jpg":
            tf_adaptor.jpeg_cnn_executor(UDLI, creator, taskIdx, learning_option)
        elif learning_option.get('file_format') == 'mnist':
            tf_adaptor.mnist_rnn_executor(UDLI, creator, taskIdx, learning_option)
        elif learning_option.get('file_format') == 'ptb':
            tf_adaptor.ptb_lstm_executor(UDLI, creator, taskIdx, learning_option)
        elif learning_option.get('file_format') == 'glaucoma':
            tf_adaptor.galucoma_cnn_executor(UDLI, creator, taskIdx, learning_option)
        else:
            print 'None'
            pass

        network = UDLI.network
        cluster = UDLI.cluster
        learning_option = UDLI.learning_option

        training_context = tf.placeholder_with_default(True, shape=(),
                                                       name='training_context')  # get procedure placeholder which represents the procedure(training or testing)
        global_step = tf.train.get_or_create_global_step()  # create global step
        network.run(learning_option, cluster)

        file_format = learning_option.get('file_format')  # get input file format
        ouuput_list = learning_option.get('output_list')  # get output format list

        # datasets setting
        train_bs = learning_option.get('batch_size')  # datasets batch size
        train_dpath = learning_option.get('data_path')  # datasets data path
        train_display = learning_option.get('train_display')  # interval to display
        total_iter = learning_option.get('iteration')  # total iteration to datasets

        # get  training dataset reader
        dataset_format = learning_option.get('file_format').upper()
        train_dataset = globals()[dataset_format + 'Reader'](learning_option, istrain=1)

        # test setting : test iteration must be 1 epoch
        is_test = 1 if learning_option.get('option') == 'train_test' else 0  # only training or with testing
        if is_test == 1:
            test_bs = learning_option.get('test_batch_size')
            test_dpath = learning_option.get('test_data_path')
            test_interval = learning_option.get('test_interval')
            test_dataset = train_dataset = globals()[dataset_format + 'Reader'](learning_option, istrain=0)

        data_batch = network.get_layer_by_type(file_format + '_input')[0].get_op_output('data')
        label_batch = network.get_layer_by_type(file_format + '_input')[0].get_op_output('label')

        for i, output in enumerate(output_list):
            output_list_op[i] = network.get_layer(output).get_op_output('output')

        # optimizer layer name must be 'optimizer'
        opt = network.get_layer('optimizer').get_op_output('output')
        # step, summary, checkpoint hook setting
        saver = tf.train.Saver()  # TF saver class
        ckpthooks = tf.train.CheckpointSaverHook(save_steps=total_iter - 1, checkpoint_dir=ckptdir, saver=saver)
        summaryhooks = tf.train.SummarySaverHook(save_steps=test_interval, output_dir=summarydir, summary_op=merged)

        # TF session configure
        config = tf.ConfigProto()  # make TF session configure class
        config.gpu_options.allow_growth = True  # allocate GPU memory option
        config.log_device_placement = False  # whether loggging device placement to ops and variables

        # print format
        format_str = '[{0:>6} Steps] Train Loss:{1:>8.3f}({2:>8.3f} sec/batch)'
        format_str_acc = '[{0:>6} Steps] Train Loss:{1:>8.3f}, Train Acc:{2:>8.3%}({3:>8.3f} sec/batch)'
        format_str_test = '[{0:>6} Steps] Average Test Accuracy:{1:>8.3%}'

        with tf.train.MonitoredTrainingSession(master=creator, is_chief=(taskIdx == 0), hooks=[ckpthooks],
                                               config=config) as sess:
            step = 0
            while step < total_iter - 1:
                start_time = time.time()
                batched_images, batched_labels = train_dataset.next_batch(shuffle=True)
                _, step = sess.run([opt, global_step], feed_dict={images: batched_images, labels: batched_labels})
                if step % train_display == 0:
                    if isAcc == 1:
                        # Calculate batch loss and accuracy
                        train_loss, train_acc = sess.run([loss, acc],
                                                         feed_dict={images: batched_images, labels: batched_labels})
                        print(format_str_acc.format(step, train_loss, train_acc, time.time() - start_time))
                else:
                    train_loss = sess.run(loss, feed_dict={images: batched_images, labels: batched_labels})
                    print(format_str.format(step, train_loss, time.time() - start_time))


#################################################################

@staticmethod
def mnist_rnn_executor(UDLI, creator, taskIdx, learning_option):
    print 'mnist_rnn_executor'
    network = UDLI.network
    cluster = UDLI.cluster

    network.run(learning_option, cluster)

    iteration = learning_option.get('iteration')
    train_batch_size = learning_option.get('batch_size')
    train_data = learning_option.get('data_path')

    mnist = input_data.read_data_sets(train_data, one_hot=True)

    train_display = learning_option.get('train_display')

    # for rnn
    num_units = learning_option.get('num_units')
    num_steps = learning_option.get('num_steps')

    isTrainingTest = 1 if learning_option.get('option') == 'train_test' else 0

    test_data = learning_option.get('test_data_path')  # not used: assume it exists same place

    # Test must contain "train_test" option
    if test_data != None and isTrainingTest == 0:
        raise ValueError('option must be "train_test" if model contains testing procedure')

    test_iteration = learning_option.get('test_iteration')
    test_interval = learning_option.get('test_interval')
    test_batch_size = learning_option.get('test_batch_size')

    images = network.get_layer_by_type('mnist_input')[0].get_op_output('image')
    labels = network.get_layer_by_type('mnist_input')[0].get_op_output('label')

    cost = network.get_layer_by_type('r_loss')[0].get_op_output('output')

    opt = network.get_layer('optimizer').get_op_output('output')
    global_step = network.get_layer('optimizer').get_op_output('global_step')

    isAcc = 1 if network.get_layer_by_type('r_accuracy')[0] != [] else 0
    # twkim - need to modify
    # testing procedure only shows accuracy, so accuracy layer must be contatined in .dlmdl file
    if isTrainingTest:
        acc = network.get_layer_by_type('r_accuracy')[0].get_op_output('output')
        if acc == None:
            raise ValueError('Accuracy layer must be contained in .dlmdl file while testing procedure')

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.log_device_placement = False  # Check
    config.allow_soft_placement = True

    # init = tf.global_variables_initializer()
    merged = tf.summary.merge_all()

    # step, summary, checkpoint hook setting
    ckptdir = UDLI.getArgs("parameter_out")
    summarydir = UDLI.getArgs("log_out")
    saver = tf.train.Saver()
    hooks = tf.train.StopAtStepHook(last_step=iteration)
    ckpthooks = tf.train.CheckpointSaverHook(save_steps=iteration - 1, checkpoint_dir=ckptdir,
                                             saver=saver)
    summaryhooks = tf.train.SummarySaverHook(save_steps=test_interval, output_dir=summarydir, summary_op=merged)

    # sv = tf.datasets.Supervisor(logdir=summarydir)
    # with tf.Session(config=config) as session:
    with tf.train.MonitoredTrainingSession(master=creator, is_chief=(taskIdx == 0),
                                           hooks=[hooks, ckpthooks, summaryhooks], config=config) as sess:
        # session.run(init)
        # print format
        step = 0
        format_str = '[{0:>6} Steps] Train Loss:{1:>8.3f}({2:>8.3f} sec/batch)'
        format_str_acc = '[{0:>6} Steps] Train Loss:{1:>8.3f}, Train Acc:{2:>8.3%}({3:>8.3f} sec/batch)'
        format_str_test = '[{0:>6} Steps] Average Test Accuracy:{1:>8.3%}'

        # iteration begins
        # for step in range(1, iteration + 1):
        while step < iteration - 1:
            # get datasets mnist batch
            batch_x, batch_y = mnist.train.next_batch(train_batch_size)
            # for rnn, it is resized
            batch_x = batch_x.reshape((train_batch_size, num_steps, num_units))
            start_time = time.time()
            costs = 0.0

            _, step = sess.run([opt, global_step], feed_dict={images: batch_x, labels: batch_y})
            if step % train_display == 0 or step == 1:
                if isTrainingTest:
                    # Calculate batch loss and accuracy
                    loss, acc_ = sess.run([cost, acc], feed_dict={images: batch_x, labels: batch_y})
                    print(format_str_acc.format(step, loss, acc_, time.time() - start_time))

                else:
                    # Calculate only batch loss
                    loss = sess.run(cost, feed_dict={images: batch_x, labels: batch_y})
                    print(format_str.format(step, loss, time.time() - start_time))

            if isTrainingTest == 1 and step % test_interval == 0:
                # test
                print 'Testing......'
                # get test mnist batch
                test_acc_mean = 0.0
                for test_iter in xrange(1, test_iteration):
                    batch_x, batch_y = mnist.test.next_batch(test_batch_size)
                    # for rnn, it is resized
                    batch_x = batch_x.reshape((test_batch_size, num_steps, num_units))

                    acc_ = sess.run(acc, feed_dict={images: batch_x, labels: batch_y})
                    test_acc_mean += acc_
                test_acc_mean = test_acc_mean / test_iteration
                print(format_str_test.format(step, test_acc_mean))

        print("[DLMDL] Optimization Finished!")
        print("[DLMDL] Saving model to %s." % ckptdir)


@staticmethod
def jpeg_cnn_executor(UDLI, creator, taskIdx, learning_option):
    print 'jpeg_cnn_executor'
    network = UDLI.network
    cluster = UDLI.cluster

    network.run(learning_option, cluster)

    iteration = learning_option.get('iteration')
    file_format = learning_option.get('file_format')
    image_size = network.get_layer_by_type('jpeg_input')[0].get_attr('image_size')

    train_batch_size = learning_option.get('batch_size')
    train_data = learning_option.get('data_path')
    train_label = learning_option.get('label_path')
    test_data = learning_option.get('test_data_path')

    test_iteration = learning_option.get('test_iteration')
    test_interval = learning_option.get('test_interval')
    test_label = learning_option.get('test_label_path')
    test_batch_size = learning_option.get('test_batch_size')
    train_display = learning_option.get('train_display')

    isTrainingTest = 1 if learning_option.get('option') == 'train_test' else 0
    isAcc = 1 if network.get_layer_by_type('c_accuracy')[0] != [] else 0

    train_dataset = JPEGreader(train_data, file_format, image_size, train_label, train_batch_size)
    if isTrainingTest == 1:
        test_dataset = JPEGreader(test_data, file_format, image_size, test_label, test_batch_size)

    images = network.get_layer_by_type('jpeg_input')[0].get_op_output('image')
    labels = network.get_layer_by_type('jpeg_input')[0].get_op_output('label')

    loss = network.get_layer_by_type('c_loss')[0].get_op_output('output')
    if isAcc == 1:
        acc = network.get_layer_by_type('c_accuracy')[0].get_op_output('output')

    # optimizer layer name must be 'optimizer'
    opt = network.get_layer('optimizer').get_op_output('output')
    global_step = network.get_layer('optimizer').get_op_output('global_step')

    merged = tf.summary.merge_all()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.log_device_placement = False  # Check
    config.allow_soft_placement = True

    # step, summary, checkpoint hook setting
    ckptdir = UDLI.getArgs("parameter_out")
    summarydir = UDLI.getArgs("log_out")
    saver = tf.train.Saver()
    hooks = tf.train.StopAtStepHook(last_step=iteration)
    ckpthooks = tf.train.CheckpointSaverHook(save_steps=iteration - 1, checkpoint_dir=ckptdir,
                                             saver=saver)
    summaryhooks = tf.train.SummarySaverHook(save_steps=test_interval, output_dir=summarydir, summary_op=merged)

    '''  synthetic input for VGG
                        images = tf.Variable(
                            tf.random_normal([train_batch_size, image_size[0], image_size[1], 3], dtype=tf.float32, stddev=1e-1))
                        labels = tf.Variable(tf.constant(0, dtype=tf.int32, shape=[train_batch_size, 1000]))
   '''

    # print format
    format_str = '[{0:>6} Steps] Train Loss:{1:>8.3f}({2:>8.3f} sec/batch)'
    format_str_acc = '[{0:>6} Steps] Train Loss:{1:>8.3f}, Train Acc:{2:>8.3%}({3:>8.3f} sec/batch)'
    format_str_test = '[{0:>6} Steps] Average Test Accuracy:{1:>8.3%}'

    with tf.train.MonitoredTrainingSession(master=creator, is_chief=(taskIdx == 0),
                                           hooks=[hooks, ckpthooks, summaryhooks], config=config) as sess:
        step = 0
        # while not sess.should_stop(): for twkim
        while step < iteration - 1:
            start_time = time.time()
            batched_images, batched_labels = train_dataset.next_batch(shuffle=True)
            _, step = sess.run([opt, global_step], feed_dict={images: batched_images, labels: batched_labels})
            if step % train_display == 0:
                if isAcc == 1:
                    # Calculate batch loss and accuracy
                    train_loss, train_acc = sess.run([loss, acc],
                                                     feed_dict={images: batched_images, labels: batched_labels})
                    print(format_str_acc.format(step, train_loss, train_acc, time.time() - start_time))
                else:
                    train_loss = sess.run(loss, feed_dict={images: batched_images, labels: batched_labels})
                    print(format_str.format(step, train_loss, time.time() - start_time))
            # validation procedure
            if isTrainingTest == 1 and step % test_interval == 0:
                # test
                print 'Testing......'
                # get test mnist batch
                test_acc_mean = 0.0
                for test_iter in xrange(1, test_iteration):
                    batched_images, batched_labels = test_dataset.next_batch(shuffle=True)
                    test_acc = sess.run(acc, feed_dict={images: batched_images, labels: batched_labels})
                    test_acc_mean += test_acc
                test_acc_mean = test_acc_mean / test_iteration
                print(format_str_test.format(step, test_acc_mean))
        print("[DLMDL] Optimization Finished!")
        print("[DLMDL] Saving model to %s." % ckptdir)


@staticmethod
def ptb_lstm_executor(UDLI, creator, taskIdx, learning_option):
    print 'ptb_lstm_executor'
    network = UDLI.network
    cluster = UDLI.cluster

    global_step = tf.train.get_or_create_global_step()  # global step create
    network.run(learning_option, cluster)

    # datasets setting
    train_bs = learning_option.get('batch_size')  # datasets batch size
    train_dpath = learning_option.get('data_path')  # datasets data path
    train_display = learning_option.get('train_display')  # interval to display
    total_iter = learning_option.get('iteration')  # total iteration to datasets

    # test setting : test iteration must be 1 epoch
    is_test = 1 if learning_option.get('option') == 'train_test' else 0  # only training or with testing
    if is_test == 1:
        test_bs = learning_option.get('test_batch_size')
        test_dpath = learning_option.get('test_data_path')
        test_interval = learning_option.get('test_interval')

    # get procedure placeholder which represents the procedure(training or testing)
    is_train = learning_option.get('is_train')

    # get lstm status
    initial_state = network.get_layer_by_type('r_multicell')[0].get_op_output('initial_state')
    final_state = network.get_layer_by_type('r_multicell')[0].get_op_output('final_state')
    num_steps = learning_option.get('num_steps')
    print "initial = ", initial_state  # print initial status

    # get loss value
    cost = network.get_layer_by_type('r_sequenceloss')[0].get_op_output('output')
    # get optimizer
    opt = network.get_layer('optimizer').get_op_output('output')

    # step, summary, checkpoint hook setting
    ckptdir = UDLI.getArgs("parameter_out")  # path to write intermediate parameters of model
    summarydir = UDLI.getArgs("log_out")  # path to write TF summary
    merged = tf.summary.merge_all()  # merge all summaries

    # get ptb input
    train_reader = PTBreader(train_dpath, train_bs, num_steps, istrain=1)
    # train_data, train_targets = train_reader.ptb_producer(name='train_ptb')
    train_epoch_size = train_reader.get_epoch_size()
    if is_test == 1:
        test_reader = PTBreader(test_dpath, test_bs, num_steps, istrain=0)
        # test_data, test_targets = train_reader.ptb_producer(name='test_ptb')
        test_epoch_size = test_reader.get_epoch_size()
    is_train = learning_option.get('is_train')

    # get input placeholder
    # data_placeholder = network.get_layer_by_type('ptb_input')[0].get_op_output('text')
    data_placeholder = learning_option.get('data_placeholder')
    targets_placeholder = network.get_layer_by_type('ptb_input')[0].get_op_output('targets')

    # step, summary, checkpoint hook setting
    saver = tf.train.Saver()  # TF saver class
    # stophooks=tf.datasets.StopAtStepHook(last_step=total_iter / train_epoch_size) # hook to stop the tranining epochs based on total iterations
    ckpthooks = tf.train.CheckpointSaverHook(save_steps=total_iter - 1, checkpoint_dir=ckptdir, saver=saver)
    summaryhooks = tf.train.SummarySaverHook(save_steps=test_interval, output_dir=summarydir, summary_op=merged)

    # TF session configure
    config = tf.ConfigProto()  # make TF session configure class
    config.gpu_options.allow_growth = True  # allocate GPU memory option
    config.log_device_placement = False  # whether loggging device placement to ops and variables

    # open TF monitoring session with configure and hooks
    with tf.train.MonitoredTrainingSession(master=creator, is_chief=(taskIdx == 0), hooks=[ckpthooks, summaryhooks],
                                           config=config) as sess:
        # current_epoch = 0
        # while sess.should_stop():
        for j in range(total_iter // train_epoch_size):
            start_time = time.time()
            idx = 0  # for test procedure
            iters = 0
            costs = 0.0
            train_data, train_targets = train_reader.next_batch(0)
            state = sess.run(initial_state,
                             feed_dict={data_placeholder: train_data, targets_placeholder: train_targets})
            # datasets procedure
            for step in range(train_epoch_size):
                feed_dict = {}
                train_data, train_targets = train_reader.next_batch(step)
                feed_dict[data_placeholder] = train_data
                feed_dict[targets_placeholder] = train_targets
                for i, (c, h) in enumerate(initial_state):
                    feed_dict[c] = state[i].c
                    feed_dict[h] = state[i].h

                feed_dict[is_train] = True
                eval_op_val, cost_val, final_state_val = sess.run([opt, cost, final_state], feed_dict)

                state = final_state_val

                # costs += cost_val
                iters += learning_option.get("num_steps")

                if step % train_display == 0:
                    format_str = '[{0:>6.3f} Epoch]  cost: {1:>9.3f}(speed: {2:9.4f} wps)'
                    print(format_str.format(j + step * 1.0 / train_epoch_size, cost_val,
                                            iters * train_bs / (time.time() - start_time)))

                    # format_str = '[{0:>6.3f} Epoch]  perplexity: {1:>9.3f}(speed: {2:9.4f} wps)'
                    # print(format_str.format(j + step*1.0/ train_epoch_size, np.exp(costs / iters), iters * train_bs / (time.time() - start_time)))

            # test: after end of each epoch
            if (train_epoch_size * j) // test_interval > idx:
                print 'Testing.....'
                test_iters = 0
                test_costs = 0.0
                test_data, test_targets = test_reader.next_batch(0)
                test_state = sess.run(initial_state,
                                      feed_dict={data_placeholder: test_data, targets_placeholder: test_targets})
                # datasets procedure
                for step in range(test_epoch_size):
                    feed_dict = {}
                    test_data, test_targets = test_reader.next_batch(step)
                    feed_dict[data_placeholder] = test_data
                    feed_dict[targets_placeholder] = test_targets
                    for i, (c, h) in enumerate(initial_state):
                        feed_dict[c] = test_state[i].c
                        feed_dict[h] = test_state[i].h

                    feed_dict[is_train] = False
                    cost_val, final_state_val = sess.run([cost, final_state], feed_dict)

                    test_state = final_state_val

                    test_costs += cost_val
                    test_iters += learning_option.get("num_steps")

                format_str = '[ Test Epoch ]  test perplexity: {0:>9.3f}'
                print(format_str.format(np.exp(test_costs / test_iters)))
                idx += 1
            # current_epoch += 1

        print("[DLMDL] Optimization Finished!")
        print("[DLMDL] Saving model to %s." % ckptdir)


@staticmethod
def galucoma_cnn_executor(UDLI, creator, taskIdx, learning_option):
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

    def get_dependencies(tensor):
        dependencies = set()
        dependencies.update(tensor.op.inputs)
        for sub_op in tensor.op.inputs:
            dependencies.update(get_dependencies(sub_op))
        return dependencies

    print 'galucoma_cnn_executor'
    network = UDLI.network
    cluster = UDLI.cluster

    network.run(learning_option, cluster)

    iteration = learning_option.get('iteration')

    train_batch_size = learning_option.get('batch_size')
    train_data = learning_option.get('data_path')
    train_display = learning_option.get('train_display')

    test_data = learning_option.get('test_data_path')
    test_batch_size = learning_option.get('test_batch_size')
    test_iteration = learning_option.get('test_iteration')
    test_interval = learning_option.get('test_interval')

    isTrainingTest = 1 if learning_option.get('option') == 'train_test' else 0
    isAcc = 1 if network.get_layer_by_type('c_accuracy')[0] != [] else 0

    # test version: only test dataset is available - twkim
    train_images = open(train_data + '/train_dataset.npy')
    train_labels = open(train_data + '/train_labels.npy')

    raw_train_images = np.load(train_images)
    raw_train_labels = np.load(train_labels)

    raw_train_labels_dense = np.empty((raw_train_labels.shape[0]), np.int32)
    for idx, col in enumerate(raw_train_labels):
        raw_train_labels_dense[idx] = 0 if col[0] == 1 else 1
    print(raw_train_labels_dense.dtype)

    if isTrainingTest == 1:
        # test version: only test dataset is available - twkim
        test_images = open(test_data + '/test_dataset.npy')
        test_labels = open(test_data + '/test_labels.npy')

        raw_test_images = np.load(test_images)
        raw_test_labels = np.load(test_labels)

        raw_test_labels_dense = np.empty((raw_test_labels.shape[0]), np.int32)
        for idx, col in enumerate(raw_test_labels):
            raw_test_labels_dense[idx] = 0 if col[0] == 1 else 1
        print(raw_test_labels_dense.dtype)

    images_raw = tf.placeholder(tf.float32, shape=(None, 240, 240, 3), name='raw_images')
    labels_raw = tf.placeholder(tf.int32, shape=(None), name='raw_labels')

    with tf.device('/cpu:0'):
        train_dataset, train_target = data_aug(images_raw, labels_raw)
        test_dataset, test_target = data_aug(images_raw, labels_raw)

    # print(type(raw_train_labels))
    images = network.get_layer_by_type('galucoma_input')[0].get_op_output('image')
    labels = network.get_layer_by_type('galucoma_input')[0].get_op_output('label')

    loss = network.get_layer_by_type('c_loss')[0].get_op_output('output')
    if isAcc == 1:
        acc = network.get_layer_by_type('c_accuracy')[0].get_op_output('output')

    # optimizer layer name must be 'optimizer'
    opt = network.get_layer('optimizer').get_op_output('output')
    global_step = network.get_layer('optimizer').get_op_output('global_step')

    merged = tf.summary.merge_all()
    # merged = tf.summary.merge([tf.summary.scalar('Loss', loss), tf.summary.scalar('Acc', acc)])

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.log_device_placement = False  # Check
    config.allow_soft_placement = True

    # step, summary, checkpoint hook setting
    ckptdir = UDLI.getArgs("parameter_out")
    summarydir = UDLI.getArgs("log_out")
    saver = tf.train.Saver()
    hooks = tf.train.StopAtStepHook(last_step=iteration)
    ckpthooks = tf.train.CheckpointSaverHook(save_steps=iteration - 1, checkpoint_dir=ckptdir,
                                             saver=saver)
    summaryhooks = tf.train.SummarySaverHook(save_steps=test_interval, output_dir=summarydir, summary_op=merged)

    # print format
    format_str = '[{0:>6} Steps] Train Loss:{1:>8.3f}({2:>8.3f} sec/batch)'
    format_str_acc = '[{0:>6} Steps] Train Loss:{1:>8.3f}, Train Acc:{2:>8.3%}({3:>8.3f} sec/batch)'
    format_str_test = '[{0:>6} Steps] Average Test Accuracy:{1:>8.3%}'
    print(tf.GraphKeys.SUMMARIES)

    with tf.train.MonitoredTrainingSession(master=creator, is_chief=(taskIdx == 0), hooks=[hooks, ckpthooks],
                                           config=config) as sess:
        step = 0
        summary_writer = tf.summary.FileWriter(summarydir)

        train_data_aug, train_labels_aug = sess.run([train_dataset, train_target],
                                                    feed_dict={images_raw: raw_train_images,
                                                               labels_raw: raw_train_labels_dense})
        train_datas = GALUCOMAReader(train_data_aug, train_labels_aug)
        # print(train_datas.labels)

        if isTrainingTest == 1:
            test_data_aug, test_labels_aug = sess.run([test_dataset, test_target],
                                                      feed_dict={images_raw: raw_test_images,
                                                                 labels_raw: raw_test_labels_dense})
            test_datas = GALUCOMAReader(test_data_aug, test_labels_aug)

        # while not sess.should_stop(): for twkim
        while step < iteration - 1:
            start_time = time.time()
            batched_images, batched_labels = train_datas.next_batch(train_batch_size)
            # convert label format -twkim
            # print(batched_labels)

            _, step = sess.run([opt, global_step], feed_dict={images: batched_images, labels: batched_labels})
            if step % train_display == 0:
                if isAcc == 1:
                    # Calculate batch loss and accuracy
                    train_loss, train_acc = sess.run([loss, acc],
                                                     feed_dict={images: batched_images, labels: batched_labels})
                    print(format_str_acc.format(step, train_loss, train_acc, time.time() - start_time))
                else:
                    train_loss = sess.run(loss, feed_dict={images: batched_images, labels: batched_labels})
                    print(format_str.format(step, train_loss, time.time() - start_time))
            # validation procedure
            if isTrainingTest == 1 and step % test_interval == 0:
                # test
                print 'Testing......'
                # get test mnist batch
                test_acc_mean = 0.0
                for test_iter in xrange(1, test_iteration):
                    batched_images, batched_labels = test_datas.next_batch(test_batch_size)
                    summ, test_acc = sess.run([merged, acc], feed_dict={images: batched_images, labels: batched_labels})
                    test_acc_mean += test_acc
                test_acc_mean = test_acc_mean / test_iteration
                print(format_str_test.format(step, test_acc_mean))
                summary_writer.add_summary(summ, step)
        print("[DLMDL] Optimization Finished!")
        print("[DLMDL] Saving model to %s." % ckptdir)


@staticmethod
def grpc_server(server):
    print 'grpc_server'
    server.join()


@staticmethod
def t(obj):
    return json.dumps(obj).replace('true', 'True').replace('false', 'False')