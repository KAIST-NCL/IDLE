# -*- coding: utf-8 -*-
from ..layer_operation import LayerOperation
import tensorflow as tf
from PIL import Image
import os
import re
import sys
import collections

class op_tf_input(LayerOperation):

    _attributes = """[{"default": 20, "source": "layer", "mandatory": "tf", "name": "num_steps"}, \
    {"default": 10000, "source": "layer", "mandatory": "tf", "name": "vocab_size"}, \
    {"default": 200, "source": "layer", "mandatory": "tf", "name": "hidden_size"}]"""

    def compile_time_operation(self, learning_option, cluster):
        pass

    def run_time_operation(self, learning_option, cluster):
        def _read_words(filename):
            with tf.gfile.GFile(filename, "r") as f:
                if sys.version_info[0] >= 3:
                    return f.read().replace("\n", "<eos>").split()
                else:
                    return f.read().decode("utf-8").replace("\n", "<eos>").split()

        def _build_vocab(filename):
            data = _read_words(filename)

            counter = collections.Counter(data)
            count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

            words, _ = list(zip(*count_pairs))
            word_to_id = dict(zip(words, range(len(words))))

            return word_to_id

        def _file_to_word_ids(filename, word_to_id):
            data = _read_words(filename)
            return [word_to_id[word] for word in data if word in word_to_id]

        def ptb_raw_data(data_path=None):

            train_path = os.path.join(data_path, "ptb.datasets.txt")
            word_to_id = _build_vocab(train_path)
            train_data = _file_to_word_ids(train_path, word_to_id)

            vocabulary = len(word_to_id)
            return train_data, vocabulary

        def ptb_producer(raw_data, batch_size, num_steps, name=None):
            with tf.name_scope(name, "PTBProducer", [raw_data, batch_size, num_steps]):
                raw_data = tf.convert_to_tensor(raw_data, name="raw_data", dtype=tf.int32)

                data_len = tf.size(raw_data)
                batch_len = data_len // batch_size
                data = tf.reshape(raw_data[0: batch_size * batch_len],
                                  [batch_size, batch_len])

                epoch_size = (batch_len - 1) // num_steps
                assertion = tf.assert_positive(
                    epoch_size,
                    message="epoch_size == 0, decrease batch_size or num_steps")
                with tf.control_dependencies([assertion]):
                    epoch_size = tf.identity(epoch_size, name="epoch_size")

                i = tf.train.range_input_producer(epoch_size, shuffle=False).dequeue()
                x = tf.strided_slice(data, [0, i * num_steps],
                                     [batch_size, (i + 1) * num_steps])
                x.set_shape([batch_size, num_steps])
                y = tf.strided_slice(data, [0, i * num_steps + 1],
                                     [batch_size, (i + 1) * num_steps + 1])
                y.set_shape([batch_size, num_steps])
                return x, y

        def is_grey_scale(imgPath, file_format):
            ImgFile = ''
            for root, dirs, files in os.walk(imgPath):
                for file in files:
                    if os.path.splitext(file)[1].lower() == '.' + file_format:
                        ImgFile = os.path.join(root, file)
                        break

            img = Image.open(ImgFile).convert('RGB')
            w, h = img.size
            for i in range(w):
                for j in range(h):
                    r, g, b = img.getpixel((i, j))
                    if r != g != b: return False
            return True
        def apiConstructor(image_size, is_grey):
            images_placeholder = tf.placeholder(tf.float32, shape=(None, image_size[0], image_size[1], 1 if is_grey == True else 3))
            labels_placeholder = tf.placeholder(tf.int64, shape=(None))
            return images_placeholder, labels_placeholder

        if model_type == 'cnn':
            """
            input layer operation returns random batch from input image
            :return: [image, label]
           """
            file_format = learning_option.get("file_format")
            data_path = learning_option.get("data_path")
            image_size = self.get_attr('image_size')
            is_grey = is_grey_scale(data_path, file_format)

            device = self.get_attr('device')
            num = re.sub('[^0-9]','',cluster.get('types')[device])
            type = cluster.get('types')[device].replace(str(num),'')

            if learning_option.get("parallel", None) != "DP":
                with tf.device('/job:worker/task:{0}/{1}:{2}'.format(device, type, num)):
                        images_placeholder, labels_placeholder = apiConstructor(image_size, is_grey)
            else:
                images_placeholder, labels_placeholder = apiConstructor(image_size, is_grey)

            outdim = list(images_placeholder.get_shape()[i].value for i in xrange(len(images_placeholder.get_shape())))
            self.set_output('image', images_placeholder)
            self.set_output('label', labels_placeholder)
            self.set_dimension('image', outdim)

        #TODO: api constructor wrapping, device setting
        else:   #for rnn, lstm
            raw_data = ptb_raw_data(learning_option.get("data_path")) #TODO: only training
            data, _ = raw_data

            batch_size = learning_option.get('batch_size')
            num_steps = self.get_attr('num_steps')
            epoch_size = ((len(data) // batch_size) - 1) // num_steps
            input_data, targets = ptb_producer(data, batch_size, num_steps)
            vocab_size = self.get_attr('vocab_size')
            hidden_size = self.get_attr('hidden_size')

            learning_option['epoch_size'] = epoch_size
            learning_option['num_steps'] = num_steps
            learning_option['vocab_size'] = vocab_size
            learning_option['hidden_size'] = hidden_size

            with tf.device("/cpu:0"):
                embedding = tf.get_variable(
                        "embedding", [vocab_size, hidden_size],
                        dtype=tf.float32)
                input_data = tf.nn.embedding_lookup(embedding, input_data)
            self.set_output('text', input_data)
            self.set_output('targets', targets)
