import itertools
import sys
import caffe
import os
import re

caffe_solver = {}
tempNet = caffe.NetSpec()

class caffe_adaptor:

    @staticmethod
    def compile(UDLI):
        network = UDLI.network
        learning_option = UDLI.learning_option
        cluster = UDLI.cluster
        network.compile(learning_option, cluster)
        n = caffe.NetSpec()

        # if option is TRAIN(training and testing model)
        if learning_option.get('option') == 'TRAIN':
            # make network obj for caffe
            for outputs in list(itertools.chain.from_iterable(network.get_all_outputs())):
                name, obj = outputs
                if obj is not None:
                    setattr(n, name, obj)

            # set net_name to prefix of --compile_out
            net_name = 'name: "' + str(UDLI.getArgs('compile_out').split('/')[-1]) + '"\n'

            # write net.prototxt
            with open("%s_net.prototxt" % UDLI.getArgs('compile_out'), 'w') as f:
                f.write(net_name + str(tempNet.to_proto()) + str(n.to_proto()))

            # solver setting
            caffe_solver['net'] = "%s_net.prototxt" % UDLI.getArgs('compile_out')
            caffe_solver['display'] = learning_option['train_display']
            caffe_solver['solver_mode'] = cluster.get('types')[0].replace(re.sub('[^0-9]', '', cluster.get('types')[0]),'').upper() # only consider one node training
            # cluster.get('types')[device].replace(str(num), '')

            # train/test iteration setting
            caffe_solver['max_iter'] = learning_option.get('iteration')
            caffe_solver['test_iter'] = learning_option.get('test_iteration')
            caffe_solver['test_interval'] = learning_option.get('test_interval')

            # snapshot(checkpoint) setting
            ckpt_path = learning_option.get('checkpoint_path')

            # create checkpoint directory
            ckpt_dir = ckpt_path + '/'
            if not os.path.exists(ckpt_dir):
                os.makedirs(ckpt_dir)

            ckpt_interval = learning_option.get('checkpoint_interval')
            if ckpt_path is not None and ckpt_interval is not None:
                caffe_solver['snapshot'] = ckpt_interval
                caffe_solver['snapshot_prefix'] = ckpt_path

            # optimizer setting
            for key, val in learning_option.get('opt_dic').items():
                caffe_solver[key] = val

            # learning rate scheduler setting
            for key, val in learning_option.get('lr_sched_dic').items():
                caffe_solver[key] = val

            # regularization setting
            reg_type = learning_option.get('caffe_reg_type')
            if reg_type is not None:
                caffe_solver['regularization_type'] = reg_type
                # weight decay params are described in each layer
                caffe_solver['weight_decay'] = 1.0 # WARNING: check

            print caffe_solver
            solver_str = ""

            for param in caffe_solver:
                if param in ["net", "lr_policy", "type", "snapshot_prefix", "regularization_type"]:
                    solver_str += str(param) + ": " + '"' + str(caffe_solver[param]) + '"\n'
                else:
                    solver_str += str(param) + ": " + str(caffe_solver[param]) + '\n'

            with open("%s_sol.prototxt" % UDLI.getArgs("compile_out"), 'w') as f:
                f.write(solver_str)

        elif learning_option.get('option') == 'RETRAIN':
            pass

        
    @staticmethod
    def run(UDLI):
        network = UDLI.network
        learning_option = UDLI.learning_option
        cluster = UDLI.cluster
        network.run(learning_option, cluster)

        num = int(re.sub('[^0-9]', '', cluster.get('types')[0]))
        type = cluster.get('types')[0].replace(str(num), '')

        caffe.set_device(num)  # WARNING: only consider single node execution
        caffe.set_mode_gpu() if type == 'gpu' else caffe.set_mode_cpu() # WARNING: only consider single node execution

        solver = caffe.get_solver("%s_sol.prototxt" % UDLI.getArgs("compile_out"))

        solver.solve()

        os.environ['GLOG_minloglevel'] = '3'

