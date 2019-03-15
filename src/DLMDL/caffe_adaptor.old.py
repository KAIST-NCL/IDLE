# -*- coding: utf-8 -*-
######################################
## TODO : 현재 Train 이나 Train/Test 만 가능 => Test만 하는 것도 추가해야 함 (불러올 snapshot 지정 등 생각 필요)

import itertools
import sys
import caffe
import os
import re

caffe_solver = {} # TODO: DLMDL에 solver 따로 만들 경우 재검토 필요
tempNet = caffe.NetSpec() # TODO: DLMDL에 solver 따로 만들거나 변경시 재검토 필요

class caffe_adaptor:

    @staticmethod
    def compile(UDLI):
        network = UDLI.network
        learning_option = UDLI.learning_option
        cluster = UDLI.cluster
        network.compile(learning_option, cluster)
        n = caffe.NetSpec()
        isTrainTest = 0
        isTest = 0
        print(network.get_all_outputs())

        # get test info if option = train_test
        for key in learning_option:
            if (key == 'option') and (learning_option[key] == 'train_test'):
                isTrainTest = 1
                print(tempNet.to_proto())
            elif (key == 'option') and (learning_option[key] == 'test'):
                isTest = 1

                
        # delete unneccessary learning options
        if isTrainTest == 1:
            try:
                del learning_option['option']
            except KeyError:
                pass
        elif isTest == 0:
            try:
                del learning_option['test_iter']
                del learning_option['test_interval']
            except KeyError:
                pass

        # make network obj for caffe
        for outputs in  list(itertools.chain.from_iterable(network.get_all_outputs())):
            name, obj = outputs
            if obj is not None:
                setattr(n, name, obj)
        print(n.to_proto())

        # network 이름 <= --compile_out 에서 지정한 경로의 prefix로 적용
        net_name = 'name: "' + str(UDLI.getArgs('compile_out').split('/')[-1]) + '"\n'

        # network 에 대한 prototxt 파일 쓰기
        with open("%s_net.prototxt" % UDLI.getArgs('compile_out'), 'w') as f:
            f.write(net_name + str(tempNet.to_proto()) + str(n.to_proto()))

        # caffe만 사용하는 정보 default 로 설정
        caffe_solver['net'] = "%s_net.prototxt" % UDLI.getArgs('compile_out')
        caffe_solver['lr_policy'] = "fixed" # TODO : 지정 옵션에 맞게 설정 필요
        caffe_solver['display'] = learning_option['train_display']
        del learning_option['train_display']
        caffe_solver['solver_mode'] = cluster.get('types')[0].replace(re.sub('[^0-9]', '',cluster.get('types')[0]),'').upper() # TODO: 분산 실행 시 재검토 필요, 현재는 무조건 node가 1개일때만 고려
        
        # dlmdl에 표기한 값 적용
        for key in learning_option:
            caffe_solver[key] = learning_option[key]
            print key

        # snapshot 설정
        ## TODO : dlmdl에 snapshot 몇 iteration 마다 찍을지 지정하는 옵션도 넣을지 고려 가능
        ##        현재 마지막 iteration 결과만 snapshot 찍게 만듬
        caffe_solver['snapshot'] = learning_option['max_iter']
        caffe_solver['snapshot_prefix'] = "%s" % UDLI.getArgs('parameter_out')

        # solver 정보 출력
        print caffe_solver
        solver_str = ""

        # solver 정보 읽어 prototxt 파일로 쓰기
        for param in caffe_solver:
            if param in ["net", "lr_policy", "type", "snapshot_prefix"]:
                solver_str += str(param)+": "+'"'+str(caffe_solver[param])+'"\n'
            else:
                solver_str += str(param)+": "+str(caffe_solver[param])+'\n'

        with open("%s_sol.prototxt" % UDLI.getArgs("compile_out"), 'w') as f:
            f.write(solver_str)

        
    @staticmethod
    def run(UDLI):
        network = UDLI.network
        learning_option = UDLI.learning_option
        cluster = UDLI.cluster
        network.run(learning_option, cluster)
        # TODO: caffe framework 실행
        num = int(re.sub('[^0-9]', '', cluster.get('types')[0]))
        type = cluster.get('types')[0].replace(str(num), '')

        caffe.set_device(num)  # TODO: 분산 실행 시 재검토 필요 (입력 인자에 따라 설정하도록)
        caffe.set_mode_gpu() if type == 'gpu' else caffe.set_mode_cpu() # TODO: 분산 실행 시 재검토 필요

        solver = caffe.get_solver("%s_sol.prototxt" % UDLI.getArgs("compile_out"))
        solver.solve()

        os.environ['GLOG_minloglevel'] = '3'
        pass
