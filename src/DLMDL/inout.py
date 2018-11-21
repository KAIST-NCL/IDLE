# -*- coding: utf-8 -*-

import abc
from collections import OrderedDict


class InOut(object):

    def __init__(self, layer):
        self.layer = layer
        self.arg_list = {}

    def get_layer(self):
        return self.layer

    @abc.abstractmethod
    def get_obj(self, arg_name):
        pass

    @abc.abstractmethod
    def set_obj(self, arg_name, obj):
        pass

    @abc.abstractmethod
    def get_dimension(self, arg_name):
        pass

    @abc.abstractmethod
    def set_dimension(self, arg_name, dimension):
        pass

    @staticmethod
    def get_split_name(full_name):  # full_name을 . 을 기준으로 단어를 구분해 배열로 반환하는 함수
        return full_name.split('.')

    def get_full_name(self, arg_name):  # 해당 layer.name과 arg_name을 아래의 format형태의 문자열로 만들어 반환하는 함수
        return self.layer.get_full_name(arg_name)

    def get_connection(self, arg_name):
        return self.arg_list[arg_name]

    @staticmethod
    def list_intersection(list1, list2):  # list1의 items와 list2의 items를 비교하여 같은 값이 있을경우, list로 묶어 반환하는 함수
        return list(set(list1).intersection(set(list2)))


class In(InOut):

    def __init__(self, layer, arg_list):
        super(In, self).__init__(layer)  # Inout 클래스의 init 함수 호출
        for arg_name, conn in arg_list.items():  # arg_list의 key(arg_name)와 그 value(conn)를 쌍으로 묶어 배열로 반환하며, 그 배열의 items 개수만큼 loop
            self.arg_list[arg_name] = OrderedDict()  # array type인 self.arg_list[arg_name]의 값들을 순서를 가진 dictionary type으로 적용
            for conn_name in conn:
                self.arg_list[arg_name][conn_name] = self.create_connection(arg_name, conn_name)

    def create_connection(self, arg_name, conn_name):
        """
        인자를 받아 아래 형태의 connection dictionary를 생성해 반환하는 함수
        :param arg_name: input name
        :param conn_name: input으로 들어온 prev_output name
        :return: name, target, link를 key로 하고 그에 따른 value를 저장한 dictionary (input 객체를 처음 생성한 상태이므로 key 'link'의 value는 None으로 설정)
        """
        return {'name': self.get_full_name(arg_name), 'target': conn_name, 'link': None}

    def add_link(self, in_name, prev_output):
        """
        input name의 list들과 이전 layer의 output list들을 비교해 같은 이름에 대해서 상호간 연결을 지원하는 함수
        link의 value에 서로의 주소값을 넣어 연결을 지원
        :param in_name: input name
        :param prev_output: 이전 layer의 output 객체
        """
        conn_names = self.list_intersection(self.arg_list[in_name].keys(), prev_output.arg_list.keys())
        for conn_name in conn_names:
            if self.get_link(arg_name=in_name, conn_name=conn_name) is None:  # link
                self.set_link(arg_name=in_name, conn_name=conn_name, prev_output=prev_output)
                prev_output.set_link(conn_name, self)  # 이전 layer
                self.layer.add_to_visit_count(1)

    def get_connection(self, arg_name, conn_name=None, attr=None):
        """
        input name의 connection dictionary를 반환하는 함수
        conn_name이 없을 경우에는 관련 리스트를 반환할 수 있음
        attr이 명시되면 해당 attr에 해당하는 value를 반환
        :param arg_name: input name
        :param conn_name: input으로 들어온 prev_output name
        :param attr: input으로 들어온 prev_output name의 key
        """
        arg = super(In, self).get_connection(arg_name=arg_name)  # InOut의 get_connection 함수 호출
        if conn_name is not None:
            return arg[conn_name] if attr is None else arg[conn_name][attr]  # arg[conn_name]: input으로 들어온 prev_output name의 value / arg[conn_name][attr]: input으로 들어온 prev_output name의 key 'attr'의 value
        else:
            return [(conn if attr is None else conn[attr]) for conn in arg.values()]  # attr is None: input으로 들어온 prev_output name들의 value들을 배열 / else: input으로 들어온 prev_output name들의 key 'attr'의 value들을 배열

    def get_obj(self, arg_name):
        conns = super(In, self).get_connection(arg_name=arg_name)
        obj_list = [conn['link'].get_obj(conn['target']) for conn in conns.values()]
        return obj_list[0] if len(obj_list) == 1 else obj_list

    def set_obj(self, arg_name, obj):
        raise NotImplementedError

    def get_dimension(self, arg_name):
        conns = super(In, self).get_connection(arg_name=arg_name)
        obj_list = [conn['link'].get_dimension(conn['target']) for conn in conns.values()]
        return obj_list[0] if len(obj_list) == 1 else obj_list

    def set_dimension(self, arg_name, dimension):
        raise NotImplementedError

    def get_link(self, arg_name, conn_name=None):
        """
        input으로 들어온 prev_output name으로 connection을 찾아 key 'link'에 접근해서 value를 반환하는 함수
        :param arg_name: input name
        :param conn_name: input으로 들어온 prev_output name
        :return: input으로 들어온 prev_output name의 connection의 key 'link'의 value
        """
        return self.get_connection(arg_name, conn_name, 'link')

    def set_link(self, arg_name, conn_name, prev_output):
        """
        connection을 찾아 key 'link'에 이전 layer의 output 객체의 주소를 저장하는 함수
        :param arg_name: input name
        :param conn_name: input으로 들어온 prev_output name
        :param prev_output: 이전 layer의 output 객체
        """
        self.get_connection(arg_name, conn_name)['link'] = prev_output

    def get_input_list(self):
        return self.arg_list.keys()

    def get_input_tuple(self):
        return self.arg_list.items()

    def __repr__(self):
        return '({name}) Layer input : '.format(name=self.layer.name).join([self.get_full_name(arg_name) for arg_name in self.arg_list.keys()])


class Out(InOut):

    def __init__(self, layer, arg_list):
        super(Out, self).__init__(layer)
        for arg_name in arg_list:
            self.arg_list[self.get_full_name(arg_name)] = self.create_connection(arg_name)

    def create_connection(self, arg_name):
        """
        인자를 받아 아래 형태의 connection dictionary를 생성해 반환하는 함수
        :param arg_name: output name
        :return: name, obj, dimension, link를 key로 하고 그에 따른 value를 저장한 dictionary (output 객체를 처음 생성한 상태이고 여러 layer로 output될 수 있으므로 key 'link'의 value는 빈 배열로 설정)
        """
        return {'name': self.get_full_name(arg_name), 'obj': None, 'dimension': None, 'link': []}

    def get_connection(self, arg_name, attr=None):
        """
        output name의 connection dictionary를 반환하는 함수
        attr이 명시되면 해당 attr에 해당하는 value를 반환
        :param arg_name: output name
        :param attr: output name의 key
        """
        arg = super(Out, self).get_connection(arg_name=arg_name)  # InOut의 get_connection 함수 호출
        return arg if attr is None else arg[attr]

    def get_obj(self, arg_name):
        return self.get_connection(arg_name=arg_name, attr='obj')

    def set_obj(self, arg_name, obj):
        self.get_connection(arg_name=self.get_full_name(arg_name))['obj'] = obj

    def get_dimension(self, arg_name):
        return self.get_connection(arg_name=arg_name, attr='dimension')

    def set_dimension(self, arg_name, dimension):
        self.get_connection(arg_name=self.get_full_name(arg_name))['dimension'] = dimension

    def get_link(self, arg_name):
        """
        output name으로 connection을 찾아 key 'link'에 접근해서 value를 반환하는 함수
        :param arg_name: output name
        :return: output name의 connection의 key 'link'의 value
        """
        return self.get_connection(arg_name=arg_name, attr='link')

    def set_link(self, arg_name, next_input):
        """
        connection을 찾아 key 'link'의 value에 다음 layer 객체의 주소를 추가하는 함수
        :param arg_name: output name
        :param next_input: 다음 layer의 객체
        """
        self.get_connection(arg_name=arg_name)['link'].append(next_input)

    def get_full_args(self):
        return self.arg_list.keys()

    def __repr__(self):
        return 'Layer {name} output'.format(name=self.layer.name)
