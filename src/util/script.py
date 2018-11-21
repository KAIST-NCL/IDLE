import json
from collections import OrderedDict


def t(obj, attr):
    return json.dumps(obj[attr]).replace('true', 'True').replace('false', 'False')


with open('data/VGG.dlmdl') as f:
    obj_def = json.load(f, object_pairs_hook=OrderedDict)

    for l in obj_def['layers']:
        template = """self.Layer(type={type}, name={name}, inputs={inputs}, outputs={outputs},
                   attributes={attributes}, device={device})"""
        r = template.format(type=t(l, 'type'), name=t(l, 'name'), inputs=t(l, 'inputs'), outputs=t(l, 'outputs'),
                            attributes=t(l, "attributes"), device=t(l, 'device'))
        print r