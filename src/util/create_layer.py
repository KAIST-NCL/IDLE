import re
import codecs
import py_compile
import json
import codegenUtil

def parseDLLDL(lines):
    result = {}
    key = None
    for line in lines:
        if line[0] == '@':
            str = line.rstrip().split(':')

            key = str[0].strip()[1:]

            if str[1] != '':
                result[key] = str[1].strip()
                key = None
            else:
                result[str[0].strip()[1:]] = ''
        else:
            if key is not None and line.strip() != '':
                indented = codegenUtil.indent(line)
                result[key] += indented if not key.endswith('import') else indented.lstrip()

    result['attributes'] = json.loads(result['attributes'])
    return result


if __name__ == '__main__':
    template_path = 'src/DLMDL/LayerOperation/template/specific_layer.template'
    with codecs.open(template_path, 'r', encoding='utf-8') as ft:
        template = ft.read()

    for layer_name in ['accuracy', 'adagradoptimizer', 'adamoptimizer', 'avg_pool', 'conv', 'fc', 'input', 'loss',
                       'max_pool']:
        for framework in ['caffe', 'tf']:
            name = 'src/DLMDL/LayerOperation/{framework}/{name}.py'.format(framework=framework, name=layer_name)
            with codecs.open(name, 'w', encoding='utf-8') as fo:
                try:
                    with codecs.open('src/DLMDL/LayerOperation/template/{name}.dlldl'.format(name=layer_name), 'r',
                                     encoding='utf-8') as fl:
                        parsed = parseDLLDL(fl.readlines())
                        print(parsed)
                        variable_template = ['{var} = self.{var}', '{var} = learning_option.get("{var}", self.{var})']
                        variable_def = ''
                        for attr in parsed.get('attributes'):
                            variable_def += codegenUtil.indentNL(variable_template[attr.get('source') != 'layer'].format(var=attr.get('name')))

                        str = template.format(framework=framework, name=parsed.get('name'),
                                              attributes=json.dumps(parsed.get('attributes')),
                                              variable_def =variable_def,
                                              import_lib=parsed.get('{framework}_import'
                                                                    .format(framework=framework), ''),
                                              compile_time=parsed.get('{framework}_compile_time'
                                                                      .format(framework=framework), codegenUtil.indentNL('pass')),
                                              run_time=parsed.get('{framework}_run_time'
                                                                  .format(framework=framework), codegenUtil.indentNL('pass')))
                        fo.write(str)
                        print(str)
                except IOError:
                    print('fail')
                    str = template.format(framework=framework, name=layer_name, import_lib='',
                                          compile_time=codegenUtil.indentNL('pass'),
                                          run_time=codegenUtil.indentNL('pass'))
                    fo.write(str)
                    print(str)
            py_compile.compile(name, name+'c')