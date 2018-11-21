# DLMDL version 1.5

## 1. 실행 환경 Setting
### Python environment 설정
<pre><code>PYTHONPATH=/path/to/caffe/python:/path/to/dlmdl/root</pre></code>
※ --input과 --compile_out argument에는 확장자(.dlmdl, .py, .prototxt)를 붙이면 안됨.
### Machine description 설정
<pre><code>{
name:"{worker name}",
ip:"{ip address:port}",
type: "{device type(cpu0/gpu0)}",
task: "{worker task(ps/worker)}"
}
</pre></code>

## 2. DLMDLtoTF
### Compile output 생성 
<pre><code>#move to dlmdl root directory
python src/UDLI.py --framework=tf --input=/path/to/input/dlmdl --compile_out=/path/to/output/python
</pre></code>
### Direct execution
<pre><code>#move to dlmdl root directory
python src/UDLI.py --framework=tf --input=/path/to/input/dlmdl -r
</pre></code>

## 3. DLMDLtoCaffe
### Compile output 생성 
<pre><code>#move to dlmdl root directory
python src/UDLI.py --framework=caffe --input=/path/to/input/dlmdl --compile_out=/path/to/output/prototxt
</pre></code>
### Direct execution
<pre><code>#move to dlmdl root directory
python src/UDLI.py --framework=caffe --input=/path/to/input/dlmdl -r
</pre></code>

## Example1. VGG model training with JPEG dataset
(memory가 상당히 많이 필요하기 때문에 batch size를 최소한으로 줄임	)
flower dataset: available in http://www.robots.ox.ac.uk/~vgg/data/flowers/
### Caffe 실행 
<pre><code>#move to dlmdl root directory
python src/UDLI.py --framework=caffe --input=data/VGG --compile_out=/path/to/output/prototxt --parameter_out=/path/to/result/param -r
</pre></code>

### TensorFlow 실행 
<pre><code>#move to dlmdl root directory
python src/UDLI.py --framework=tf --input=data/VGG --compile_out=/path/to/output/py --log_out=/path/to/out/summary --parameter_out=/path/to/result/param -r
</pre></code>

## Example2. RNN model training with MNIST dataset
### TensorFlow 실행 
<pre><code>#move to dlmdl root directory
python src/UDLI.py --framework=tf --input=data/MNIST_RNN --compile_out=/path/to/output/py --log_out=/path/to/out/summary --parameter_out=/path/to/result/param -r
</pre></code>

## Example3. Multi LSTM model training with ptb dataset
### TensorFlow 실행 
<pre><code>#move to dlmdl root directory
python src/UDLI.py --framework=tf --input=data/LSTM --compile_out=/path/to/output/py --log_out=/path/to/out/summary --parameter_out=/path/to/result/param -r
</pre></code>