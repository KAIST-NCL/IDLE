# Integrated Deep Learning Engine (IDLE)

IDLE provides nice cross-compiler for * [TensorFlow](https://www.tensorflow.org/) and * [Caffe](http://caffe.berkeleyvision.org/) with simple format, called UDLI (Unified Deep Learning Interface).  
Overall architecture of IDLE is introduced in * [Taewoo Kim, Eunju Yang, Soyoon Bae and Chan-Hyun Youn, “IDLE: Integrated Deep Learning Engine with Adaptive Task Scheduling on Heterogeneous GPUs”, to appear in IEEE TENCON, 2018]()



## Getting Started

#### Dependencies
- Ubuntu 16.04 (>=)
- python 2.7
- TensorFlow-gpu >= 1.90
- Caffe 1.0
- cuDNN-6.0
- CUDA 9.0

#### Install prerequisite : caffe, pyCaffe, TensorFlow

#####  Caffe

Python 2.7
```bash
$ sudo apt-get install python-pip python-dev 
```

Liberaries
```bash
$ sudo apt-get update
$ sudo apt-get install libprotobuf-dev libleveldb-dev libsnappy-dev libopencv-dev libhdf5-serial-dev protobuf-compiler
$ sudo apt-get install --no-install-recommends libboost-all-dev
```

Install CUDA : * [CUDA 9.0](https://developer.nvidia.com/cuda-downloads)

BLAS
```bash
$ sudo apt-get install libatlas-base-dev
```

Other dependencies
```bash
$ sudo apt-get install libgflags-dev libgoogle-glog-dev liblmdb-dev
```

Install Caffe
```bash
// Download source code
$ cd {workspace}
$ git clone https://github.com/BVLC/caffe.git
$ cd caffe
$ cp Makefile.config.example Makefile.config
```

```bash
// Change Make configuration --> check python include, lib path
$ vi Makefile.config
# NOTE: this is required only if you will compile the python interface. # We need to be able to find Python.h and numpy/arrayobject.h. PYTHON_INCLUDE := /usr/include/python2.7 \
/usr/local/lib/python2.7/dist-packages/numpy/core/include/

# We need to be able to find libpythonX.X.so or .dylib. PYTHON_LIB := /usr/lib
```

```bash
// Compile
$ make clean
$ make all
$ make test
$ make runtest
```

Install * [PyCaffe]()
- Dependencies
```bash
$ cd {workspace}/caffe/python
$ for req in $(cat requirements.txt); do sudo pip install $req; done
```

- Setting environment variables
```bash
$ echo export PYTHONPATH={caffe_path}/python:$PYTHONPATH >> ~/.bashrc
$ echo export PYTHONPATH={python_path}:$PYTHONPATH >> ~/.bashrc
$ echo export CAFFE_ROOT={caffe_path} >> ~/.bashrc
$ echo export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH >> ~/.bashrc $ source ~/.bashrc
```

- Recompiile
```bash
$ make clean
$ make pycaffe
```

- Test PyCaffe
```bash
$ python
>> import caffe
```

#####  TensorFlow
```bash
$ sudo apt-get install python-pip python-dev
$ pip install tensorflow-gpu
```

#####  Python dependency module

Parmiko

```bash
$ pip install parmiko
```

Pillow
```bash
$ pip install pillow
```


#####  Setting SSH Keys (for distributed training)

To support distributed training, all workers should share `ssh-key` each other.

* SSH key generation (local machine)
```bash
$ ssh-keygen -t -rsa
$ ls -al ~/.ssh
```

* send each ssh-key to all workers
```bash
$ scp ~/.ssh/id_rsa.pub (remote_machine_id@ip_address):($destination)
```

* register each worker key to authorized_keys
```bash
$ cat $destination/id_rsa.pub >> ~/.ssh/authorized_keys
```

#### Install IDLE

Install IDLE on your environment
```bash
$ cd {working directory}
$ git clone https://github.com/KAIST-NCL/IDLE.git
$ echo export PYTHONPATH={working directory}:$PYTHONPATH >> ~/.bashrc
$ source !$
```


#### How to Run

Sample DL-MDL files are wtored under /model folder.
all files with `dlmdl` extension shows the examples.

##### Edit sample DLMDL file
Edit the sample files for your environment

##### Running IDLE 

```bash
$ cd {working directory}
$ python src/UDLI.py --framework=tf --input=model/VGG -r
```

arguments are described in below

|	Argument	|	Explanation	|	Example		|
|-----------------------|-----------------------|-----------------------|
| ` --framework`	| Choose framework	| `--framework = tf` , `--framework=caffe`|
| ` --input`		| Path of input file (.dlmdl file)	| `--input=model/VGG.dlmdl`|
| ` -r`		| Execution or not (If -r is set, execution is conducted right after compilation	| `-r`|
| `--compile_out`	| compile output path	| `--compile_out=output/VGG`|
| `--log_out`		| path to save TensorFlow log | `--log_out=/tmp/IDLE_LOG/TF`|

#### License
This project is released under the  BSD 2-Clause license, see LICENSE for more information.
