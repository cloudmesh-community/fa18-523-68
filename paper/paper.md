# Distributed TensorFlow

| Selahattin Akkas | [sakkas@iu.edu](mailto:sakkas@iu.edu) | Indiana University 
| hid: fa18-523-68 | github: [☁️](https://github.com/cloudmesh-community/fa18-523-68/blob/master/paper/paper.md)

## Keywords

HID fa18-523-68, Distributed TensorFlow, Distribution Strategies, 
Distributed Deep Learning, Distributed Keras



## Introduction

TensorFlow [@fa18-523-68-www-tensorflow] [@fa18-523-68-abadi2016tensorflow] is 
an open-source machine learning library that is developed by Google. It was 
first released in 2015. It is the most starred (more than 144,000) machine 
learning library on GitHub [@fa18-523-68-www-github-tensorflow]. It can be used 
both in research and production, and it can be easily deployed on mobile devices
 [@fa18-523-68-www-tensorflow-lite].

Recently, deep learning models show good performance, and they have become 
really important. Training some of the deep learning models on a single machine 
may take several days, even with very powerful CPUs and GPUs. There are also 
some special hardware like Tensor Processing Unit (TPU) [@fa18-523-68-www-tpu] 
to boost the training speed. Yet, the speedup is limited by the accelerator. 
The alternative approach is to make computations distributed to speed up the 
training. 

Distributed Tensorflow [@fa18-523-68-www-distributed-tf] enables us to train our 
model using multiple machines. There are several approaches we can use in the 
distributed TensorFlow. We explain parallelism types of deep learning loads, 
synchronization types, and distributed training strategies of TensorFlow for 
data parallelism and show how to use them in this paper. Then, we show how to 
run the MNIST [@fa18-523-68-lecun1998mnist] problem in Distributed TensorFlow.

## Parallelism Types of Deep Learning

Distributed training can be categorized as a model and data parallelism. In the 
model parallelism, the model is split into multiple devices since the model 
does not fit into a single device. The main limitation of model parallelism is 
that excessive communications and data transfer are needed between the devices 
even for a single iteration. In the data parallelism, the same model is 
deployed on each device, and each device consumes different parts of the data. 
Each device makes local computations, and the results are exchanged between the 
devices [@fa18-523-68-www-oreilly-distributed-tf]. This paper focuses on the 
data parallelism approach. Please refer to Mesh TensorFlow 
[@fa18-523-68-www-mesh-tf] if you are interested in model parallelism.

### Synchronous and Asynchronous Training

Distributed training can be categorized into two based on the synchronization 
mechanism.

####  Synchronous training

In the synchronous training, each worker makes computation using different 
mini-batches of data. After workers complete their computation, results are 
synchronized, and workers proceed next to the next step. This approach may 
speed up the computation; however, if there is a slow worker, all other workers 
need to wait for a slow worker to complete its task. Hence, synchronous 
training may slow down training in heterogeneous environments [@fa18-523-68-www-oreilly-distributed-tf].

#### Asynchronous training

In the asynchronous training, all workers make computations. Then they send 
their results to other workers or a central worker. Still, a worker does not 
wait for another worker to finish the computation. Generally, there is a 
central worker in the asynchronous training. A worker sends its data to a 
central worker, and parameters on the central worker are updated. Other workers 
read the updated values and proceed to the next iteration using another batch 
of data [@fa18-523-68-www-oreilly-distributed-tf]. In this approach, some 
workers may read old values, and this may slow down the convergence.

## Estimators and Keras API

Distributed TensorFlow can be used with Estimators [@fa18-523-68-www-estimators] 
[@fa18-523-68-cheng2017tensorflow] and tf.keras [@fa18-523-68-www-tf-keras]. 
Estimators are high-level libraries that simplify machine learning in 
TensorFlow. It encapsulates:

> "training, evaluation, prediction, and export for serving" [@fa18-523-68-www-estimators].

Users do not need to struggle for low-level details. There are many premade 
estimators for well-known machine learning algorithms. Users can also build 
their own model, just providing a model function. Estimators allow users to run 
their code locally or distributed. Building the computational graph is handled 
by estimators. When running a distributed training, a small number of codes 
need to be added to make training distributed.

Keras is also a high-level API that works with TensorFlow. It is easier to 
prototype deep learning models in Keras and has better-distributed training 
support.



## TensorFlow Distribution Strategies

Tensorflow has an experimental API to make distribute training easily on 
multiple devices/machines. Currently, TensorFlow has six different distribution 
strategies that support Keras and Estimator API. These strategies can be used 
via minimalist code changes; however, some features are experimental as can be 
seen in the table below:

>| Training API             | MirroredStrategy     | TPUStrategy          | MultiWorkerMirroredStrategy | CentralStorageStrategy   | ParameterServerStrategy    | OneDeviceStrategy |
>| :----------------------- | :------------------- | :------------------- | :-------------------------- | :----------------------- | :------------------------- | :---------------- |
>| **Keras API**            | Supported            | Experimental support | Experimental support        | Experimental support     | Supported planned post 2.0 | Supported         |
>| **Custom training loop** | Experimental support | Experimental support | Support planned post 2.0    | Support planned post 2.0 | No support yet             | Supported         |
>| **Estimator API**        | Limited Support      | Not supported        | Limited Support             | Limited Support          | Limited Support            | Limited Support   |
>[@fa18-523-68-www-distributed-tf]



### Parameter Server Strategy

Parameter server strategy can be used with multiple GPUs on a single machine or 
using multiple machines. In this approach, training is done asynchronously. 
When using multiple machines, some machines are used as parameter servers, and 
others are used as workers. While we use parameter servers to store our 
training variables, workers to make the computations. Variables are not 
replicated and stored on a parameter server [@fa18-523-68-www-distributed-tf].



### Mirrored Strategy

Mirrored Strategy can be used with a machine that has multiple GPUs. This 
strategy creates the same variables on each device, which are called as 
*MirroredVariable*. After each device completes its task, results are combined, 
and all parameters are synchronized using all-reduce [@fa18-523-68-www-distributed-tf]. 
This approach uses Nvidia NCCL all-reduce implementation as default.



### Central Storage Strategy

Variables are assigned to CPU or GPU when there is a single GPU. 
If there are multiple GPUs, variables are stored on the main memory and not 
mirrored on the GPUs in the Central Storage Strategy. Variables are aggregated 
and updated after each iteration [@fa18-523-68-www-distributed-tf].



### Multi Worker Mirrored Strategy

Multi Worker Mirrored Strategy is used when we have multiple workers. Each 
worker may have one or more GPUS. But it works well even if there is no GPU. 
It uses collective communication to synchronize parameters on each worker [@fa18-523-68-www-distributed-tf].
 Users can choose to use either ring all reduce or Nvidia NCCL all-reduce 
 implementation in the strategy.



### TPU Strategy

TPU is a special hardware designed for artificial intelligence tasks. It works 
with TensorFlow and can be used on Google Cloud. TPU strategy is the same as 
the Mirrored Strategy, but it is designed to run distributed training on TPUs [@fa18-523-68-www-distributed-tf].



### One Device Strategy

One Device strategy is designed to test the code before trying real distributed 
strategies. It is not exactly distributed and runs on a single device [@fa18-523-68-www-distributed-tf].



## Dataset API

When a distribution strategy is used, we need an input function that returns a 
*tf.data* object. Dataset API [@fa18-523-68-chien2018characterizing] [@fa18-523-68-www-tf-data] 
makes creating an effective input pipeline for distributed learning easy. 
Dataset API uses the *Extract* – *Transform* – *Load* approach. It first 
extracts data from the source. The source can be an HDFS, GCS, or a disk. It 
can read the text, CSV, and tfrecord files. Then, it makes some preprocess 
operations, and a subpart of the data is loaded to memory to use in the 
computation. TensorFlow also allows us to parallelize some of the data pipeline 
transformations. In order to use resources efficiently, the data input pipeline 
is created on the CPU device, and GPU devices are used for compute-intensive 
tasks.

Some of the most used *tf.data* transformations:

- *map*: It is used to apply some transformations to data. It requires a python 
function or a lambda expression. This can be seen at the tutorial.
- *shuffle*: It shuffles the data and takes the buffer size as parameter.
- *batch*: It creates batches.
- *repeat*: Repeats the dataset after reaching the last record.
- *cache*: Caches the data in memory or disk and saves some operations during 
the later epochs.
- *prefetch*: It prefetches the next to the device. While GPU makes computation,
 CPU prepares the next batch and makes the next batch ready.



##  Distributed TensorFlow Tutorial

The following tutorial is tested on TensorFlow 2.1. It shows how to run MNIST 
with *MultiWorkerMirroredStrategy*. The first part contains imports and 
arguments. We need to have a worker index for each worker, and it should be 
provided when running the code.

```python
import os
import argparse
import json
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
import numpy as np
import tensorflow_datasets as tfds

parser = argparse.ArgumentParser(description='Distributed MNIST Argument Parser')
parser.add_argument('--index', required=True, help='index of the worker')
args = parser.parse_args()
```

*TF_CONFIG* is a system variable, and it should be set for multi worker 
training. There is an example provided below for two-worker training. 
Note that the task index should be provided as a parameter. The *localhost* 
should be replaced with an ip address to run it on multiple nodes.

```pyt
os.environ["TF_CONFIG"] = json.dumps({
    "cluster": {"worker": ["localhost:5050", "localhost:5051"]},
   "task": {"type": "worker", "index": args.index}})
```

The strategy that we will use is defined below. Note that this should be 
handled before creating our deep learning model.

```python
strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
datasets, info = tfds.load(name='mnist', with_info=True,as_supervised=True)
```

The *scale* function is used to scale pixel values between 0 and 1. The 
original range is between 0 and 255.

```python
def scale(image, label):
	image = tf.cast(image, tf.float32)
	image /= 255
	return image, label
```

TensorFlow divides the batch between workers automatically. That is why we need 
to set a batch size larger compared to the single worker case.

```python
NUM_WORKERS = 2
GLOBAL_BATCH_SIZE = 50 * NUM_WORKERS # each worker will get 50 images in every batch
```

The following part handles the data pipeline. When *AutoShardPolicy* is set to 
*DATA*, each worker gets the whole data but only uses a related part of the 
batch. In the data pipeline, data is mapped, then shuffled, and the batch is 
created.

```python
options = tf.data.Options()
options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA

train_dataset = datasets['train'].map(scale).repeat().shuffle(1000).batch(GLOBAL_BATCH_SIZE)
train_dataset = train_dataset.with_options(options)
```

The following function is used to build the Keras model.

```python
def cnn_model():
	model = tf.keras.Sequential()
	model.add(Conv2D(32, kernel_size=(3,3), activation='relu',input_shape=(28, 28, 1)))
	model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2,2)))
	model.add(Flatten())
	model.add(Dense(64, activation='relu'))
	model.add(Dense(10, activation='softmax'))

	model.compile(
		loss=tf.keras.losses.sparse_categorical_crossentropy,
		optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, amsgrad=True),
		metrics=['accuracy'])
	return model
```

The last part of the code creates the model and trains it. The main difference 
between non-distributed training and distributed training is that the model is 
created in *strategy.scope*. This model will be trained for 50 epochs. 
*steps_per_epoch* should be set in the distributed training. When it is set to 
50, it will use 50 batches of data for an epoch.

```python
with strategy.scope():
	multi_worker_model = cnn_model()

multi_worker_model.summary()
multi_history = multi_worker_model.fit(x=train_dataset, epochs=50, steps_per_epoch=50)
```

In order to run the tutorial, we need two different terminal windows. We need 
to run the following code on both terminals by changing the *index* parameter.

```bash
python file_name.py --index 0 # on the first terminal
python file_name.py --index 1 # on the second terminal
```

## Horovod

We can also run distributed training using Horovod [@fa18-523-68-sergeev2018horovod]. 
Before TensorFlow has a collective all-reduce strategy (Multi Worker Mirrored 
Strategy), UBER developed a package called Horovod. It showed better 
performance than TensorFlow’s parameter strategy approach when it was released. 
It runs on top of MPI, and there is no need to run the code one by one on each 
worker. The code change is minimal compared to non-distributed models. The 
*horovod.init* function should be called at the top of code, and the optimizer 
of the model should be wrapped with *horovod.DistributedOptimizer(optimizer)*. 
To run distributed training on two nodes with 2 GPUs each:

```bash
horovodrun -np 4 -H server1:2,server2:2 python train.py
```

Since  some of distributed TensorFlow functionalities are experimental, 
Horovod can be used for synchronous distributed training. 

## Conclusion

Datasets are getting larger and larger, and it is impractical to train a model 
on a single device. In this paper, TensorFlow distribution strategies and 
related TensorFlow components are introduced. When multiple machines/multiple 
GPUs are available, distributed TensorFlow reduces the training time 
dramatically. The usage is mostly straightforward, and developers of the 
TensorFlow are making the distributed TensorFlow easier. However, it is still 
under heavy development, and there is no backward combability. Therefore, code 
changes are needed when the TensorFlow version is updated.

