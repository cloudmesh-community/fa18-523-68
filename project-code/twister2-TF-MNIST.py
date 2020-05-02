#  // Licensed under the Apache License, Version 2.0 (the "License");
#  // you may not use this file except in compliance with the License.
#  // You may obtain a copy of the License at
#  //
#  // http://www.apache.org/licenses/LICENSE-2.0
#  //
#  // Unless required by applicable law or agreed to in writing, software
#  // distributed under the License is distributed on an "AS IS" BASIS,
#  // WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  // See the License for the specific language governing permissions and
#  // limitations under the License.



import numpy as np

from twister2.TSetContext import TSetContext
from twister2.Twister2Environment import Twister2Environment
from twister2.tset.fn.SourceFunc import SourceFunc
import argparse
import time

parser = argparse.ArgumentParser(description='Twister2 Distributed MNIST Argument Parser')
parser.add_argument('--numberOfWorkers', required=True, help='number of workers')
parser.add_argument('--cpu', required=True, help='number of CPU per worker')
parser.add_argument('--ram', required=True, help='memory per worker')
parser.add_argument('--useGPU', required=True, help='train on GPU')

args = parser.parse_args()

# this  is the entry point of the application
# specify how many workers you want
env = Twister2Environment(resources=[{"cpu": int(args.cpu), "ram": int(args.ram), "instances": int(args.numberOfWorkers)}])


worker_id = env.worker_id
numberOfWorkers = int(args.numberOfWorkers)
device = 'cpu' if args.useGPU == 'False' else 'gpu'


# TSet Source Function
# TSet is similar to Spark's RDD
class MNISTTrainingSource(SourceFunc):

    def __init__(self, num_worker,worker_id):
        super(MNISTTrainingSource, self).__init__()
        import os
        import math
        # to run this operation on CPU
        os.environ['CUDA_VISIBLE_DEVICES']=''
        import tensorflow as tf
        
        self.index = -1
        start_time = time.time()
        data_url = 'https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz'
        data_path = tf.keras.utils.get_file('mnist.npz', data_url)
        download_time = int(time.time() - start_time)
        
        if worker_id == 0:
            with open('/scratch/num_worker_' + str(numberOfWorkers) + '_device_' + device +'.txt', 'w') as f:
                f.write('download time: ' +  str(download_time) + '\n')
            
        

        with np.load(data_path) as data:
            record_per_worker = math.ceil(len(data['x_train']) / num_worker)
            self.train_x = data['x_train'][worker_id * record_per_worker: (worker_id + 1) * record_per_worker]
            self.train_y = data['y_train'][worker_id * record_per_worker: (worker_id + 1) * record_per_worker]
            
    def has_next(self):
        return self.index < self.train_x.shape[0] - 1

    def next(self):
        self.index += 1
        return [self.train_x[self.index], self.train_y[self.index]]




    
# A mapping function to convert data to TensorFlow's binary data format
def toTFRRecord(itr, collector, ctx:TSetContext):
    import os
    
    # to run this operation on CPU
    os.environ['CUDA_VISIBLE_DEVICES']=''
    import tensorflow as tf

    def _int64_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def _float_feature(value):
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

    def _bytes_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
    def serialize_example(feature0, feature1):
        feature = {
            'image': _bytes_feature(feature0),
            'label': _int64_feature(feature1)
        }
        example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
        return example_proto.SerializeToString()

    worker_id = ctx.worker_id

    file_name = '/scratch/mnist.tfrecord' + str(worker_id)
    with tf.io.TFRecordWriter(file_name) as writer:
        count = 0
        for record in itr:
            writer.write(serialize_example(record[0].tostring(), record[1]))
            count += 1
            if count % 1000 == 0:
                print('WORKER-' + str(worker_id) + ':', '',count , 'records has been saved')
    print('WORKER-' + str(worker_id) + ':', '',count , 'records has been saved in total')
    collector.collect(1)


# Convert Data to Tfrecord format
start_time = time.time()
data = env.create_source(MNISTTrainingSource(numberOfWorkers, worker_id), numberOfWorkers).cache()
compute = data.compute(toTFRRecord)
compute.for_each(lambda x: print(x))
end_time = time.time()


# Log time to a txt file
if worker_id == 0:
    time_diff = int(end_time - start_time)
    with open('/scratch/num_worker_' + str(numberOfWorkers) + '_device_' + device +'.txt', 'a') as f:
        f.write('dataset_creation_time: ' +  str(time_diff) + '\n')


#  Training Fucntion
def trainModel(itr, collector, ctx:TSetContext):
    import os
    import json
    worker_id = ctx.worker_id
    num_workers = ctx.parallelism
    
    # required system variable for Distributed TensorFlow
    def setTFCONFIG():
        start_port = 8000
        cluster = {'worker': []}
        for i in range(num_workers):
            cluster['worker'].append('localhost:' + str(start_port + i))
        os.environ['TF_CONFIG'] = json.dumps({'cluster': cluster,
                                              'task': {'type': 'worker', 'index': worker_id}})
    
    setTFCONFIG()
    
    
    # Train using CPU or GPU
    if args.useGPU == "True":
        os.environ['CUDA_VISIBLE_DEVICES'] = str(ctx.worker_id)
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    
    import tensorflow as tf
    
    # TensorFlow parameters to limit number of CPUs per worker
    tf.config.threading.set_intra_op_parallelism_threads(int(args.cpu))
    tf.config.threading.set_inter_op_parallelism_threads(int(args.cpu))
    
    from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    
    # Distribution Strategy
    strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
    
    # Scaling mapping function
    def scale(image, label):
        image = tf.cast(image, tf.float32)
        image /= 255
        return image, label
    
    # decode data from tfrecord file
    def decode(serialized_example):
        features = tf.io.parse_single_example(
            serialized_example,
            features={
                'image': tf.io.FixedLenFeature([], tf.string),
                'label': tf.io.FixedLenFeature([], tf.int64),
            })
        image = tf.io.decode_raw(features['image'], tf.uint8)
        label = tf.cast(features['label'], tf.int32)
        image.set_shape((28 * 28))
        image = tf.reshape(image, [28,28,1])
        return image, label

    # We give a different file for each worker. Therefore, there is no need to shard data.
    file_name = '/scratch/mnist.tfrecord' + str(worker_id)
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
    
    
    # TensorFlow sets batch size based on number of workers. batch_per_worker = batch/number_of_worker
    batch_size = 50
    global_batch_size = batch_size * num_workers
    
    # input pipeline. Read raw data(tfrecord) -> decode -> scale -> repeat -> shuffle -> create batches
    train_dataset = tf.data.TFRecordDataset(file_name).map(decode).map(scale).repeat().shuffle(1000).batch(global_batch_size)
    

    train_dataset = train_dataset.with_options(options)

    # model function
    def cnn_model():
        model = tf.keras.Sequential()
        model.add(Conv2D(32, kernel_size=(3,3), activation='relu',input_shape=(28, 28, 1)))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Conv2D(128, kernel_size=(3,3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Flatten())
        model.add(Dense(64, activation='relu'))
        model.add(Dense(10, activation='softmax'))
        model.compile(
            loss=tf.keras.losses.sparse_categorical_crossentropy,
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, amsgrad=True),
            metrics=['accuracy'])
        return model


    # create Keras model in this scope. Otherwise, Distributed TensorFlow won't work
    with strategy.scope():
        multi_worker_model = cnn_model()
    
    # only print progress for the first worker
    verbose = 1 if worker_id== 0 else 0
    
    # print model summary
    if worker_id == 0:
        multi_worker_model.summary()
        
    # training
    multi_history = multi_worker_model.fit(x=train_dataset, epochs=20, steps_per_epoch=50, verbose=verbose)
    
    # save the model and log the accuracy and loss to a txt file
    if worker_id == 0:

        multi_worker_model.save('/scratch/num_worker_' + str(num_workers) + '_device_' + device +'.h5')
        with open('/scratch/num_worker_' + str(num_workers) + '_device_' + device +'.txt', 'a') as f:
            f.write('accuracy: ' + ', '.join([str(i) for i in multi_history.history['accuracy']]) + '\n')
            f.write('loss: ' + ', '.join([str(i) for i in multi_history.history['loss']]) + '\n')

    collector.collect(1)


    
start_time = time.time()
train = data.compute(trainModel)
train.for_each(lambda x: print(x))


# Log the training time to a txt file.
if worker_id == 0:
    training_time = time.time() - start_time
    with open('/scratch/num_worker_' + str(numberOfWorkers) + '_device_' + device +'.txt', 'a') as f:
        f.write('training time: ' +  str(training_time) + '\n')

        
print('the task has been completed')
