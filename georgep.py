from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import re
from ast import literal_eval

import numpy as np
import tensorflow.compat.v1 as tf
from tensorflow.python import ipu

from google.protobuf import text_format
from tensorflow.contrib import graph_editor as ge

# Setup IPU config
config = ipu.utils.create_ipu_config()
config = ipu.utils.auto_select_ipus(config, 1)
ipu.utils.configure_ipu_system(config)

# Batch data in 
batches_per_step = 50 
gradient_accumulation_batches = 32
inference_batches_per_step = batches_per_step * gradient_accumulation_batches


# tensorflow data loader function
def data_fn(num_examples=1000):
    # Generate random data
    # dtype = np.float16 if args.dtype == 'float16' else np.float32

    # Try mixed precision next
    dtype = np.float32

    # bs = args.batch_size_train if mode == tf.estimator.ModeKeys.TRAIN else args.batch_size_infer
    bs = 22
    batches_per_step = 50
    l = batches_per_step * bs
    # if count_only:
        # return l * 10

    x = np.random.uniform(size=(num_examples, 100, 221, 6)).astype(dtype)

    # This fails when dataset is very large
    # ValueError: Cannot create a tensor proto whose content is larger than 2GB.
    # Error goes away if you enable eager mode 
    dataset = tf.data.Dataset.from_tensor_slices(x)
    dataset = dataset.batch(bs, drop_remainder=True).prefetch(l).cache().repeat()
    
    return dataset

num_examples = 100
ds = data_fn(num_examples=num_examples)
infeed = ipu.ipu_infeed_queue.IPUInfeedQueue(ds,'infeed', replication_factor=1)
outfeed = ipu.ipu_outfeed_queue.IPUOutfeedQueue('outfeed', replication_factor=1)

# real data function
def getExamples():
    inputstring=open('file3.txt', 'r').read()
    inputstring=re.sub(r"([^[])\s+([^]])", r"\1, \2", inputstring)
    table=np.array(literal_eval(inputstring))
    table=np.squeeze(table)
    print("table shape is {0}".format(table.shape))
    return table

def model(features):
    graph = tf.get_default_graph()
    g1 = tf.GraphDef()
    with tf.gfile.GFile('model.pb', 'rb') as fid:
        serialized_graph = fid.read()
        g1.ParseFromString(serialized_graph)
    from tensorflow.graph_util import extract_sub_graph
    g1 = extract_sub_graph(g1,['InceptionV3/Predictions/Softmax'])
    output = tf.import_graph_def(g1,
                              name='',
                              input_map={'input:0': features},
                              return_elements=['InceptionV3/Predictions/Softmax'])
    return output

def wrapped_model():
    def loop_body(features):
        output = model(features)
        return outfeed.enqueue(output)
    return ipu.loops.repeat(inference_batches_per_step,
                            loop_body,
                            infeed_queue=infeed)


def testInput():
    # Scenarios:
    # 1. CPU and unknown batch dim (works)
    # 2. CPU and known batch dim (fails)
    # 3. IPU and known batch dim (also fails)

    scenario = 3

    if scenario == 1:
        x = tf.placeholder(tf.float32, (None, 100, 221, 6))
        output = model(x)

    elif scenario == 2:
        x = tf.placeholder(tf.float32, (22, 100, 221, 6))
        output = model(x)

    elif scenario == 3:
        with tf.device('/device:IPU:0'):
            output = ipu.ipu_compiler.compile(wrapped_model, [])

    with tf.Session() as sess:
        sess.run(infeed.initializer)
        sess.run(output)
        outfeed_dequeue_op = outfeed.dequeue()
        pass


if __name__ == "__main__":
    testInput()