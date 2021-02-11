from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import re
from ast import literal_eval
import time
from functools import partial

import numpy as np
import tensorflow.compat.v1 as tf
from tensorflow.python.ipu import utils, ipu_compiler, scopes, loops, ipu_infeed_queue, ipu_outfeed_queue
from tensorflow.python import ipu


from google.protobuf import text_format

# tensorflow data loader function
def data_fn(num_examples=1000):
    # Generate random data
    # dtype = np.float16 if args.dtype == 'float16' else np.float32

    # Try mixed precision next
    dtype = np.float32

    # bs = args.batch_size_train if mode == tf.estimator.ModeKeys.TRAIN else args.batch_size_infer
    bs = 4
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

# real data function
def getExamples():
    inputstring=open('file3.txt', 'r').read()
    inputstring=re.sub(r"([^[])\s+([^]])", r"\1, \2", inputstring)
    table=np.array(literal_eval(inputstring))
    table=np.squeeze(table)
    print("table shape is {0}".format(table.shape))
    return table

# synthetic data function with no data preloader
def getSyntheticExamples(num_examples=50000):
    table = np.random.rand(num_examples, 100, 221, 6)
    print("table shape is {0}".format(table.shape))
    return table


def testInput():

    # Setup IPU config
    config = utils.create_ipu_config()
    config = utils.auto_select_ipus(config, 1)
    utils.configure_ipu_system(config)

    # Setup data loader
    num_examples = 100
    ds = data_fn(num_examples=num_examples)
    infeed = ipu.ipu_infeed_queue.IPUInfeedQueue(ds,'infeed', replication_factor=1)
    outfeed = ipu.ipu_outfeed_queue.IPUOutfeedQueue('outfeed', replication_factor=1)

    # Batch data in 
    batches_per_step = 50 
    gradient_accumulation_batches = 32
    inference_batches_per_step = batches_per_step * gradient_accumulation_batches

    def test_loop_op():
        def body(features):
            # predictions = model_fn(inference_features=features, mode=tf.estimator.ModeKeys.PREDICT, params=[], args=args)
            gdv = tf.Graph()
            with gdv.as_default():
                g1 = tf.GraphDef()
            with tf.gfile.GFile('model.pb', 'rb') as fid:
                serialized_graph = fid.read()
                g1.ParseFromString(serialized_graph)
                tf.import_graph_def(g1, input_map={features : gdv.get_tensor_by_name('input:0')}, name='')
                inp_tensor=gdv.get_tensor_by_name('input:0')
                out_tensor=gdv.get_tensor_by_name('InceptionV3/Predictions/Softmax:0')
            
            proba = sess.run(out_tensor,{inp_tensor:ds})
            outfeed_op = outfeed.enqueue(proba)
            return outfeed_op

        return ipu.loops.repeat(inference_batches_per_step, body, infeed_queue=infeed)



    with tf.Session() as sess:
        # inp_tensor=gdv.get_tensor_by_name('input:0')
        # out_tensor=gdv.get_tensor_by_name('InceptionV3/Predictions/Softmax:0')

        with ipu.scopes.ipu_scope('/device:IPU:0'):
            compiled = ipu.ipu_compiler.compile(partial(test_loop_op))

        ipu.utils.move_variable_initialization_to_cpu()
        init_g = tf.global_variables_initializer()
        sess.run(infeed.initializer)
        sess.run(init_g)
        outfeed_dequeue_op = outfeed.dequeue()

        tic = time.time()
        sess.run(compiled)
        

        
        # proba=sess.run(out_tensor,{inp_tensor:image_np})
        
        toc = time.time()
        duration = toc - tic
        num_images = len(image_np)

        print("Total time taken: {0} seconds".format(duration))
        print("Number of examples: {0}".format(num_images))
        print("Throughput: {0} im/s".format(num_images / duration))

if __name__ == "__main__":
    testInput()

# Ok so instead of importing the graph in the graph scope, i import the graph in the body function and in the input map dict I need to make the key the features variables and the value gdv.get_tensor_by_name('input:0')  