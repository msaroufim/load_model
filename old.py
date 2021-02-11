from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import re
from ast import literal_eval

import numpy as np
import tensorflow.compat.v1 as tf
from tensorflow.python.ipu import utils, ipu_compiler, scopes, loops, ipu_infeed_queue, ipu_outfeed_queue
from tensorflow.python.ipu.scopes import ipu_scope


from google.protobuf import text_format




def getExamples():
    inputstring=open('file3.txt', 'r').read()
    inputstring=re.sub(r"([^[])\s+([^]])", r"\1, \2", inputstring)
    table=np.array(literal_eval(inputstring))
    table=np.squeeze(table)
    print("table shape is {0}".format(table.shape))
    return table

def getSyntheticExamples(num_examples=10000):
    table = np.random.rand(num_examples, 100, 221, 6) #.astype(np.float16)
    print("table shape is {0}".format(table.shape))
    return table


def testInput():
    config = utils.create_ipu_config()
    config = utils.auto_select_ipus(config, 1)
    config = utils.create_ipu_config(profiling=True, use_poplar_text_report=True)
    utils.configure_ipu_system(config)


    # config = utils.set_convolution_options(config, {"partialsType": str('half')})
    # config = utils.set_matmul_options(config, {"partialsType": str('half')})

    gdv = tf.Graph()
    with gdv.as_default():
        g1 = tf.GraphDef()
        # Load model with pywrap isntead? https://github.com/graphcore/examples/blob/master/applications/tensorflow/cnns/training/weight_avg.py#L33
        with tf.gfile.GFile('model.pb', 'rb') as fid:
            serialized_graph = fid.read()
            g1.ParseFromString(serialized_graph)
            tf.import_graph_def(g1, name='')
    
    # How do I make this play nicely with the gdv scope?
    # with ipu_scope("/device:IPU:0"):
        # gdv = ipu_compiler.compile(gdv) # Error 'Graph' object is not Callable
        # gdb = ipu_compiler.compile(tf.function(gdv)) # unsupported callable 

    with tf.Session(graph=gdv) as sess:
        inp_tensor=gdv.get_tensor_by_name('input:0')
        out_tensor=gdv.get_tensor_by_name('InceptionV3/Predictions/Softmax:0')
        # image_np = getExamples()
        image_np = getSyntheticExamples()
        np.set_printoptions(threshold=np.inf)
        
        import time
        tic = time.time()

        # This is new and doesn't crash
        # But doesn't seem to do anything either
        with ipu_scope("/device:IPU:0"):
            proba=sess.run(out_tensor,{inp_tensor:image_np})
        
        toc = time.time()
        duration = toc - tic
        num_images = len(image_np)

        print("Total time taken: {0} seconds".format(duration))
        print("Number of examples: {0}".format(num_images))
        print("Throughput: {0} im/s".format(num_images / duration))

        
        #print(proba)


if __name__ == "__main__":
    testInput()