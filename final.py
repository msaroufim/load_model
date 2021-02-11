# If you think this is useful, let us know.

import numpy as np
import tensorflow.compat.v1 as tf
from tensorflow.python import ipu

# REQUIRES dataset to return a list/tuple structure - dicts/nested structures not supported.
def ipu_imported_graph_builder(path_to_graph_def,
                               input_names,
                               dataset,
                               output_names,
                               feed_names,
                               iterations_per_run,
                               name=None,
                               replication_factor=1):
  """
  Arguments:
  * input_names - names of input tensors in the graph
  * dataset - the dataset to be used to feed the graph - must output the same
    number of tensors as the number of names in `input_names`
  * output_names - the names of the output nodes in the imported graph.
      TODO currently all outputs must be tensors, not just tf.Operation's.
  * feed_names - string - name to be used by the infeed/outfeed queues.
  * iterations_per_run - number of iterations of the model to execute per each
    session.run
  * name - (Optional) A prefix that will be prepended to the names in graph_def.
    Note that this does not apply to imported function names. Defaults to
    "import".
  * replication_factor - (Optional) the replication factor of the model

  Returns:
  * Operation to execute the graph with
  * The infeed queue which is used to feed the inputs.
  * The outfeed queue which is to be used to access the outputs.

  """
  def convert_to_list(x):
    if not isinstance(x, (list, tuple)):
      return [x]
    return x
  
  input_names = convert_to_list(input_names)
  output_names = convert_to_list(output_names)

  infeed = ipu.ipu_infeed_queue.IPUInfeedQueue(dataset, feed_names + 'infeed', replication_factor=replication_factor)
  outfeed = ipu.ipu_outfeed_queue.IPUOutfeedQueue(feed_names + 'outfeed', replication_factor=replication_factor)

  # Inner loop function.
  def model(*args):
    inputs = convert_to_list(args)
    assert len(inputs) == len(input_names)
    input_map = dict(zip(input_names, args))

    g1 = tf.GraphDef()
    with tf.gfile.GFile(path_to_graph_def, 'rb') as fid:
        serialized_graph = fid.read()
        g1.ParseFromString(serialized_graph)
    return tf.import_graph_def(g1,
                               name=name,
                               input_map=input_map,
                               return_elements=output_names)

  def wrapped_model():
    def loop_body(*args):
        return outfeed.enqueue(model(*args))
    return ipu.loops.repeat(iterations_per_run,
                            loop_body,
                            infeed_queue=infeed)

  operation = ipu.ipu_compiler.compile(wrapped_model, [])
  return operation, infeed, outfeed

# Example run...

# Setup IPU config
config = ipu.utils.create_ipu_config()
config = ipu.utils.auto_select_ipus(config, 1)
ipu.utils.configure_ipu_system(config)

# Batch data in 
def data_fn(num_examples=1000):
    x = np.random.uniform(size=(num_examples, 100, 221, 6)).astype(np.float32)
    dataset = tf.data.Dataset.from_tensor_slices(x)
    dataset = dataset.batch(10, drop_remainder=True).repeat()
    return dataset

def export_graph():
  graph = tf.Graph()
  with graph.as_default():
    x = tf.placeholder(tf.float32, (None, 100), "test")
    y = x * x
  return graph.as_graph_def(), [x.name], [y.name]

def my_export_graph():
  gdv = tf.Graph()
  with gdv.as_default():
      g1 = tf.GraphDef()
      with tf.gfile.GFile('model.pb', 'rb') as fid:
          serialized_graph = fid.read()
          g1.ParseFromString(serialized_graph)
          tf.import_graph_def(g1, name='')
  
  return gdv.as_graph_def(), ['input:0'], ['InceptionV3/Predictions/Softmax:0']

graph_def, input_names, output_names = my_export_graph()

# Export the simple graph.
tf.train.write_graph(graph_def, ".", "network.pb", False)
num_examples = 1000
dataset = data_fn(num_examples)
# Import it onto the IPU.
with tf.device('/device:IPU:0'):
  operation, infeed, outfeed = ipu_imported_graph_builder("network.pb",
                                                          input_names,
                                                          dataset,
                                                          output_names,
                                                          "test",
                                                          num_examples)
dequeue = outfeed.dequeue()

with tf.Session() as sess:
  sess.run(infeed.initializer)

  import time
  tic = time.time()
  sess.run(operation)
  print(sess.run(dequeue))
  toc = time.time()
  duration = toc - tic
  print(f'Total duration: {duration}')
  throughput = 2000 / duration 
  print(f'Throughput: {throughput} im/s')