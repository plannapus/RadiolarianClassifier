import os
os.chdir("environment")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from lime import lime_image

def load_graph(model_file):
  graph = tf.Graph()
  graph_def = tf.GraphDef()
  with open(model_file, "rb") as f:
    graph_def.ParseFromString(f.read())
  with graph.as_default():
    tf.import_graph_def(graph_def)
  return graph

def read_tensor_from_image_file(file_name, input_height=299, input_width=299,
				input_mean=0, input_std=255):
  input_name = "file_reader"
  file_reader = tf.read_file(file_name, input_name)
  if file_name.endswith(".png"):
    image_reader = tf.image.decode_png(file_reader, channels = 3, name='png_reader')
  elif file_name.endswith(".gif"):
    image_reader = tf.squeeze(tf.image.decode_gif(file_reader, name='gif_reader'))
  elif file_name.endswith(".bmp"):
    image_reader = tf.image.decode_bmp(file_reader, name='bmp_reader')
  else:
    image_reader = tf.image.decode_jpeg(file_reader, channels = 3, name='jpeg_reader')
  float_caster = tf.cast(image_reader, tf.float32)
  dims_expander = tf.expand_dims(float_caster, 0)
  resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
  normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
  sess = tf.Session()
  result = sess.run(normalized)
  return result

model_file = "models/302.pb"
img_name = ["datasets/datasets102-202-302/dataset302/Antarctissa denticulata/Antarctissa denticulata 689B-3-3,116 Axioskop 40X jr-3a.jpg",
            "datasets/datasets102-202-302/dataset302/Antarctissa denticulata/Antarctissa denticulata 689B-3-3,116 Axioskop 40X jr-3b.jpg",
            "datasets/datasets102-202-302/dataset302/Antarctissa cylindrica/Antarctissa cylindrica 689B-3-3,116 Axioskop 40X jr-3a.jpg",
            "datasets/datasets102-202-302/dataset302/Antarctissa cylindrica/Antarctissa cylindrica 689B-3-3,116 Axioskop 40X jr-3b.jpg",
            "datasets/datasets102-202-302/dataset302/Antarctissa cylindrica/Antarctissa cylindrica 693A-18-4,101 Axioskop 40X jr-19a.jpg",
            "datasets/datasets102-202-302/dataset302/Antarctissa cylindrica/Antarctissa cylindrica 693A-18-4,101 Axioskop 40X jr-22a.jpg",
            "datasets/datasets102-202-302/dataset302/Cycladophora golli/Cycladophora golli _278-20-1,77 OlyBH2 30X dbl 5.jpg",
            "datasets/datasets102-202-302/dataset302/Cycladophora spongothorax /Cycladophora spongothorax 689B-4-4,116 Olympus BH-2 30X dbl 1a.jpg",
            "datasets/datasets102-202-302/dataset302/Cycladophora spongothorax /Cycladophora spongothorax 689B-4-4,116 Olympus BH-2 30X dbl 14a.jpg"]

graph = load_graph(model_file)
input_operation = graph.get_operation_by_name("import/input")
output_operation = graph.get_operation_by_name("import/final_result")
sess = tf.Session(graph=graph)
def predict_fn(images):
    return sess.run(output_operation.outputs[0], feed_dict={input_operation.outputs[0]: img})[0]

res=[]
for i in img_name:
    img = read_tensor_from_image_file(i,
         input_height = 224,
         input_width = 224,
         input_mean = 128,
         input_std = 128)
    explainer = lime_image.LimeImageExplainer()
    res.append(explainer.explain_instance(img, predict_fn, top_labels=5, hide_color=0, num_samples=1000))
#
# Traceback (most recent call last):
#   File "<stdin>", line 8, in <module>
#   File "/usr/local/lib/python2.7/site-packages/lime/lime_image.py", line 179, in explain_instance
#     random_seed=random_seed)
#   File "/usr/local/lib/python2.7/site-packages/lime/wrappers/scikit_image.py", line 105, in __init__
#     kwargs = self.filter_params(quickshift)
#   File "/usr/local/lib/python2.7/site-packages/lime/wrappers/scikit_image.py", line 84, in filter_params
#     if has_arg(fn, name):
#   File "/usr/local/lib/python2.7/site-packages/lime/utils/generic_utils.py", line 21, in has_arg
#     arg_spec = inspect.getargspec(fn.__call__)
#   File "/usr/local/Cellar/python@2/2.7.17/Frameworks/Python.framework/Versions/2.7/lib/python2.7/inspect.py", line 825, in getargspec
#     raise TypeError('{!r} is not a Python function'.format(func))
# TypeError: <method-wrapper '__call__' of builtin_function_or_method object at 0x1241d4c80> is not a Python function
