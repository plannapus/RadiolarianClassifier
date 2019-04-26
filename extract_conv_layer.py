import os
os.chdir("environment")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

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

model_file = "models/102.pb"
img_name = ["datasets/datasets102-202-302/dataset102/Antarctissa denticulata/Antarctissa denticulata 689B-3-3,116 Axioskop 40X jr-3a.jpg",
            "datasets/datasets102-202-302/dataset102/Antarctissa denticulata/Antarctissa denticulata 689B-3-3,116 Axioskop 40X jr-3b.jpg",
            "datasets/datasets102-202-302/dataset102/Antarctissa cylindrica/Antarctissa cylindrica 689B-3-3,116 Axioskop 40X jr-3a.jpg",
            "datasets/datasets102-202-302/dataset102/Antarctissa cylindrica/Antarctissa cylindrica 689B-3-3,116 Axioskop 40X jr-3b.jpg",
            "datasets/datasets102-202-302/dataset102/Cycladophora golli/Cycladophora golli _278-20-1,77 OlyBH2 30X dbl 5.jpg",
            "datasets/datasets102-202-302/dataset102/Cycladophora spongothorax /Cycladophora spongothorax 689B-4-4,116 Olympus BH-2 30X dbl 1a.jpg",
            "datasets/datasets102-202-302/dataset102/Cycladophora spongothorax /Cycladophora spongothorax 689B-4-4,116 Olympus BH-2 30X dbl 14a.jpg"]

graph = load_graph(model_file)
input_height = 224
input_width = 224
input_mean = 128
input_std = 128
conv_layers = [k for k in graph.get_operations() if k.type=="Relu6"]

for k in xrange(len(img_name)):
    a=[]
    img = read_tensor_from_image_file(img_name[k],input_height=input_height,input_width=input_width,input_mean=input_mean,input_std=input_std)
    sess = tf.Session(graph=graph)
    for i in xrange(len(conv_layers)):
        a.append(sess.run(conv_layers[i].outputs[0],{graph.get_operation_by_name("import/input").outputs[0]: img}))

    for i in xrange(15):
        for j in xrange(32):
            b = a[i][0,0:,0:,j]
            np.savetxt("visualization2/%s_conv%i_layer%i.csv" % (os.path.basename(img_name[k]),i,j), b, delimiter='\t')
