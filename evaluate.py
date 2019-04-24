# ==============================================================================
#
#  Classify.py
#  Radiolarian classifier
#
#  Ryan Jay Gray (with modifications by Johan Renaudie)
#  Radiolarian project 2017-18
#
# Example usage:
#  import classify
#
#  model_file = "graph.pb"
#  dataset = "dataset_directory"
#  label_file = "retrained_labels.txt"
#
#  classify.evaluateDirectory(model_file,dataset,label_file,top_n,save_file)
#
# Also may be used as script with arguments --model_file, etc.
#
# Portions Copyright 2017 The TensorFlow Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from datetime import datetime
import re

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

def load_labels(label_file):
  label = []
  proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
  for l in proto_as_ascii_lines:
    label.append(l.rstrip())
  return label

def evaluateDirectory(model_file,file_name,label_file,top_n,save_file):
  start_time = datetime.now()
  input_height = 224
  input_width = 224
  input_mean = 128
  input_std = 128
  input_layer = "input"
  output_layer = "final_result"
  graph = load_graph(model_file)
  # The extension to search for
  exten = '.jpg'
  totalCount = 0
  labels = load_labels(label_file)
  allResults = []

  for dirpath, dirnames, files in os.walk(file_name):
    files.sort()
    for name in files:
      if name.lower().endswith(exten):
        t = read_tensor_from_image_file(os.path.join(dirpath, name),
         input_height=input_height,
         input_width=input_width,
         input_mean=input_mean,
         input_std=input_std)

        temp = name.split()
        del temp[2] #remove the serialization
        name = " ".join(temp)

        input_name = "import/" + input_layer
        output_name = "import/" + output_layer
        input_operation = graph.get_operation_by_name(input_name)
        output_operation = graph.get_operation_by_name(output_name)

        with tf.Session(graph=graph) as sess:
          results = sess.run(output_operation.outputs[0], {input_operation.outputs[0]: t})
        results = np.squeeze(results)
        speciesName = " ".join(name.lower().split()[0:2])
        baseName = re.sub(exten,'',name)
        specimenName = re.sub('[a-zA-Z]$','',baseName)
        allResults.append({'image_name': name, 'specimen_name': specimenName, 'species_name': speciesName, 'results': results, 'correctImage': None, 'correctSpecimen': None})
        totalCount = totalCount + 1

  for i in allResults:
    top_k = i['results'].argsort()[::-1]
    classificationName = []
    for k in range(top_n): classificationName += [labels[top_k[k]].lower()]
    i['correctImage'] = True if i['species_name'] in classificationName else False

  accuracy_picture = sum([k['correctImage'] for k in allResults])*100.0/totalCount
  specimens = set([k['specimen_name'] for k in allResults])

  for i in specimens:
    all_focal_planes = [k for k in allResults if k['specimen_name']==i]
    m = [max(k['results']) for k in all_focal_planes]
    best_guess = [k for k in all_focal_planes if max(k['results'])==max(m)][0]
    top_k = best_guess['results'].argsort()[::-1]
    classificationName = []
    for k in range(top_n): classificationName += [labels[top_k[k]].lower()]
    correctSpecimen = True if best_guess['species_name'] in classificationName else False
    for j in allResults:
      if j['specimen_name'] == i:
        j['correctSpecimen'] = correctSpecimen

  accuracy_specimen = sum([k['correctSpecimen'] for k in allResults])*100.0/totalCount
  print("\nImages total: "+str(totalCount)+". Raw percent correctly classified: {:.2f}%".format(accuracy_picture))
  print("\nSpecimens total: "+str(len(specimens))+". Percent correctly classified: {:.2f}%".format(accuracy_specimen))
  resultFile = open(save_file,"w+")
  resultFile.write("File\t%s\tcorrectImage\tcorrectSpecimen\n" % ("\t".join(labels),))
  for i in allResults:
    resultFile.write("%s\t%s\t%s\t%s\n" % (i['image_name'],'\t'.join(map(str,i['results'])),i['correctImage'],i['correctSpecimen']))
  resultFile.close()

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--model_file", help="graph/model to be executed")
  parser.add_argument("--dataset", help="data to be classified")
  parser.add_argument("--label_file", help="name of file containing labels")
  parser.add_argument("--top_n", help="Number n top_n")
  parser.add_argument("--save_file", help="name of the file where the results will be saved")
  args = parser.parse_args()

  # Suppress TF warnings
  os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
  print("\033[1;32;40m\n\n\n\nRaw classification results for all focal planes\033[0;37;40m")
  if args.model_file:
    model_file = args.model_file
  if args.dataset:
    file_name = args.dataset
  if args.label_file:
    label_file = args.label_file
  if args.top_n:
    top_n = args.top_n
  if args.save_file:
    save_file = args.save_file
  evaluateDirectory(model_file,file_name,label_file,top_n,save_file)
