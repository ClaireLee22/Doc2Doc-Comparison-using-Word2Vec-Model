# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
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
"""This program is based on Basic word2vec example."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math
import os
import sys
import argparse
import random
from tempfile import gettempdir
from six.moves import xrange

import numpy as np
import tensorflow as tf
import codecs
from os import listdir

from tensorflow.contrib.tensorboard.plugins import projector

# Give a folder path as an argument with '--log_dir' to save
# TensorBoard summaries. Default is a log folder in current directory.
current_path = os.path.dirname(os.path.realpath(sys.argv[0]))

parser = argparse.ArgumentParser()
parser.add_argument(
    '--log_dir',
    type=str,
    default=os.path.join(current_path, 'log'),
    help='The log directory for TensorBoard summaries.')
FLAGS, unparsed = parser.parse_known_args()

# Create the directory for TensorBoard variables if there is not.
if not os.path.exists(FLAGS.log_dir):
  os.makedirs(FLAGS.log_dir)

# Read the data into a list of strings.
def read_data(filename):
  tot_lines = ''
  infile = codecs.open(filename, 'r', 'utf-8')
  inlines = infile.readlines()
  for line in inlines:
    tot_lines += line + " "
  data = tot_lines.split()
  return data


vocabulary = read_data('clean_data.txt')
print('Data size', len(vocabulary))

word_count = len(vocabulary)


def build_dataset(words, n_words):
  """Process raw inputs into a dataset."""
  count = []
  count.extend(collections.Counter(words).most_common(n_words - 1))
  dictionary = dict()
  for word, _ in count:
    dictionary[word] = len(dictionary)
  data = list()
  for word in words:
    index = dictionary.get(word, 0)
    data.append(index)
  reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
  return data, count, dictionary, reversed_dictionary


# Filling 4 global variables:
# data - list of codes (integers from 0 to vocabulary_size-1).
#   This is the original text but words are replaced by their codes
# count - map of words(strings) to count of occurrences
# dictionary - map of words(strings) to their codes(integers)
# reverse_dictionary - maps codes(integers) to words(strings)
data, count, dictionary, reverse_dictionary = build_dataset(
    vocabulary, word_count)
del vocabulary  # Hint to reduce memory.
print('Most common words', count[:5])
print('Sample data', data[:10], [reverse_dictionary[i] for i in data[:10]])

data_index = 0
vocabulary_size = len(count)

print("Unique word count: " + str(len(count)))


#################################################################
#
# This section generate a training batch for the skip-gram model
#
#################################################################
def generate_batch(batch_size, num_skips, skip_window):
  global data_index
  assert batch_size % num_skips == 0
  assert num_skips <= 2 * skip_window
  batch = np.ndarray(shape=(batch_size), dtype=np.int32)
  labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
  span = 2 * skip_window + 1  # [ skip_window target skip_window ]
  buffer = collections.deque(maxlen=span)  # pylint: disable=redefined-builtin
  if data_index + span > len(data):
    data_index = 0
  buffer.extend(data[data_index:data_index + span])
  data_index += span
  for i in range(batch_size // num_skips):
    context_words = [w for w in range(span) if w != skip_window]
    words_to_use = random.sample(context_words, num_skips)
    for j, context_word in enumerate(words_to_use):
      batch[i * num_skips + j] = buffer[skip_window]
      labels[i * num_skips + j, 0] = buffer[context_word]
    if data_index == len(data):
      buffer.extend(data[0:span])
      data_index = span
    else:
      buffer.append(data[data_index])
      data_index += 1
  # Backtrack a little bit to avoid skipping words in the end of a batch
  data_index = (data_index + len(data) - span) % len(data)
  return batch, labels


batch, labels = generate_batch(batch_size=8, num_skips=2, skip_window=1)
for i in range(8):
  print(batch[i], reverse_dictionary[batch[i]], '->', labels[i, 0],
        reverse_dictionary[labels[i, 0]])


#################################################
#
# This section build and train a skip-gram model
#
#################################################

batch_size = 128
embedding_size = 128  # Dimension of the embedding vector.
skip_window = 1  # How many words to consider left and right.
num_skips = 2  # How many times to reuse an input to generate a label.
num_sampled = 64  # Number of negative examples to sample.

# We pick a random validation set to sample nearest neighbors. Here we limit the
# validation samples to the words that have a low numeric ID, which by
# construction are also the most frequent. These 3 variables are used only for
# displaying model accuracy, they don't affect calculation.
valid_size = 16  # Random set of words to evaluate similarity on.
valid_window = 100  # Only pick dev samples in the head of the distribution.
valid_examples = np.random.choice(valid_window, valid_size, replace=False)

all_word_idxs = np.arange(0, vocabulary_size)

graph = tf.Graph()

with graph.as_default():

  # Input data.
  with tf.name_scope('inputs'):
    train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
    train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
    valid_dataset = tf.constant(valid_examples, dtype=tf.int32)
	
  all_word_dataset = tf.constant(all_word_idxs, dtype=tf.int32)

  # Ops and variables pinned to the CPU because of missing GPU implementation
  with tf.device('/cpu:0'):
    # Look up embeddings for inputs.
    with tf.name_scope('embeddings'):
      embeddings = tf.Variable(
          tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
      embed = tf.nn.embedding_lookup(embeddings, train_inputs)

    # Construct the variables for the NCE loss
    with tf.name_scope('weights'):
      nce_weights = tf.Variable(
          tf.truncated_normal(
              [vocabulary_size, embedding_size],
              stddev=1.0 / math.sqrt(embedding_size)))
    with tf.name_scope('biases'):
      nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

    # Compute the average NCE loss for the batch.
    # tf.nce_loss automatically draws a new sample of the negative labels each
    # time we evaluate the loss.
    # Explanation of the meaning of NCE loss:
    #   http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/
    with tf.name_scope('loss'):
      loss = tf.reduce_mean(
          tf.nn.nce_loss(
              weights=nce_weights,
              biases=nce_biases,
              labels=train_labels,
              inputs=embed,
              num_sampled=num_sampled,
              num_classes=vocabulary_size))

    # Add the loss value as a scalar to summary.
    tf.summary.scalar('loss', loss)

    # Construct the SGD optimizer using a learning rate of 1.0.
    with tf.name_scope('optimizer'):
      optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

    # Compute the cosine similarity between minibatch examples and all embeddings.
    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keepdims=True))
    normalized_embeddings = embeddings / norm
    valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings,
                                              valid_dataset)
    similarity = tf.matmul(
        valid_embeddings, normalized_embeddings, transpose_b=True)

    all_embeddings = tf.nn.embedding_lookup(normalized_embeddings, all_word_dataset)
    all_similarity = tf.matmul(all_embeddings, normalized_embeddings, transpose_b=True)

    distance_matrix = tf.placeholder(tf.float32, shape=(None,None,None))
    doc_to_doc_similarities = tf.reduce_sum(tf.reduce_max(distance_matrix, axis=2), axis=1)

    # Merge all summaries.
    merged = tf.summary.merge_all()

    # Add variable initializer.
    init = tf.global_variables_initializer()

    # Create a saver.
    saver = tf.train.Saver()

#################################################
#
# Begin training
#
#################################################

num_steps = 100000

def get_document_names(base_dir):
	doc_names = []
	for f in listdir(base_dir):
		fname = str(f)
		doc_names.append(fname)
	return doc_names

doc_names = get_document_names('clean_files')
num_docs = int(sys.argv[1])
if len(doc_names) < num_docs:
  num_docs = len(doc_names)

print('Number of documents: ', num_docs)

doc_words = {}
for doc in doc_names:
  infile = codecs.open(os.path.join('clean_files', doc), 'r', 'utf-8')
  inlines = infile.readlines()
  tot_lines = ''
  for line in inlines:
    tot_lines += line + " "
  toks = tot_lines.split()

  most_common_count = int(sys.argv[2])
  print('Number of top occuring words in each document: ', most_common_count)
  count = []
  count.extend(collections.Counter(toks).most_common(most_common_count))

  widx_list = []
  for tok, _ in count:
    if tok in dictionary.keys():
      tok_widx = dictionary[tok]
      if tok_widx not in widx_list:
        widx_list.append(tok_widx)
  doc_words[doc] = widx_list
  widx_array = np.asarray(widx_list, 'int32')
  doc_words[doc] = widx_array

doc_similarities = np.zeros(shape=(num_docs, num_docs), dtype='float32')
d1_d2_dist_array_3d = np.zeros(shape=(num_docs,most_common_count,most_common_count), dtype='float32')

config = tf.ConfigProto(allow_soft_placement = True)

with tf.Session(graph=graph, config=config) as session:
  # Open a writer to write summaries.
  writer = tf.summary.FileWriter(FLAGS.log_dir, session.graph)

  # We must initialize all variables before we use them.
  init.run()
  print('Initialized')

  #################################################
  #
  # This section trains the model to word vectors
  #
  #################################################

  average_loss = 0
  for step in xrange(num_steps):
    batch_inputs, batch_labels = generate_batch(batch_size, num_skips,
                                                skip_window)
    feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}

    # Define metadata variable.
    run_metadata = tf.RunMetadata()

    # We perform one update step by evaluating the optimizer op (including it
    # in the list of returned values for session.run()
    # Also, evaluate the merged op to get all summaries from the returned "summary" variable.
    # Feed metadata variable to session for visualizing the graph in TensorBoard.
    _, summary, loss_val = session.run(
        [optimizer, merged, loss],
        feed_dict=feed_dict,
        run_metadata=run_metadata)
    average_loss += loss_val

    # Add returned summaries to writer in each step.
    writer.add_summary(summary, step)
    # Add metadata to visualize the graph for the last run.
    if step == (num_steps - 1):
      writer.add_run_metadata(run_metadata, 'step%d' % step)

    if step % 2000 == 0:
      if step > 0:
        average_loss /= 2000
      # The average loss is an estimate of the loss over the last 2000 batches.
      print('Average loss at step ', step, ': ', average_loss)
      average_loss = 0

    # Note that this is expensive (~20% slowdown if computed every 10000 steps)
    if step % 10000 == 0:
      sim = similarity.eval()
      for i in xrange(valid_size):
        valid_word = reverse_dictionary[valid_examples[i]]
        top_k = 8  # number of nearest neighbors
        nearest = (-sim[i, :]).argsort()[1:top_k + 1]
        log_str = 'Nearest to %s:' % valid_word
        for k in xrange(top_k):
          close_word = reverse_dictionary[nearest[k]]
          log_str = '%s %s,' % (log_str, close_word)
        print(log_str)
  final_embeddings = normalized_embeddings.eval()

  ################################################################
  #
  # This section computes the distance matrix across vocabulary
  #
  ################################################################

  all_similarity_val = all_similarity.eval()
  print(all_similarity_val.shape)
  word_word_idxs = np.zeros(shape=(vocabulary_size,vocabulary_size), dtype='int32')
  word_word_similarities = np.zeros(shape=(vocabulary_size, vocabulary_size), dtype='float32')
  for rowIdx in xrange(vocabulary_size):
    dist_row = all_similarity_val[rowIdx]
    word_word_idxs[rowIdx,] = dist_row
    word_word_similarities[rowIdx,] = dist_row

  #######################################################
  #
  # This section compares the documents with each other
  #
  #######################################################

  max_doc_size = 0
  for docIdx in range(num_docs):
    doc_name = doc_names[docIdx]
    doc1_widxs = doc_words[doc_name]
    if (len(doc1_widxs) > max_doc_size):
      max_doc_size = len(doc1_widxs)

  #outfile = open('dist.txt', 'w')
  for docIdx in range(num_docs):
    d1_d2_dist_array_3d.fill(0)
    doc_name = doc_names[docIdx]
    doc1_widxs = doc_words[doc_name]
    for nextIdx in range(num_docs):
      next_name = doc_names[nextIdx]
      doc2_widxs = doc_words[next_name]

      rows = len(doc1_widxs)
      cols = len(doc2_widxs)
     
      for ridx in range(rows):
        d1_d2_row = np.zeros(shape=(max_doc_size), dtype='float32')
        doc1_word_to_vocabulary_values = word_word_similarities[doc1_widxs[ridx],]
        d1_d2_row[0:cols] = doc1_word_to_vocabulary_values[doc2_widxs]
        doc1_word_to_all_doc2_words_dists = d1_d2_row
        d1_d2_dist_array_3d[nextIdx,ridx,] = doc1_word_to_all_doc2_words_dists

    d1_d2_similarities = doc_to_doc_similarities.eval(feed_dict={distance_matrix: d1_d2_dist_array_3d})
    doc_similarities[docIdx,] = d1_d2_similarities


  nearest_doc_idxs = np.zeros((num_docs, num_docs), dtype='int32')
  for rowIdx in xrange(num_docs):
    similarity_row = doc_similarities[rowIdx]
    sorted_negative_similarity_row_idxs = (-similarity_row).argsort()
    nearest_doc_idxs[rowIdx,] = sorted_negative_similarity_row_idxs

  ###########################################################################
  #
  # This section outputs the top 10 documents that are closest to each one
  #
  ###########################################################################

  outfile = open('doc_comparisons.txt', 'w')
  for docIdx in range(num_docs):
    doc_name = doc_names[docIdx]
    nearest_list = []
    for i in range(1,11):
      nearest_doc_idx = nearest_doc_idxs[docIdx,i]
      tmp_name = doc_names[nearest_doc_idx]
      tmp_similarity = doc_similarities[docIdx,nearest_doc_idx]
      nearest_list.append((tmp_name, tmp_similarity))
    outfile.write(doc_name + " => " + str(nearest_list) + "\n")
  outfile.close()

  # Write corresponding labels for the embeddings.
  #with open(FLAGS.log_dir + '/metadata.tsv', 'w') as f:
  #  for i in xrange(vocabulary_size):
  #    f.write(reverse_dictionary[i] + '\n')

  # Save the model for checkpoints.
  #saver.save(session, os.path.join(FLAGS.log_dir, 'model.ckpt'))

  # Create a configuration for visualizing embeddings with the labels in TensorBoard.
  config = projector.ProjectorConfig()
  embedding_conf = config.embeddings.add()
  embedding_conf.tensor_name = embeddings.name
  embedding_conf.metadata_path = os.path.join(FLAGS.log_dir, 'metadata.tsv')
  projector.visualize_embeddings(writer, config)

writer.close()

#################################################
#
# This section Visualize the embeddings
#
#################################################
# Function to draw visualization of distance between embeddings.
def plot_with_labels(low_dim_embs, labels, filename):
  assert low_dim_embs.shape[0] >= len(labels), 'More labels than embeddings'
  plt.figure(figsize=(18, 18))  # in inches
  for i, label in enumerate(labels):
    x, y = low_dim_embs[i, :]
    plt.scatter(x, y)
    plt.annotate(
        label,
        xy=(x, y),
        xytext=(5, 2),
        textcoords='offset points',
        ha='right',
        va='bottom')

  plt.savefig(filename)
  plt.show()


try:
  from sklearn.manifold import TSNE
  import matplotlib.pyplot as plt

  tsne = TSNE(
      perplexity=30, n_components=2, init='pca', n_iter=5000, method='exact')
  plot_only = 500
  low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only, :])
  labels = [reverse_dictionary[i] for i in xrange(plot_only)]
  plot_with_labels(low_dim_embs, labels, os.path.join(gettempdir(), 'tsne.png'))

except ImportError as ex:
  print('Please install sklearn, matplotlib, and scipy to show embeddings.')
  print(ex)
