# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Example script to train the DNC on a repeated copy task."""

import time
import os
import tensorflow as tf
import uuid

import sonnet as snt
import numpy as np
from dnc import dnc
from data import get_sample

np.random.seed(1)
tf.set_random_seed(1)
FLAGS = tf.flags.FLAGS

# Model parameters
tf.flags.DEFINE_integer("hidden_size", 256, "Size of LSTM hidden layer.")
tf.flags.DEFINE_integer("memory_size", 256, "The number of memory slots.")
tf.flags.DEFINE_integer("word_size", 64, "The width of each memory slot.")
tf.flags.DEFINE_integer("num_read_heads", 4, "Number of memory read heads.")

# Optimizer parameters.
tf.flags.DEFINE_float("max_grad_norm", 10, "Gradient clipping norm limit.")
tf.flags.DEFINE_float("learning_rate", 1e-4, "Optimizer learning rate.")
tf.flags.DEFINE_float("optimizer_epsilon", 1e-10,
                      "Epsilon used for RMSProp optimizer.")

# Task parameters
tf.flags.DEFINE_integer("batch_size", 1, "Batch size for training.")
tf.flags.DEFINE_string("data_dir", os.path.join("babi", 'data', 'en-10k'), "Path to babi data")

# Training options.
tf.flags.DEFINE_integer("num_training_iterations", 1000000, "Number of iterations to train for.")
tf.flags.DEFINE_integer("report_interval", 100, "Iterations between reports (samples, valid loss).")
tf.flags.DEFINE_string("checkpoint_dir", "/tmp/tf/dnc", "Checkpointing directory.")
tf.flags.DEFINE_integer("checkpoint_interval", 10000, "Checkpointing step interval.")

dataset = get_sample(FLAGS.batch_size, FLAGS.data_dir)

def run_model(input_sequence, output_size):
  """Runs model on input sequence."""

  memory_config = {
      "words_num": FLAGS.memory_size,
      "word_size": FLAGS.word_size,
      "read_heads_num": FLAGS.num_read_heads,
  }
  controller_config = {
      "hidden_size": FLAGS.hidden_size,
  }

  dnc_core = dnc.DNC(controller_config, memory_config, output_size, classic_dnc_output=False)
  initial_state = dnc_core.initial_state(FLAGS.batch_size)
  output_sequence, _ = tf.nn.dynamic_rnn(
      cell=dnc_core,
      inputs=input_sequence,
      time_major=True,
      initial_state=initial_state)

  return output_sequence


def train(num_training_iterations, report_interval):
  """Trains the DNC and periodically reports the loss."""

  input_data1, target_output1, output_size, target_mask1 = next(dataset)

  input_data = tf.placeholder(tf.float32, [None, 1, output_size])
  target_output = tf.placeholder(tf.float32, [None, 1, output_size])
  target_mask = tf.placeholder(tf.float32, [None, 1, 1])

  output_logits = run_model(input_data, output_size)

  train_loss = tf.reduce_mean(
    target_mask * tf.nn.softmax_cross_entropy_with_logits(logits=output_logits, labels=target_output)
  )

  # Set up optimizer with global norm clipping.
  trainable_variables = tf.trainable_variables()
  grads, _ = tf.clip_by_global_norm(
    tf.gradients(train_loss, trainable_variables),
    FLAGS.max_grad_norm
  )

  global_step = tf.get_variable(
      name="global_step",
      shape=[],
      dtype=tf.int64,
      initializer=tf.zeros_initializer(),
      trainable=False,
      collections=[tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.GLOBAL_STEP])

  learning_rate = tf.train.exponential_decay(FLAGS.learning_rate, global_step,
                                           10000, 0.99, staircase=True)

  optimizer = tf.train.RMSPropOptimizer(
      learning_rate,
      epsilon=FLAGS.optimizer_epsilon,
      momentum=0.9,
  )
  train_step = optimizer.apply_gradients(
      zip(grads, trainable_variables), global_step=global_step)

  saver = tf.train.Saver()


  if FLAGS.checkpoint_interval > 0:
    hooks = [
        tf.train.CheckpointSaverHook(
            checkpoint_dir=FLAGS.checkpoint_dir,
            save_steps=FLAGS.checkpoint_interval,
            saver=saver)
    ]
  else:
    hooks = []

  tf.summary.scalar("learning_rate", learning_rate)
  tf.summary.scalar("train_loss", train_loss)
  merged_summary_op = tf.summary.merge_all()

  # Train.
  with tf.train.SingularMonitoredSession(
      hooks=hooks, checkpoint_dir=FLAGS.checkpoint_dir) as sess:

    start_iteration = sess.run(global_step)
    total_loss = 0
    summary_writer = tf.summary.FileWriter(
        os.path.join(os.path.dirname(__file__), "logs", str(uuid.uuid4())),
        tf.get_default_graph(),
    )

    for train_iteration in range(start_iteration, num_training_iterations):
      a, b, _, d = next(dataset)
      _, loss, summary = sess.run([train_step, train_loss, merged_summary_op], feed_dict={
        input_data: a,
        target_output: b,
        target_mask: d,
      })
      total_loss += loss
      summary_writer.add_summary(summary, train_iteration)

      if (train_iteration + 1) % report_interval == 0:

        tf.logging.info("%d: Avg training loss %f.\n",
                        train_iteration, total_loss / report_interval,)
        total_loss = 0


def main(unused_argv):
  tf.logging.set_verbosity(3)  # Print INFO log messages.
  train(FLAGS.num_training_iterations, FLAGS.report_interval)


if __name__ == "__main__":
  tf.app.run()
