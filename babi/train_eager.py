# -*- coding: utf-8 -*-
"""Train DNC on babi task."""

import logging
import os
import time
import uuid
from types import SimpleNamespace

import numpy as np
import tensorflow as tf
from absl import app, flags

from data import get_sample, get_word_space_size
from dnc import dnc

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
np.random.seed(1)
tf.enable_eager_execution()
FLAGS = flags.FLAGS

# hyperparameters taken from https://www.nature.com/articles/nature20101
PARAMS = SimpleNamespace(
    units=256,               # Size of controller LSTM hidden layer
    memory_size=256,         # The number of memory slots
    word_size=64,            # The width of each memory slot
    num_read_heads=4,        # Number of memory read heads
    max_grad_norm=10,        # Gradient clipping norm limit
    learning_rate=1e-4,      # Optimizer learning rate
    optimizer_epsilon=1e-10, # Optimizer epsilon
    batch_size=1
)

flags.DEFINE_string("data_dir", os.path.join("babi", 'data', 'en-10k'), "Path to babi data")
flags.DEFINE_integer("num_training_steps", 1000000, "Number of steps to train for")
flags.DEFINE_integer("report_interval", 100, "Steps before logging a report")
flags.DEFINE_string("checkpoint_dir", "exp", "Checkpoint directory")
flags.DEFINE_integer("checkpoint_interval", 10000, "Checkpointing interval")


def train():
    """Trains the DNC and periodically reports the loss."""

    dataset = get_sample(PARAMS.batch_size, FLAGS.data_dir)
    output_size = get_word_space_size(FLAGS.data_dir)

    # wrap DNC recurrent cell to form complete model
    x = tf.keras.Input(shape=(None, output_size, ))
    dnc_cell = dnc.DNC(
        output_size,
        controller_units=PARAMS.units,
        memory_size=PARAMS.memory_size,
        word_size=PARAMS.word_size,
        num_read_heads=PARAMS.num_read_heads
    )
    dnc_initial_state = dnc_cell.get_initial_state(batch_size=PARAMS.batch_size)
    rnn = tf.keras.layers.RNN(dnc_cell, return_sequences=True)
    y = rnn(x, initial_state=dnc_initial_state)
    model = tf.keras.models.Model(x, y)


    learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(
        PARAMS.learning_rate, 10000, 0.99, staircase=True
    )
    optimizer = tf.keras.optimizers.RMSprop(epsilon=PARAMS.optimizer_epsilon,
                                            momentum=0.9, learning_rate=learning_rate,
                                            clipnorm=PARAMS.max_grad_norm)
    os.makedirs(os.path.join(FLAGS.checkpoint_dir, 'model'), exist_ok=True)
    os.makedirs(os.path.join(FLAGS.checkpoint_dir, 'summaries'), exist_ok=True)

    step = 0
    logging.info("Starting training...")
    for step in range(FLAGS.num_training_steps):
        x, y, _, y_mask = next(dataset)
        with tf.GradientTape() as tape:
            logits = model(x)
            loss = tf.reduce_mean(
                y_mask *
                tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=y)
            )
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        step +=1 
        if step % FLAGS.report_interval == 0:
            logger.info("Loss at step {:d}: {:.6f}".format(step, loss))
        if step % FLAGS.checkpoint_interval == 0:
            model.save(os.path.join(FLAGS.checkpoint_dir, 'model', 'step{}.h5'.format(step)))

    model.save(os.path.join(FLAGS.checkpoint_dir, 'model', 'final.h5'.format(step)))

def main(unused_argv):
    del unused_argv
    train()

if __name__ == "__main__":
    app.run(main)
