# -*- coding: utf-8 -*-
# pylint: disable=W0221
"""
Differentiable Neural Computer model definition.

Reference:
    http://www.nature.com/nature/journal/vaop/ncurrent/full/nature20101.html

Conventions:
    B - batch size
    N - number of slots in memory
    R - number of read heads
    W - size of each memory slot i.e word size
"""

from collections import namedtuple, OrderedDict
import tensorflow as tf
from tensorflow.python.util import nest

from .memory import Memory, EPSILON


class DNC(tf.keras.layers.Layer):
    """DNC recurrent module that connects together the controller and memory.

    Performs a write and read operation against memory given 1) the previous state
    and 2) an interface vector defining how to interact with the memory at the
    current time step.

    Args:
        output_size (int): size of final output dimension for the whole DNC cell at each time step
        controller_units (int): size of hidden state in controller
        memory_size (int): number of slots in external memory
        word_size (int): the width of each memory slot
        num_read_heads (int): number of memory read heads
    """

    state = namedtuple("dnc_state", [
        "memory_state",
        "controller_state",
        "read_vectors",
    ])

    interface = namedtuple("interface", [
        "read_keys",
        "read_strengths",
        "write_key",
        "write_strength",
        "erase_vector",
        "write_vector",
        "free_gates",
        "allocation_gate",
        "write_gate",
        "read_modes",
    ])

    def __init__(self, output_size, controller_units=256, memory_size=256,
                 word_size=64, num_read_heads=4, **kwargs):
        super().__init__(name='DNC', **kwargs)

        self._output_size = output_size
        self._N = memory_size
        self._R = num_read_heads
        self._W = word_size
        self._interface_vector_size = self._R * self._W + 3 * self._W + 5 * self._R + 3
        self._clip = 20.0

        self._controller = tf.keras.layers.LSTMCell(units=controller_units)
        self._controller_to_interface_dense = tf.keras.layers.Dense(
            self._interface_vector_size,
            name='controller_to_interface'
        )
        self._memory = Memory(memory_size, word_size, num_read_heads)
        self._final_output_dense = tf.keras.layers.Dense(self._output_size)

    def _parse_interface_vector(self, interface_vector):
        r, w = self._R, self._W

        sizes = [r * w, r, w, 1, w, w, r, 1, 1, 3 * r]
        fns = OrderedDict([
            ("read_keys", lambda v: tf.reshape(v, (-1, w, r))),
            ("read_strengths", lambda v: 1 + tf.nn.softplus((tf.reshape(v, (-1, r))))),
            ("write_key", lambda v: tf.reshape(v, (-1, w, 1))),
            ("write_strength", lambda v: 1 + tf.nn.softplus((tf.reshape(v, (-1, 1))))),
            ("erase_vector", lambda v: tf.nn.sigmoid(tf.reshape(v, (-1, w)))),
            ("write_vector", lambda v: tf.reshape(v, (-1, w))),
            ("free_gates", lambda v: tf.nn.sigmoid(tf.reshape(v, (-1, r)))),
            ("allocation_gate", lambda v: tf.nn.sigmoid(tf.reshape(v, (-1, 1)))),
            ("write_gate", lambda v: tf.nn.sigmoid(tf.reshape(v, (-1, 1)))),
            ("read_modes", lambda v: tf.nn.softmax(tf.reshape(v, (-1, 3, r)), axis=1)),
        ])
        indices = [[sum(sizes[:i]), sum(sizes[:i + 1])] for i in range(len(sizes))]
        zipped_items = zip(fns.keys(), fns.values(), indices)
        interface = {name: fn(interface_vector[:, i[0]:i[1]]) for name, fn, i in zipped_items}

        return DNC.interface(**interface)

    def _flatten_read_vectors(self, x):
        return tf.reshape(x, (-1, self._W * self._R))

    def call(self, inputs, prev_dnc_state):
        prev_dnc_state = nest.pack_sequence_as(self.state_size_nested, prev_dnc_state)
        with tf.name_scope("inputs_to_controller"):
            read_vectors_flat = self._flatten_read_vectors(prev_dnc_state.read_vectors)
            input_augmented = tf.concat([inputs, read_vectors_flat], 1)
            controller_output, controller_state = self._controller(
                input_augmented,
                prev_dnc_state.controller_state,
            )
            controller_output = tf.clip_by_value(controller_output, -self._clip, self._clip)

        with tf.name_scope("parse_interface"):
            interface = self._controller_to_interface_dense(controller_output)
            interface = self._parse_interface_vector(interface)

        with tf.name_scope("update_memory"):
            read_vectors, memory_state = self._memory(interface, prev_dnc_state.memory_state)
            state = DNC.state(
                memory_state=memory_state,
                controller_state=controller_state,
                read_vectors=read_vectors,
            )

        with tf.name_scope("join_outputs"):
            read_vectors_flat = self._flatten_read_vectors(read_vectors)
            final_output = tf.concat([controller_output, read_vectors_flat], 1)
            final_output = self._final_output_dense(final_output)
            final_output = tf.clip_by_value(final_output, -self._clip, self._clip)

        return final_output, nest.flatten(state)

    @property
    def state_size_nested(self):
        return DNC.state(
            memory_state=self._memory.state_size,
            controller_state=self._controller.state_size,
            read_vectors=tf.TensorShape([self._W, self._R]),
        )

    @property
    def state_size(self):
        return nest.flatten(self.state_size_nested)

    def get_initial_state(self, inputs=None, batch_size=None, dtype=tf.float32):
        del inputs
        initial_state_nested = DNC.state(
            memory_state=self._memory.get_initial_state(batch_size, dtype=dtype),
            controller_state=self._controller.get_initial_state(batch_size=batch_size, dtype=dtype),
            read_vectors=tf.fill([batch_size, self._W, self._R], EPSILON),
        )
        return nest.flatten(initial_state_nested)

    @property
    def output_size(self):
        return self._output_size

    def get_config(self):
        config = {
            'controller': {
                'class_name': self._controller.__class__.__name__,
                'config': self._controller.get_config()
            },
            'memory': {
                'read_heads_num': self._R,
                'word_size': self._W,
                'words_num': self._N,
            },
            'clip': self._clip,
            'output_size': self._output_size,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
