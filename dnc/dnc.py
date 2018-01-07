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
import sonnet as snt
from .memory import Memory, EPSILON


class DNC(snt.RNNCore):
    """DNC recurrent module that connects together the controller and memory.

    Performs a write and read operation against memory given 1) the previous state
    and 2) an interface vector defining how to interact with the memory at the
    current time step.

    Args:
        controller_config (dict): required fields: `hidden_size`
        memory_config (dict): required fields:  `words_num`, `word_size`, `read_heads_num`
        output_size (int): size of output dimension for the DNC module at each time step
        classic_dnc_output (bool): whether to concat the outputs of the contoller and memory
            strictly according to the originial formulation in the paper
        clip_value (int): threshold to use in clipping activations
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

    def __init__(self, controller_config, memory_config, output_size, classic_dnc_output=False,
                 clip_value=20, name="dnc"):
        super().__init__(name=name)

        self._output_size = output_size
        self._R = memory_config["read_heads_num"]
        self._W = memory_config["word_size"]
        self._interface_vector_size = self._R * self._W + 3 * self._W + 5 * self._R + 3
        self._clip_value = clip_value
        self._classic_dnc_output = classic_dnc_output

        with self._enter_variable_scope():
            self._controller = snt.LSTM(**controller_config, cell_clip_value=clip_value)
            self._memory = Memory(**memory_config)
            self._controller_to_interface_weights = snt.Linear(
                self._interface_vector_size,
                name='controller_to_interface'
            )
            if not self._classic_dnc_output:
                self._controller_to_output_weights = snt.Linear(
                    self._output_size,
                    name="controller_to_output"
                )
                self._memory_to_output_weights = snt.Linear(
                    self._output_size,
                    name="memory_to_output"
                )

    def _parse_interface_vector(self, interface_vector):
        r = self._R
        w = self._W

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
            ("read_modes", lambda v: tf.nn.softmax(tf.reshape(v, (-1, 3, r)), dim=1)),
        ])
        indices = [[sum(sizes[:i]), sum(sizes[:i + 1])] for i in range(len(sizes))]
        zipped_items = zip(fns.keys(), fns.values(), indices)
        interface = {name: fn(interface_vector[:, i[0]:i[1]]) for name, fn, i in zipped_items}

        return DNC.interface(**interface)

    def _flatten_read_vectors(self, x):
        return tf.reshape(x, (-1, self._W * self._R))

    def _build(self, inputs, prev_memory_state):
        with tf.name_scope("concat_inputs"):
            read_vectors_flat = self._flatten_read_vectors(prev_memory_state.read_vectors)
            input_augmented = tf.concat([inputs, read_vectors_flat], 1)

        controller_output, controller_state = self._controller(
            input_augmented,
            prev_memory_state.controller_state,
        )
        interface = self._controller_to_interface_weights(controller_output)

        with tf.name_scope("parse_interface"):
            interface = self._parse_interface_vector(interface)

        read_vectors, memory_state = self._memory(interface, prev_memory_state.memory_state)

        with tf.name_scope("join_outputs"):
            read_vectors_flat = self._flatten_read_vectors(read_vectors)

            if self._classic_dnc_output:
                read_vectors_flat = self._flatten_read_vectors(read_vectors)
                final_output = tf.concat([controller_output, read_vectors_flat], 1)
            else:
                memory_result = self._memory_to_output_weights(read_vectors_flat)
                controller_result = self._controller_to_output_weights(controller_output)
                final_output = memory_result + controller_result

        dnc_state = DNC.state(
            memory_state=memory_state,
            controller_state=controller_state,
            read_vectors=read_vectors,
        )

        return final_output, dnc_state

    @property
    def state_size(self):
        return DNC.state(
            memory_state=self._memory.state_size,
            controller_state=self._controller.state_size,
            read_vectors=tf.TensorShape([self._W, self._R]),
        )

    def initial_state(self, batch_size, name=None):
        return DNC.state(
            memory_state=self._memory.initial_state(batch_size),
            controller_state=self._controller.initial_state(batch_size),
            read_vectors=tf.fill([batch_size, self._W, self._R], EPSILON),
        )

    @property
    def output_size(self):
        return self._output_size
