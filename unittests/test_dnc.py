# -*- coding: utf-8 -*-
# pylint: disable=W0212
import numpy as np
import tensorflow as tf
from tensorflow.python.training import rmsprop

from dnc.dnc import DNC


class TestDNC(tf.test.TestCase):

    def test_constructor(self):
        memory_config = {
            'memory_size': 4,
            'word_size': 5,
            'num_read_heads': 2,
        }
        dnc = DNC(10, controller_units=64, **memory_config)
        input_size = 17
        test_input = np.random.uniform(-3, 3, (2, dnc._W *
                                               dnc._R + input_size)).astype(np.float32)
        initial_state = dnc.get_initial_state(batch_size=2)
        _, _ = dnc(test_input, initial_state)
        self.assertEqual(dnc._interface_vector_size, 38)
        self.assertEqual(dnc.output_size, 10)
        self.assertEqual(dnc.get_config()["name"], "DNC")

    def test_dnc_output_shape(self):
        batch_size = 3
        memory_config = {
            'memory_size': 16,
            'word_size': 5,
            'num_read_heads': 7,
        }
        output_size = 10

        for input_size in [10, 17, 49]:
            dnc = DNC(output_size, controller_units=64, **memory_config)
            initial_state = dnc.get_initial_state(batch_size=batch_size)
            input_shape = dnc._W * dnc._R + input_size
            test_input = np.random.uniform(-3, 3,
                                           (batch_size, input_shape)).astype(np.float32)
            example_output, _ = dnc(
                tf.convert_to_tensor(test_input),
                initial_state,
            )
            self.assertEqual(example_output.shape, (batch_size, output_size))

    def test_eager_dnc_optimization(self):
        batch_size = 7
        input_size = 15
        memory_config = {
            'memory_size': 27,
            'word_size': 9,
            'num_read_heads': 10,
        }
        output_size = 36

        x = tf.keras.Input(shape=(None, input_size, ))
        dnc_cell = DNC(output_size, controller_units=30, **memory_config)
        dnc_initial_state = dnc_cell.get_initial_state(batch_size=batch_size)
        layer = tf.keras.layers.RNN(dnc_cell)
        y = layer(x, initial_state=dnc_initial_state)

        model = tf.keras.models.Model(x, y)
        model.compile(optimizer=rmsprop.RMSPropOptimizer(learning_rate=0.001),
                      loss='mse', run_eagerly=True)
        model.train_on_batch(
            np.zeros((batch_size, 5, input_size)), np.zeros((batch_size, output_size)))
        self.assertEqual(model.output_shape[1], output_size)

    def test_parse_interface_vector(self):
        output_size = 10
        batch_size = 2
        memory_config = {
            'memory_size': None,
            'word_size': 5,
            'num_read_heads': 2,
        }
        interface_vector_size = 38
        interface = np.random.uniform(-3, 3, (batch_size, interface_vector_size))
        interface = interface.astype(np.float32)

        def softmax_dim1(x):
            y = np.atleast_2d(x)
            y = y - np.expand_dims(np.max(y, axis=1), 1)
            y = np.exp(y)
            y_summed = np.expand_dims(np.sum(y, axis=1), 1)
            return y / y_summed

        expected_interface = {
            "read_keys": np.reshape(interface[:, :10], (-1, 5, 2)),
            "read_strengths": 1 + np.log(np.exp(np.reshape(interface[:, 10:12], (-1, 2, ))) + 1),
            "write_key": np.reshape(interface[:, 12:17], (-1, 5, 1)),
            "write_strength": 1 + np.log(np.exp(np.reshape(interface[:, 17], (-1, 1))) + 1),
            "erase_vector": 1.0 / (1 + np.exp(-1 * np.reshape(interface[:, 18:23], (-1, 5)))),
            "write_vector": np.reshape(interface[:, 23:28], (-1, 5)),
            "free_gates": 1.0 / (1 + np.exp(-1 * np.reshape(interface[:, 28:30], (-1, 2)))),
            "allocation_gate": 1.0 / (1 + np.exp(-1 * interface[:, 30, np.newaxis])),
            "write_gate": 1.0 / (1 + np.exp(-1 * interface[:, 31, np.newaxis])),
            "read_modes": softmax_dim1(np.reshape(interface[:, 32:], (-1, 3, 2))),
        }

        dnc = DNC(output_size, controller_units=64, **memory_config)
        parsed_interface = dnc._parse_interface_vector(interface)._asdict()

        for item in expected_interface:
            with self.subTest(name=item):
                self.assertAllClose(
                    parsed_interface[item],
                    expected_interface[item],
                )


if __name__ == '__main__':
    tf.test.main()
