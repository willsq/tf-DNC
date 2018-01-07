import numpy as np
import tensorflow as tf
from tensorflow.python.ops import rnn
from dnc.dnc import DNC


class DNCTest(tf.test.TestCase):

    def test_constructor(self):
        controller_config = {
            "hidden_size": 64,
        }
        memory_config = {
            'read_heads_num': 2,
            'word_size': 5,
            'words_num': None,
        }
        dnc = DNC(controller_config, memory_config, 10, classic_dnc_output=False)
        self.assertEqual(dnc._interface_vector_size, 38)
        self.assertEqual(dnc._controller_to_interface_weights.output_size, 38)
        self.assertEqual(dnc._controller.output_size, tf.TensorShape([64]))
        self.assertEqual(dnc._controller_to_output_weights.output_size, 10)
        self.assertEqual(dnc._memory_to_output_weights.output_size, 10)

    def test_dnc_output_shape(self):
        batch_size = 3
        controller_config = {
            "hidden_size": 64,
        }
        memory_config = {
            'read_heads_num': 7,
            'word_size': 5,
            'words_num': 16,
        }
        output_size = 10

        for input_size in [10, 17, 49]:
            dnc = DNC(controller_config, memory_config, output_size, classic_dnc_output=False)
            initial_state = dnc.initial_state(batch_size)
            input_shape = dnc._W * dnc._R + input_size
            test_input = np.random.uniform(-3, 3, (batch_size, input_shape)).astype(np.float32)
            example_output_op, _ = dnc(
                tf.convert_to_tensor(test_input),
                initial_state,
            )
            init = tf.global_variables_initializer()

            with self.test_session() as sess:
                init.run()
                example_output = sess.run(example_output_op)
            self.assertEqual(example_output.shape, (batch_size, output_size))

    def test_dnc_optimization(self):
        batch_size = 7
        time_steps = 15
        input_size = 30
        controller_config = {
            "hidden_size": 64,
        }
        memory_config = {
            'read_heads_num': 10,
            'word_size': 9,
            'words_num': 27,
        }
        output_size = 36

        dnc = DNC(controller_config, memory_config, output_size, classic_dnc_output=False)
        dnc_initial_state = dnc.initial_state(batch_size)
        inputs = tf.random_normal([time_steps, batch_size, input_size])
        dnc_output_op, _ = rnn.dynamic_rnn(
            cell=dnc,
            inputs=inputs,
            initial_state=dnc_initial_state,
            time_major=True
        )

        targets = np.random.rand(time_steps, batch_size, output_size)
        loss = tf.reduce_mean(tf.square(dnc_output_op - targets))
        optimizier_op = tf.train.GradientDescentOptimizer(5).minimize(loss)
        init_op = tf.global_variables_initializer()

        with self.test_session():
          init_op.run()
          optimizier_op.run()

    def test_parse_interface_vector(self):
        output_size = 10
        batch_size = 2
        controller_config = {
            "hidden_size": 64,
        }
        memory_config = {
            'read_heads_num': 2,
            'word_size': 5,
            'words_num': None,
        }
        interface_vector_size = 38
        interface = np.random.uniform(-3, 3, (batch_size, interface_vector_size)).astype(np.float32)

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

        dnc = DNC(controller_config, memory_config, output_size, classic_dnc_output=False)
        parse_op = dnc._parse_interface_vector(interface)
        init_op = tf.global_variables_initializer()
        with self.test_session() as sess:
            init_op.run()
            interface = sess.run(parse_op)
            parsed_interface = interface._asdict()

        for item in expected_interface.keys():
            with self.subTest(name=item):
                self.assertAllClose(
                    parsed_interface[item],
                    expected_interface[item],
                )

    def test_final_output(self):
        output_size = 19
        batch_size = 6
        controller_config = {
            "hidden_size": 64,
        }
        memory_config = {
            'words_num': 20,
            'word_size': 5,
            'read_heads_num': 2
        }
        dnc = DNC(controller_config, memory_config, output_size, classic_dnc_output=False)
        intermediate_output = np.random.uniform(-1, 1, (batch_size, output_size)).astype(np.float32)
        new_read_vectors = np.random.uniform(0, 1, (batch_size, 5, 2)).astype(np.float32)

        memory_result = dnc._memory_to_output_weights(
            tf.convert_to_tensor(np.reshape(new_read_vectors, (-1, dnc._W * dnc._R)))
        )
        controller_result = dnc._controller_to_output_weights(
            tf.convert_to_tensor(intermediate_output)
        )
        final_result = memory_result + controller_result

        init_op = tf.global_variables_initializer()
        with self.test_session() as sess:
            init_op.run()
            output = sess.run(final_result)
        self.assertEqual(output.shape, (6, 19))


if __name__ == '__main__':
    tf.test.main()
