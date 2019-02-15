import tensorflow as tf
from dnc.dnc import DNC
import numpy as np

np.random.seed(1)

g = tf.Graph()
with g.as_default():
    batch_size = 4
    output_size = 20
    input_size = 10

    dnc = DNC(output_size, controller_units=128, memory_size=256, word_size=64, num_read_heads=4)
    initial_state = dnc.get_initial_state(batch_size=batch_size)
    example_input = np.random.uniform(0, 1, (batch_size, input_size)).astype(np.float32)
    output_op, _ = dnc(
        tf.convert_to_tensor(example_input),
        initial_state,
    )
    init = tf.global_variables_initializer()
    with tf.Session(graph=g) as sess:
        init.run()
        example_output = sess.run(output_op)

    tf.summary.FileWriter("graphs", g).close()
