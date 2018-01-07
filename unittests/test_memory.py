import os
import tensorflow as tf
import numpy as np
from dnc.dnc import DNC
from dnc.memory import (
    ContentAddressing,
    TemporalLinkAddressing,
    AllocationAdressing,
    Memory,
)


class DNCMemoryTests(tf.test.TestCase):

    @staticmethod
    def softmax_sample(shape, axis=1):
        """Draw from a categorical distribution."""
        x = np.random.uniform(0, 1, shape).astype(np.float32)
        return np.exp(x) / np.sum(np.exp(x), axis=axis, keepdims=True)

    def test_init_memory(self):
        memory = Memory(words_num=13, word_size=7, read_heads_num=2)
        state = memory.initial_state(9)

        self.assertEqual(state.memory_matrix.shape, (9, 13, 7))
        self.assertEqual(state.usage_vector.shape, (9, 13))
        self.assertEqual(state.link_matrix.shape, (9, 13, 13))
        self.assertEqual(state.precedence_vector.shape, (9, 13))
        self.assertEqual(state.write_weighting.shape, (9, 13))
        self.assertEqual(state.read_weightings.shape, (9, 13, 2))

    def test_einsum_equivalence(self):
        write_weighting = np.random.uniform(0, 1, (3, 9))
        write_vector = np.random.uniform(0, 1, (3, 9))
        erase_vector = np.random.uniform(0, 1, (3, 9))

        write_weighting_expanded = np.expand_dims(write_weighting, 2)
        write_vector_expanded = np.expand_dims(write_vector, 1)
        erase_vector_expanded = np.expand_dims(erase_vector, 1)

        self.assertAllClose(
            np.matmul(write_weighting_expanded, erase_vector_expanded),
            np.einsum("bn,bw->bnw", write_weighting, erase_vector)
        )
        self.assertAllClose(
            np.matmul(write_weighting_expanded, write_vector_expanded),
            np.einsum("bn,bw->bnw", write_weighting, write_vector)
        )


class TemporalLinkAddressingTest(tf.test.TestCase):

    def test_weightings(self):
        link_matrix = np.random.uniform(0, 1, (7, 9, 9)).astype(np.float32)
        read_weightings = DNCMemoryTests.softmax_sample((7, 9, 6))
        expected_forward = np.matmul(link_matrix, read_weightings)
        expected_backward = np.matmul(
            np.transpose(link_matrix, [0, 2, 1]),
            read_weightings
        )

        forward_op, backward_op = TemporalLinkAddressing.weightings(
            link_matrix,
            tf.convert_to_tensor(read_weightings)
        )
        with self.test_session() as session:
            forward_weighting = forward_op.eval()
            backward_weighting = backward_op.eval()

        self.assertAllClose(forward_weighting, expected_forward)
        self.assertAllClose(backward_weighting, expected_backward)

    def test_update_link_matrix(self):
        precedence_vector = DNCMemoryTests.softmax_sample((5, 4))
        write_weighting = DNCMemoryTests.softmax_sample((5, 4))
        prev_link_matrix = np.random.uniform(0, 1, (5, 4, 4)).astype(np.float32)
        np.fill_diagonal(prev_link_matrix[0,:], 0)
        np.fill_diagonal(prev_link_matrix[1,:], 0)

        expected_link_matrix = np.zeros((5, 4, 4)).astype(np.float32)
        off_diagonal_entries = [(x, y) for x in range(4) for y in range(4) if x != y]
        for i, j in off_diagonal_entries:
            prev_scale = (1 - write_weighting[:,i] - write_weighting[:,j])
            expected_link_matrix[:, i, j]  = prev_scale * prev_link_matrix[:, i , j] + write_weighting[:, i] * precedence_vector[:, j]

        new_link_matrix_op = TemporalLinkAddressing.update_link_matrix(
            tf.convert_to_tensor(prev_link_matrix),
            tf.convert_to_tensor(precedence_vector),
            tf.convert_to_tensor(write_weighting),
        )
        with self.test_session() as session:
            new_link_matrix = new_link_matrix_op.eval()

        self.assertAllClose(new_link_matrix, expected_link_matrix)

    def test_update_precedence_vector(self):
        prev_precedence_vector = DNCMemoryTests.softmax_sample((5, 11))
        write_weighting = DNCMemoryTests.softmax_sample((5, 11))
        write_strength = write_weighting.sum(axis=1, keepdims=True)
        expected_precedence_vector = (1 - write_strength) * prev_precedence_vector + write_weighting

        new_precedence_vector_op = TemporalLinkAddressing.update_precedence_vector(
            tf.convert_to_tensor(prev_precedence_vector),
            write_weighting
        )
        with self.test_session() as session:
            new_precedence_vector = new_precedence_vector_op.eval()

        self.assertEqual(new_precedence_vector.shape, (5, 11))
        self.assertAllClose(new_precedence_vector, expected_precedence_vector)

    def test_precedence_vector_behaviour_zero_weighting(self):
        """
        Test that we return the previous precedence vector if the weighting
        is zero everywhere.
        """
        prev_precedence_vector = DNCMemoryTests.softmax_sample((3, 9))
        write_weighting = np.full((3, 9), 0).astype(np.float32)
        new_precedence_vector_op = TemporalLinkAddressing.update_precedence_vector(
            tf.convert_to_tensor(prev_precedence_vector),
            write_weighting
        )
        with self.test_session() as session:
            new_precedence_vector = new_precedence_vector_op.eval()
        self.assertAllEqual(new_precedence_vector, prev_precedence_vector)

    def test_precedence_vector_behaviour_zero_weighting(self):
        """
        Test that we return the write weighting itself if the weighting
        sums to one.
        """
        prev_precedence_vector = DNCMemoryTests.softmax_sample((3, 9))
        write_weighting = DNCMemoryTests.softmax_sample((3, 9))
        new_precedence_vector_op = TemporalLinkAddressing.update_precedence_vector(
            tf.convert_to_tensor(prev_precedence_vector),
            write_weighting
        )
        with self.test_session() as session:
            new_precedence_vector = new_precedence_vector_op.eval()
        self.assertAllClose(new_precedence_vector, write_weighting)


class ContentAddressingTest(tf.test.TestCase):

    def test_weighting(self):
        memory_matrix = np.random.uniform(0, 1, (5, 4, 8)).astype(np.float32)
        keys = np.random.uniform(0, 1, (5, 8, 2)).astype(np.float32)
        strengths = np.random.uniform(0, 1, (5, 2)).astype(np.float32)
        sharpness_op = tf.identity

        memory_normalised = memory_matrix / np.sqrt(np.sum(memory_matrix ** 2, axis=2, keepdims=True))
        keys_normalised = keys / np.sqrt(np.sum(keys ** 2, axis=1, keepdims=True))
        similiarity = np.matmul(memory_normalised, keys_normalised)
        query = similiarity * np.expand_dims(strengths, 1)
        expected_weighting = np.exp(query) / np.sum(np.exp(query), axis=1, keepdims=True)

        weighting_op = ContentAddressing.weighting(
            tf.convert_to_tensor(memory_matrix),
            keys,
            strengths,
            sharpness_op=sharpness_op,
        )
        with self.test_session() as session:
            weighting = weighting_op.eval()

        self.assertEqual(weighting.shape, (5, 4, 2))
        self.assertAllClose(weighting, expected_weighting)


class AllocationAddressingTest(tf.test.TestCase):

    def test_update_usage_vector(self):
        free_gates = np.random.uniform(0, 1, (6, 5)).astype(np.float32)
        prev_read_weightings = DNCMemoryTests.softmax_sample((6, 3, 5))
        prev_write_weighting = DNCMemoryTests.softmax_sample((6, 3))
        prev_usage_vector = np.random.uniform(0, 1, (6, 3)).astype(np.float32)

        retention = np.product(1 - np.expand_dims(free_gates, 1) * prev_read_weightings, axis=2)
        expected_usage_vector = (prev_usage_vector + prev_write_weighting - prev_usage_vector * prev_write_weighting) * retention

        new_usage_vector_op = AllocationAdressing.update_usage_vector(
            free_gates,
            tf.convert_to_tensor(prev_read_weightings),
            tf.convert_to_tensor(prev_write_weighting),
            tf.convert_to_tensor(prev_usage_vector),
        )
        with self.test_session() as session:
            new_usage_vector = new_usage_vector_op.eval()

        self.assertEqual(new_usage_vector.shape, (6, 3))
        self.assertAllClose(new_usage_vector, expected_usage_vector)

    def test_weighting_behaviour_minmax(self):
        """
        Test that we allocate based on the inverse of the usage vector.
        """
        usage_vector = np.random.uniform(0, 1, (8, 13)).astype(np.float32)
        weighting_op = AllocationAdressing.weighting(
            tf.convert_to_tensor(usage_vector)
        )
        with self.test_session():
            weighting = weighting_op.eval()

        # we require that max usage gets the min weighting
        self.assertAllEqual(np.argmax(usage_vector, axis=1), np.argmin(weighting, axis=1))
        # we require that min usage gets the max weighting
        self.assertAllEqual(np.argmin(usage_vector, axis=1), np.argmax(weighting, axis=1))

    def test_weighting_behaviour_full_usage(self):
        """
        Test that when all slots are used up that the allocation weighting
        goes to zero everywhere.
        """
        usage_vector = np.full((5, 9), 1).astype(np.float32)
        weighting_op = AllocationAdressing.weighting(
            tf.convert_to_tensor(usage_vector)
        )
        with self.test_session():
            weighting = weighting_op.eval()

        expected_weighting = np.full((5, 9), 0).astype(np.float32)
        self.assertAllEqual(weighting, expected_weighting)

    def test_weighting_calculation(self):
        """
        Test that the vectorized implementation is correct and that
        the calculation forces the weighting to sum to one.
        """
        usage_vector = np.random.uniform(0, 1, (3, 13)).astype(np.float32)
        weighting_op = AllocationAdressing.weighting(
            tf.convert_to_tensor(usage_vector)
        )
        with self.test_session() as session:
            weighting = weighting_op.eval()

        free_list = np.argsort(usage_vector, axis=1)
        expected_weighting = np.zeros((3, 13)).astype(np.float32)
        free_list_indicies = [(x, y) for x in range(3) for y in range(13)]
        for b, j in free_list_indicies:
            prod = np.prod([usage_vector[b, free_list[b, i]] for i in range(j)])
            free_list_entry = free_list[b, j]
            expected_weighting[b, free_list_entry] = (1 - usage_vector[b, free_list_entry]) * prod

        self.assertEqual(weighting.shape, (3, 13))
        self.assertAllClose(weighting, expected_weighting)
        self.assertAllClose(np.sum(weighting, axis=1), np.ones(3), 1e-2)


class MemoryWrite(tf.test.TestCase):

    def test_construction(self):
        interface = DNC.interface(
            read_keys=None,
            read_strengths=None,
            write_key=np.random.uniform(0, 1, (3, 9, 1)).astype(np.float32),
            write_strength=np.random.uniform(0, 1, (3, 1)).astype(np.float32),
            erase_vector=tf.convert_to_tensor(np.zeros((3, 9)).astype(np.float32)),
            write_vector=tf.convert_to_tensor(np.random.uniform(0, 1, (3, 9)).astype(np.float32)),
            free_gates=np.random.uniform(0, 1, (3, 5)).astype(np.float32),
            allocation_gate=np.random.uniform(0, 1, (3, 1)).astype(np.float32),
            write_gate=np.random.uniform(0, 1, (3, 1)).astype(np.float32),
            read_modes=None,
        )

        memory = Memory(13, 9, 5)
        memory_state = memory.initial_state(3)
        write_op = memory.write(memory_state, interface)
        init_op = tf.global_variables_initializer()

        with self.test_session() as session:
            init_op.run()
            usage, write_weighting, memory, link_matrix, precedence = session.run(write_op)

        self.assertEqual(usage.shape, (3, 13))
        self.assertEqual(write_weighting.shape, (3, 13))
        self.assertEqual(memory.shape, (3, 13, 9))
        self.assertEqual(link_matrix.shape, (3, 13, 13))
        self.assertEqual(precedence.shape, (3, 13))


class MemoryRead(tf.test.TestCase):

    def test_read_vectors_and_weightings(self):
        m = Memory.state(
            memory_matrix=np.random.uniform(-1, 1, (5, 11, 7)).astype(np.float32),
            usage_vector=None,
            link_matrix=None,
            precedence_vector=None,
            write_weighting=None,
            read_weightings=DNCMemoryTests.softmax_sample((5, 11, 3), axis=1),
        )
        i = DNC.interface(
            read_keys=np.random.uniform(0, 1, (5, 7, 3)).astype(np.float32),
            read_strengths=np.random.uniform(0, 1, (5, 3)).astype(np.float32),
            write_key=None,
            write_strength=None,
            erase_vector=None,
            write_vector=None,
            free_gates=None,
            allocation_gate=None,
            write_gate=None,
            read_modes=tf.convert_to_tensor(DNCMemoryTests.softmax_sample((5, 3, 3), axis=1)),
        )
        # read uses the link matrix that is produced after a write operation
        new_link_matrix = np.random.uniform(0, 1, (5, 11, 11)).astype(np.float32)

        # assume ContentAddressing and TemporalLinkAddressing are already correct
        op_ca = ContentAddressing.weighting(m.memory_matrix, i.read_keys, i.read_strengths)
        op_f, op_b  = TemporalLinkAddressing.weightings(new_link_matrix, m.read_weightings)
        read_op = Memory.read(m.memory_matrix, m.read_weightings, new_link_matrix, i)
        with self.test_session() as session:
            lookup_weightings = session.run(op_ca)
            forward_weighting, backward_weighting = session.run([op_f, op_b])
            updated_read_weightings, updated_read_vectors = session.run(read_op)
            # hack to circumvent tf bug in not doing `convert_to_tensor` in einsum reductions correctly
            read_modes_numpy = tf.Session().run(i.read_modes)

        self.assertEqual(updated_read_weightings.shape, (5, 11, 3))
        self.assertEqual(updated_read_vectors.shape, (5, 7, 3))

        expected_read_weightings = np.zeros((5, 11, 3)).astype(np.float32)
        for read_head in range(3):
            backward_weight = read_modes_numpy[:, 0, read_head, np.newaxis] * backward_weighting[:, :, read_head]
            lookup_weight = read_modes_numpy[:, 1, read_head, np.newaxis] * lookup_weightings[:, :, read_head]
            forward_weight =  read_modes_numpy[:, 2, read_head, np.newaxis] * forward_weighting[:, :, read_head]
            expected_read_weightings[:, :, read_head] = backward_weight + lookup_weight + forward_weight
        expected_read_vectors = np.matmul(np.transpose(m.memory_matrix, [0, 2, 1]), updated_read_weightings)

        self.assertAllClose(updated_read_weightings, expected_read_weightings)
        self.assertEqual(updated_read_weightings.shape, (5, 11, 3))
        self.assertAllClose(updated_read_vectors, expected_read_vectors)


if __name__ == '__main__':
    tf.test.main()
