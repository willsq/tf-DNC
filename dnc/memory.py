# -*- coding: utf-8 -*-
"""DNC memory operations and state.

Conventions:
    B - batch size
    N - number of slots in memory
    R - number of read heads
"""

from collections import namedtuple
import tensorflow as tf

EPSILON = 1e-6


class ContentAddressing:
    """
    Access memory content using cosine similarity.

        Used for: reading, writing
    """

    @staticmethod
    def weighting(memory_matrix, keys, strengths, sharpness_op=tf.math.softplus):
        """Get content-based weighting using cosine similarity. The weighting
        for each memory slot will be high if the key points in the same
        direction as the memory contents at that slot.

        Args:
            memory_matrix (Tensor [B, N, W]): the memory matrix to query
            keys (Tensor [B, W, R]): the keys to query the memory
            strengths (Tensor [B, R]): strengths for each lookup key
            sharpness_op (fn): operation to transform strengths before softmax

        Returns:
            Tensor [B, N, R]: lookup weightings for each key
        """
        memory_normalised = tf.math.l2_normalize(memory_matrix, 2, epsilon=EPSILON)
        keys_normalised = tf.math.l2_normalize(keys, 1, epsilon=EPSILON)
        similiarity = tf.matmul(memory_normalised, keys_normalised)
        strengths = tf.expand_dims(sharpness_op(strengths), 1)

        return tf.math.softmax(similiarity * strengths, 1)


class TemporalLinkAddressing:
    """
    Access memory content by considering which interactions have happened
    recently in time.

        Used for: reading
    """

    @staticmethod
    def update_precedence_vector(prev_precedence_vector, write_weighting):
        """Return next precedence vector by taking into account the writing
        action that has just happened via `write_weighting`.

        The precedence vector at position `i` denotes the degree to which
        memory location `i` has been recently written.

        Args:
            prev_precedence_vector (Tensor [B, N]): precedence vector from time t-1
            write_weighting (Tensor [B, N]): final weighting used to write at time t

        Returns:
            Tensor [B, N]: precedence vector to use at next time step
        """
        write_strength = tf.reduce_sum(input_tensor=write_weighting, axis=1, keepdims=True)
        updated_precedence_vector = (1 - write_strength) * prev_precedence_vector + write_weighting

        return updated_precedence_vector

    @staticmethod
    def update_link_matrix(prev_link_matrix, prev_precedence_vector, write_weighting):
        """Adjust the link matrix by taking into account the writing
        action that has just happened and the previous precedence vector.

        Link matrix at `L[t,i,j]` describes the degree to which memory location
        `i` was written after location `j` between time `t` and `t+1`.

        Args:
            prev_link_matrix (Tensor [B, N, N]): link matrix from time t-1
            prev_precedence_vector (Tensor [B, N]): precedence vector from time t-1
            write_weighting (Tensor [B, N)): final weighting used to write at time t

        Returns:
            Tensor [B, N, N]: temporal link matrix to use at next time step
        """
        batch_size = prev_link_matrix.shape[0]
        words_num = prev_link_matrix.shape[1]

        write_weighting_i = tf.expand_dims(write_weighting, 2)  # [b x N x 1 ] duplicate columns
        write_weighting_j = tf.expand_dims(write_weighting, 1)  # [b x 1 X N ] duplicate rows
        prev_precedence_vector_j = tf.expand_dims(prev_precedence_vector, 1)  # [b x 1 X N]

        link_matrix = (
            (1 - write_weighting_i - write_weighting_j) * prev_link_matrix
            + (write_weighting_i * prev_precedence_vector_j)
        )
        zero_diagonal = tf.zeros([batch_size, words_num], dtype=link_matrix.dtype)

        return tf.linalg.set_diag(link_matrix, zero_diagonal)

    @staticmethod
    def weightings(link_matrix, prev_read_weightings):
        """Calculate weightings for each read head so they have a preference
        towards directionality.

        Args:
            link_matrix (Tensor [B, N, N])
            prev_read_weightings (Tensor [B, N, R]): read weightings from time t-1

        Returns:
            Tuple(Tensor [B, N, R], Tensor [B, N, R]): temporal weightings for each memory slot
        """
        forward_weighting = tf.matmul(link_matrix, prev_read_weightings)
        backward_weighting = tf.matmul(link_matrix, prev_read_weightings, adjoint_a=True)

        return forward_weighting, backward_weighting


class AllocationAdressing:
    """
    Access memory content by considering which memory slots can be allocated to.
    This is used to provide a differentiable form of dynamic memory allocation
    where slots can only be written to if they are determined to be free.

        Used for: writing
    """

    @staticmethod
    def update_usage_vector(free_gates, prev_read_weightings,
                            prev_write_weighting, prev_usage_vector):
        """Adjust the usage vector based on reads and writes from previous time
        step.

        The usage vector is a helper data structure to aid in the calculation of
        the allocation weighting. `u[t,i]` describes the usage between [0,1]
        inside memory slot `i` at time `t`. Elements of usage vector may add up
        to a maximum of `N`.

        The free gate allows reads to happen over multiple time steps at the same
        location, otherwise we would always say a location is unused immediately
        after a read has occurred.

        Args:
            free_gates (Tensor [B, R]): current free gate
            prev_read_weightings (Tensor [B, N, R]): read weightings from time t-1
            prev_write_weighting (Tensor [B, N]): write weighting from time t-1
            prev_usage_vector (Tensor [B, N]): usage vector from time t-1

        Returns:
            Tensor [B, N]: new usage vector
        """
        with tf.name_scope('allocation_addressing'):
            retention_vector = tf.reduce_prod(
                input_tensor=1 - tf.expand_dims(free_gates, 1) * prev_read_weightings,
                axis=2,
            )
            usage_vector = (
                (prev_usage_vector + prev_write_weighting
                 - (prev_usage_vector * prev_write_weighting))
                * retention_vector
            )
            return usage_vector

    @staticmethod
    def batch_unsort(tensor, indices):
        """Permute each batch in a batch first tensor according to tensor
        of indices.
        """
        unpacked = tf.unstack(indices)
        indices_inverted = tf.stack(
            [tf.math.invert_permutation(permutation) for permutation in unpacked]
        )

        unpacked = zip(tf.unstack(tensor), tf.unstack(indices_inverted))
        return tf.stack([tf.gather(value, index) for value, index in unpacked])

    @staticmethod
    def weighting(usage_vector):
        """Calculate allocation weighting so we know which memory slots are
        free to be written to. Tells us the degree to which each memory location
        is "allocable".

        Args:
            usage_vector (Tensor [B, N]): newly calculated usage vector at time t

        Returns:
            Tensor [B, N]: allocation weighting for each memory slot
        """
        usage = (1 - EPSILON) * usage_vector + EPSILON
        emptiness = 1 - usage

        words_num = usage_vector.get_shape().as_list()[1]
        emptiness_sorted, free_list = tf.nn.top_k(emptiness, k=words_num)
        usage_sorted = 1 - emptiness_sorted
        allocation_sorted = emptiness_sorted * tf.math.cumprod(usage_sorted, axis=1, exclusive=True)

        return AllocationAdressing.batch_unsort(allocation_sorted, free_list)


class Memory:
    """Differentiable memory for the DNC.

    This module implements a recurrent module interface and tracks memory state
    through time. Performs a write and read operation given the previous state
    and an interface vector defining how to interact with the memory at the
    current time step.

    Note: although this layer behaves similar to an rnn, it has no parameters
    and is actually a deterministic operation:
        (interface, prev_memory_state) -> (read_vectors, new_memory_state)

    Args:
        words_num (int): number of memory slots
        word_size (int): size of each memory slot
        read_heads_num (int): number of read heads to use inside memory
    """

    state = namedtuple(
        "memory_state", [
            'memory_matrix',
            'usage_vector',
            'link_matrix',
            'precedence_vector',
            'write_weighting',
            'read_weightings',
        ]
    )

    def __init__(self, words_num=256, word_size=64, read_heads_num=4):
        self._N = words_num
        self._W = word_size
        self._R = read_heads_num

    def __call__(self, interface, prev_memory_state):
        """Define op for the recurrent module.

        Args:
            interface (namedtuple): parsed interface vector
            prev_memory_state (namedtuple): object containing the memory plus all
                the helper data structures used to interface with the memory

        Returns:
            Tuple:
                read vectors (Tensor [N, R]): read vectors taken out of the memory
                next memory state (namedtuple): new state after write and read
        """
        with tf.name_scope("write"):
            usage, write_weighting, memory_matrix, link_matrix, precedence = Memory.write(
                prev_memory_state,
                interface,
            )

        with tf.name_scope("read"):
            read_weightings, read_vectors = Memory.read(
                memory_matrix,
                prev_memory_state.read_weightings,
                link_matrix,
                interface,
            )
        return read_vectors, Memory.state(
            memory_matrix=memory_matrix,
            usage_vector=usage,
            link_matrix=link_matrix,
            precedence_vector=precedence,
            write_weighting=write_weighting,
            read_weightings=read_weightings,
        )

    @staticmethod
    def read(memory_matrix, prev_read_weightings, link_matrix, interface):
        """Perform read on memory.

        Args:
            memory_matrix (Tensor [B, N, W]): memory matrix after recent write at time t
            prev_read_weightings (Tensor [B, N, R]): read weightings from time t-1
            link_matrix (Tensor [B, N, N]): link matrix after recent write at time t
            interface (namedtuple): parsed interface vector

        Returns:
            Tuple:
                read_weightings (Tensor [B, N, R]): read vectors taken out of the memory
                read_vectors (Tensor [B, W, R]): read vectors taken out of the memory

        """
        with tf.name_scope("content_addressing"):
            lookup_weighting = ContentAddressing.weighting(
                memory_matrix,
                interface.read_keys,
                interface.read_strengths
            )
        with tf.name_scope("temporal_link_addressing"):
            forward_weighting, backward_weighting = TemporalLinkAddressing.weightings(
                link_matrix,
                prev_read_weightings,
            )

        with tf.name_scope("blend_addressing_modes"):
            read_weightings = tf.einsum(
                "bsr,bnrs->bnr",
                interface.read_modes,
                tf.stack([backward_weighting, lookup_weighting, forward_weighting], axis=3)
            )
        read_vectors = tf.matmul(memory_matrix, read_weightings, adjoint_a=True)

        return read_weightings, read_vectors

    @staticmethod
    def write(prev_memory_state, interface):
        """Perform write on memory.

        Args:
            prev_memory_state (namedtuple): memory state from time t-1
            interface (namedtuple): parsed interface vector

        Returns:
            Tuple:
                usage_vector (Tensor [B, N])
                write_weighting (Tensor [B, N])
                memory_matrix (Tensor [B, N, W])
                link_matrix (Tensor [B, N, N])
                precedence_vector (Tensor [B, N])
        """
        m = prev_memory_state
        i = interface

        with tf.name_scope("calculate_weighting"):
            with tf.name_scope("allocation_addressing"):
                usage_vector = AllocationAdressing.update_usage_vector(
                    i.free_gates,
                    m.read_weightings,
                    m.write_weighting,
                    m.usage_vector
                )
                allocation_weighting = AllocationAdressing.weighting(usage_vector)
            with tf.name_scope("content_addressing"):
                lookup_weighting = ContentAddressing.weighting(
                    m.memory_matrix,
                    i.write_key,
                    i.write_strength
                )
            write_weighting = (
                i.write_gate * (i.allocation_gate * allocation_weighting +
                                (1 - i.allocation_gate) * tf.squeeze(lookup_weighting))
            )

        with tf.name_scope("erase_and_write"):
            erase = m.memory_matrix * (
                (1 - tf.einsum("bn,bw->bnw", write_weighting, i.erase_vector)))
            write = tf.einsum("bn,bw->bnw", write_weighting, i.write_vector)
            memory_matrix = erase + write

        with tf.name_scope("final_update"):
            link_matrix = TemporalLinkAddressing.update_link_matrix(
                m.link_matrix,
                m.precedence_vector,
                write_weighting
            )
            precedence_vector = TemporalLinkAddressing.update_precedence_vector(
                m.precedence_vector,
                write_weighting
            )

        return usage_vector, write_weighting, memory_matrix, \
            link_matrix, precedence_vector

    @property
    def state_size(self):
        return Memory.state(
            memory_matrix=tf.TensorShape([self._N, self._W]),
            usage_vector=tf.TensorShape([self._N]),
            link_matrix=tf.TensorShape([self._N, self._N]),
            precedence_vector=tf.TensorShape([self._N]),
            write_weighting=tf.TensorShape([self._N]),
            read_weightings=tf.TensorShape([self._N, self._R]),
        )

    def get_initial_state(self, batch_size, dtype=tf.float32):
        return Memory.state(
            memory_matrix=tf.fill([batch_size, self._N, self._W], EPSILON),
            usage_vector=tf.zeros([batch_size, self._N], dtype=dtype),
            link_matrix=tf.zeros([batch_size, self._N, self._N], dtype=dtype),
            precedence_vector=tf.zeros([batch_size, self._N], dtype=dtype),
            write_weighting=tf.fill([batch_size, self._N], EPSILON),
            read_weightings=tf.fill([batch_size, self._N, self._R], EPSILON),
        )
