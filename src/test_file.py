"""Test the functions from preprocess.py and script_generator.py."""

import numpy as np
import preprocess
import script_generator
import tensorflow as tf
import unittest
from tensorflow.contrib import rnn


class TestHelper(unittest.TestCase):
    """Test all functions and verify they are functioning as intended."""

    pp = preprocess.DataPreprocessor()
    sg = script_generator.ScriptGenerator()

    def test_tokenize(self):
        """Test preprocess.py function, token_lookup()."""
        with tf.Graph().as_default():
            symbols = set(['.', ',', '"', ';', '!', '?', '(', ')', '--', '\n'])
            token_dict = self.pp.token_lookup()

            # Check type
            self.assertIsInstance(token_dict, dict)

            # Check symbols
            self.assertEqual(symbols, set(token_dict.keys()))

            # Check value types
            bad_value_table = [type(val) for val in token_dict.values()
                               if not isinstance(val, str)]
            self.assertIsNotNone(bad_value_table)

            # Check for spaces
            key_has_space = [key for key in token_dict.keys()
                             if ' ' in key]
            value_has_space = [value for value in token_dict.values()
                               if ' ' in value]
            self.assertIsNotNone(key_has_space)
            self.assertIsNotNone(value_has_space)

            # Check for symbols in values
            symbol_val = ()
            for symbol in symbols:
                for val in token_dict.values():
                    if symbol in val:
                        symbol_val = (symbol, val)

            self.assertIsNotNone(symbol_val)

    def test_create_lookup_table(self):
        """Test preprocess.py function, _create_lookup_table()."""
        with tf.Graph().as_default():
            test_text = '''
                        Moe_Szyslak Moe's Tavern Where the elite meet to drink
                        Bart_Simpson Eh yeah hello is Mike there Last name
                                     Rotch
                        Moe_Szyslak Hold on I'll check Mike Rotch Mike Rotch
                                    Hey has anybody seen Mike Rotch lately
                        Moe_Szyslak Listen you little puke One of these days
                                    I'm gonna catch you and I'm gonna carve my
                                    name on your back with an ice pick
                        Moe_Szyslak Whats the matter Homer You're not your
                                    normal effervescent self
                        Homer_Simpson I got my problems Moe Give me another one
                        Moe_Szyslak Homer hey you should not drink to forget
                                    your problems
                        Barney_Gumble Yeah you should only drink to enhance
                                      your social skills
                        '''
            test_text = test_text.lower()
            test_text = test_text.split()

            vocab_to_int, int_to_vocab = self.pp._create_lookup_tables(test_text)

            # Check types
            self.assertIsInstance(vocab_to_int, dict)
            self.assertIsInstance(int_to_vocab, dict)

            # Compare length of dicts
            self.assertEqual(len(vocab_to_int), len(int_to_vocab))

            # Check if dictionaries have same words
            vocab_to_int_set = set(vocab_to_int.keys())
            int_to_vocab_set = set(int_to_vocab.values())
            self.assertEqual(vocab_to_int_set, int_to_vocab_set)

            # Check if dictionaries have the same word ids
            vocab_to_int_idx_set = set(vocab_to_int.values())
            int_to_vocab_idx_set = set(int_to_vocab.keys())
            self.assertEqual(vocab_to_int_idx_set, int_to_vocab_idx_set)

            # Make sure the dicts make the same lookup
            missmatches = [word for word, id in vocab_to_int.items()
                           if int_to_vocab[id] != word]
            self.assertIsNotNone(missmatches)

            # Check if length of vocab is appropriate size
            self.assertTrue(len(vocab_to_int) > len(set(test_text)) / 2)

    def test_get_inputs(self):
        """Test script_generator.py function, _get_inputs()."""
        with tf.Graph().as_default():
            inputs, targets, lr = self.sg._get_inputs()

            # Check type
            self.assertEqual(inputs.op.type, "Placeholder")
            self.assertEqual(targets.op.type, "Placeholder")
            self.assertEqual(lr.op.type, "Placeholder")

            # Check name
            self.assertEqual(inputs.name, "input:0")

            # Check rank (dimensionality)
            input_rank = 0 if inputs.get_shape() == None else len(inputs.get_shape())
            targets_rank = 0 if targets.get_shape() == None else len(targets.get_shape())
            lr_rank = 0 if lr.get_shape() == None else len(lr.get_shape())

            self.assertEqual(input_rank, 2)
            self.assertEqual(targets_rank, 2)
            self.assertEqual(lr_rank, 0)

    def test_get_init_cell(self):
        """Test script_generator.py function, _get_init_cell()."""
        with tf.Graph().as_default():
            test_batch_size_ph = tf.placeholder(tf.int32, [])
            cell, init_state = self.sg._get_init_cell(test_batch_size_ph)

            # Check type
            self.assertIsInstance(cell, tf.contrib.rnn.MultiRNNCell)

            # Check name attribute
            self.assertTrue(hasattr(init_state, "name"))

            # Check name
            self.assertEqual(init_state.name, "initial_state:0")

    def test_get_batches(self):
        """Test script_generator.py function, _get_batches()."""
        with tf.Graph().as_default():
            test_int_text = list(range(1000 * self.sg.seq_length))
            batches = self.sg._get_batches(test_int_text)

            # Create index in order to generate comparable arrays
            compare_idx = batches.shape[0] * batches.shape[3]

            # Check type
            self.assertIsInstance(batches, np.ndarray)

            # Check shape
            self.assertEqual(batches.shape, (7, 2, self.sg.batch_size,
                                             self.sg.seq_length))

            # Loop through sequences, check contents of sequences
            for x in range(batches.shape[2]):
                x2 = x * compare_idx
                self.assertTrue(np.array_equal(batches[0, 0, x],
                                np.array(range(x2, x2 + batches.shape[3]))))
                self.assertTrue(np.array_equal(batches[0, 1, x],
                                np.array(range(x2 + 1, x2 + 1 + batches.shape[3]))))

            # Check last value for target = first value of input
            compare_idx2 = (batches.shape[0] - 1) * batches.shape[3] + 1
            last_seq_target = (self.sg.batch_size - 1) * compare_idx + compare_idx2
            last_seq = np.array(range(last_seq_target,
                                      last_seq_target + batches.shape[3]))
            last_seq[-1] = batches[0, 0, 0, 0]
            self.assertTrue(np.array_equal(batches[-1, 1, -1], last_seq))

    def test_get_embed(self):
        """Test script_generator.py function, _get_embed()."""
        with tf.Graph().as_default():
            embed_shape = [50, 5, 256]

            test_input_data = tf.placeholder(tf.int32, embed_shape[:2])
            test_vocab_size = 27
            test_embed_dim = embed_shape[2]

            embed = self.sg._get_embed(test_input_data, test_vocab_size,
                                       test_embed_dim)

            # Check shape
            self.assertEqual(embed.shape, embed_shape)

    def test_build_rnn(self):
        """Test script_generator.py function, _build_rnn()."""
        with tf.Graph().as_default():
            test_cell = rnn.MultiRNNCell([rnn.BasicLSTMCell(self.sg.rnn_size)
                                          for _ in range(self.sg.rnn_layer_size)])

            test_inputs = tf.placeholder(tf.float32,
                                         [None, None, self.sg.rnn_size])
            outputs, final_state = self.sg._build_rnn(test_cell,
                                                      test_inputs)

            # Check name
            self.assertTrue(hasattr(final_state, "name"))
            self.assertEqual(final_state.name, "final_state:0")

            # Check shape
            self.assertEqual(outputs.get_shape().as_list(),
                             [None, None, self.sg.rnn_size])
            self.assertEqual(final_state.get_shape().as_list(),
                             [self.sg.rnn_layer_size, 2, None,
                              self.sg.rnn_size])

    def test_build_nn(self):
        """Test script_generator.py function, _build_nn()."""
        with tf.Graph().as_default():
            test_input_data_shape = [128, 5]
            test_input_data = tf.placeholder(tf.int32, test_input_data_shape)
            test_vocab_size = 27
            test_cell = rnn.MultiRNNCell([rnn.BasicLSTMCell(self.sg.rnn_size)
                                         for _ in range(self.sg.rnn_layer_size)])

            logits, final_state = self.sg._build_nn(test_cell,
                                                    test_input_data,
                                                    test_vocab_size)

            # Check name
            self.assertTrue(hasattr(final_state, "name"))
            self.assertEqual(final_state.name, "final_state:0")

            # Check shape
            self.assertEqual(logits.get_shape().as_list(),
                             test_input_data_shape + [test_vocab_size])
            self.assertEqual(final_state.get_shape().as_list(),
                             [self.sg.rnn_layer_size, 2, None,
                              self.sg.rnn_size])

    def test_get_loaded_tensors(self):
        """Test script_generator.py function, _get_loaded_tensors()."""
        test_graph = tf.Graph()
        with test_graph.as_default():
            test_input = tf.placeholder(tf.int32, name="input")
            test_initial_state = tf.placeholder(tf.int32, name="initial_state")
            test_final_state = tf.placeholder(tf.int32, name="final_state")
            test_probs = tf.placeholder(tf.float32, name="probs")

        input_text, initial_state, final_state, probs = self.sg._get_loaded_tensors(test_graph)

        # Check tensors
        self.assertEqual(input_text, test_input)
        self.assertEqual(initial_state, test_initial_state)
        self.assertEqual(final_state, test_final_state)
        self.assertEqual(probs, test_probs)

    def test_pick_word(self):
        """Test script_generator.py function, _pick_word()."""
        test_graph = tf.Graph()
        with test_graph.as_default():
            test_probabilities = np.array([0.1, 0.8, 0.05, 0.05])
            test_int_to_vocab = {ii: word for ii, word
                                 in enumerate(['this', 'is', 'a', 'test'])}

            pred_word = self.sg._pick_word(test_probabilities,
                                           test_int_to_vocab)

            # Check type
            self.assertIsInstance(pred_word, str)

            # Check word is from vocab
            self.assertIn(pred_word, test_int_to_vocab.values())


if __name__ == "__main__":
    unittest.main()
