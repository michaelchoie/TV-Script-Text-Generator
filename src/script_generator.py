"""Train an RNN and output a self-generated tv script."""

import numpy as np
import preprocess
import tensorflow as tf
from tensorflow.contrib import seq2seq


class ScriptGenerator(object):
    """
    An RNN that utilizes word2vec and LSTM cells to train model on script text.

    Attributes:
        num_epochs (int): num times data is fully processed during backprop
        batch_size (int): size of mini-batches during backprop
        rnn_size (int): size of LSTM cells
        rnn_layer_size (int) = num of stacked LSTM cells
        embed_dim (int): size of word2vec embedded layer
        seq_length (int): number of words processed at a time
        learning_rate (float): learning rate for optimization purposes
        show_every_n_batches (int): output training results every n batches
        gen_length (int): length of generated script
        save_dir (str): path to checkpoint file
        prime_word (str): word to prime the RNN model
    """

    def __init__(self):
        """Initalize object with optimization hyperparameters."""
        self.num_epochs = 1 # 50
        self.batch_size = 128
        self.rnn_size = 256 # 1024
        self.rnn_layer_size = 2
        self.embed_dim = 256 # 512
        self.seq_length = 16
        self.learning_rate = 0.001
        self.show_every_n_batches = 11
        self.gen_length = 200
        self.save_dir = './save'
        self.prime_word = 'homer_simpson'

    def _get_inputs(self):
        """
        Create TF Placeholders for input, targets, and learning rate.

        Returns:
            tuple (input, targets, learning rate)
        """
        inputs = tf.placeholder(tf.int32, [None, None], name="input")
        targets = tf.placeholder(tf.int32, [None, None], name="targets")
        learning_rate = tf.placeholder(tf.float32, name="learning_rate")

        return (inputs, targets, learning_rate)

    def _get_init_cell(self, batch_size):
        """
        Create an RNN Cell and initialize it.

        Args:
            batch_size (int): Size of batches
        Returns:
            tuple (cell, initialize state)
        """
        lstm = tf.contrib.rnn.BasicLSTMCell(self.rnn_size)
        cell = tf.contrib.rnn.MultiRNNCell([lstm] * self.rnn_layer_size)
        initial_state = cell.zero_state(batch_size, tf.float32)
        initial_state = tf.identity(initial_state, name="initial_state")

        return (cell, initial_state)

    def _get_embed(self, input_data, vocab_size, embed_dim):
        """
        Create embedding for input_data.

        Args:
            input_data (tensor): TF placeholder for text input
            vocab_size (int): Number of words in vocabulary
            embed_dim (int): Number of embedding dimensions
        Returns:
            embed (str): embedding layer
        """
        embedding = tf.Variable(tf.random_uniform((vocab_size, embed_dim),
                                                  -1, 1))
        embed = tf.nn.embedding_lookup(embedding, input_data)

        return embed

    def _build_rnn(self, cell, inputs):
        """
        Create an RNN using a RNN Cell.

        Args:
            cell (tensor): RNN Cell
            inputs (str): Input text data
        Returns:
            tuple (outputs, final_state)
        """
        outputs, final_state = tf.nn.dynamic_rnn(cell, inputs,
                                                 dtype=tf.float32)
        final_state = tf.identity(final_state, name="final_state")

        return (outputs, final_state)

    def _build_nn(self, cell, input_data, vocab_size):
        """
        Build part of the neural network.

        Args:
            cell (tensor): RNN cell
            input_data (ndarray): Input data
            vocab_size (int): Vocabulary size
        Returns:
            tuple (Logits, FinalState)
        """
        embed = self._get_embed(input_data, vocab_size, self.embed_dim)
        outputs, final_state = self._build_rnn(cell, embed)
        logits = tf.contrib.layers.fully_connected(outputs, vocab_size,
                                                   activation_fn=None)

        return (logits, final_state)

    def _get_loss_optimizer(self, logits, targets, input_data_shape, lr):
        """
        Create cost function and optimizer for backprop.

        Args:
            logits (tensor): output of neural net
            targets (tensor): desired output of neural net
            input_data_shape (tuple): dimensions of data
            lr (tensor): learning rate tensor
        Returns:
            cost (tensor)
            optimizer (op)
        """
        cost = seq2seq.sequence_loss(logits, targets,
                                     tf.ones([input_data_shape[0],
                                              input_data_shape[1]]),
                                     name="cost_fn")
        optimizer = tf.train.AdamOptimizer(lr)

        return (cost, optimizer)

    def _get_gradients(self, optimizer, cost):
        """
        Create gradients for backprop.

        Args:
            optimizer (tensor): optimizing function
            cost (tensor): cost function
        Returns:
            capped_gradients (tensor): prevent exploding gradient problem
            train_op (op): TensorFlow optimizing operation
        """
        gradients = optimizer.compute_gradients(cost)
        capped_gradients = [(tf.clip_by_value(grad, -1., 1.), var)
                            for grad, var in gradients
                            if grad is not None]
        train_op = optimizer.apply_gradients(capped_gradients,
                                             name="train_op")

        return (capped_gradients, train_op)

    def _get_bare_tensors(self, train_graph):
        """
        Get all tensors from untrained graph.

        Args:
            train_graph (Graph): Tensorflow graph loaded from memory
        Returns:
            tuple (initial_state, input_text, targets, lr, final_state,
                   train_op)
        """
        initial_state = train_graph.get_tensor_by_name("initial_state:0")
        input_text = train_graph.get_tensor_by_name("input:0")
        targets = train_graph.get_tensor_by_name("targets:0")
        lr = train_graph.get_tensor_by_name("learning_rate:0")
        # cost = train_graph.get_tensor_by_name("cost_fn:0")
        final_state = train_graph.get_tensor_by_name("final_state:0")
        train_op = train_graph.get_operation_by_name("train_op")

        return (initial_state, input_text, targets, lr, final_state,
                train_op)

    def _get_loaded_tensors(self, loaded_graph):
        """
        Get input, initial state, final state, and probabilities tensor.

        Args:
            loaded_graph (Graph): TensorFlow graph loaded from file
        Returns:
            tuple (input_tensor, initial_state, final_state, probs_tensor)
        """
        input_tensor = loaded_graph.get_tensor_by_name("input:0")
        initial_state = loaded_graph.get_tensor_by_name("initial_state:0")
        final_state = loaded_graph.get_tensor_by_name("final_state:0")
        probs_tensor = loaded_graph.get_tensor_by_name("probs:0")

        return (input_tensor, initial_state, final_state, probs_tensor)

    def _pick_word(self, probabilities, int_to_vocab):
        """
        Pick the next word in the generated text.

        Args:
            probabilities (ndarray): probabilites of the next word
            int_to_vocab (dict): keys are word ids, values are words
        Returns:
            str (most likely word to appear after sequence)
        """
        return int_to_vocab[np.random.choice(np.arange(len(probabilities)),
                                             p=probabilities)]

    def _get_batches(self, int_text):
        """
        Return batches of input and target.

        Args:
            int_text (list): TV script data represented in integers
        Returns:
            ndarray (mini-batch)
        """
        n_sequences = len(int_text) // (self.batch_size * self.seq_length)
        n_words = n_sequences * self.batch_size * self.seq_length
        inputs = np.array(int_text[:n_words])

        input_batches = np.split(inputs.reshape(self.batch_size, -1),
                                 n_sequences, axis=1)
        targets = np.array(int_text[1:n_words + 1])
        targets[-1] = int_text[0]
        target_batches = np.split(targets.reshape(self.batch_size, -1),
                                  n_sequences, axis=1)

        return np.array(list(zip(input_batches, target_batches)))

    def _pretty_print(self, epoch, batch, batches, train_loss):
        """Print results of training after a number of mini-batches."""
        if (epoch * len(batches) + batch) % self.show_every_n_batches == 0:
            print("Epoch {} Batch {}/{} train_loss = {:.3f}".format(epoch,
                                                                    batch,
                                                                    len(batches),
                                                                    train_loss))

    def _remove_tokens(self, gen_sentences, token_dict):
        """
        Remove punctuation tokens from self generated tv script.

        Args:
            gen_sentences (list): generated words for tv script
            token_dict (dict): key = punctuation, value = token
        Returns:
            tv_script(str): tv script string without tokens
        """
        tv_script = ' '.join(gen_sentences)
        token_dict = preprocess.DataPreprocessor.token_lookup()
        for key, token in token_dict.items():
            tv_script = tv_script.replace(' ' + token.lower(), key)
        tv_script = tv_script.replace('\n ', '\n')
        tv_script = tv_script.replace('( ', '(')

        return tv_script

    def _build_graph(self, int_to_vocab):
        """
        Build the computational graph.

        Args:
            int_to_vocab (dict): key = embedding layer index, key = word
        Returns:
            train_graph (Graph): populated Tensorflow Graph object
        """
        train_graph = tf.Graph()
        with train_graph.as_default():
            # Load tensors
            vocab_size = len(int_to_vocab)
            input_text, targets, lr = self._get_inputs()
            input_data_shape = tf.shape(input_text)
            cell, initial_state = self._get_init_cell(input_data_shape[0])
            logits, final_state = self._build_nn(cell, input_text, vocab_size)

            # Probabilities for generating words
            probs = tf.nn.softmax(logits, name="probs")

            # Loss function & Optimizer
            cost, optimizer = self._get_loss_optimizer(logits, targets,
                                                       input_data_shape, lr)

            # Initialize clipped gradients
            capped_gradients, train_op = self._get_gradients(optimizer, cost)

        return train_graph, cost

    def train_neural_net(self, int_to_vocab, int_text):
        """
        Train neural net and save parameters to checkpoint file.

        Args:
            train_graph (Graph): trained Tensorflow graph
            int_text (list): TV script data represented in integers
        """
        train_graph, cost = self._build_graph(int_to_vocab)
        initial_state, input_text, targets, lr, final_state, train_op = self._get_bare_tensors(train_graph)
        batches = self._get_batches(int_text)

        # Train neural network
        with tf.Session(graph=train_graph) as sess:
            sess.run(tf.global_variables_initializer())

            for epoch in range(self.num_epochs):
                state = sess.run(initial_state, {input_text: batches[0][0]})

                for batch, (x, y) in enumerate(batches):
                    feed = {input_text: x,
                            targets: y,
                            initial_state: state,
                            lr: self.learning_rate}
                    train_loss, state, _ = sess.run([cost, final_state, train_op],
                                                    feed)

                # Print results
                self._pretty_print(epoch, batch, batches, train_loss)

            # Save model
            saver = tf.train.Saver()
            saver.save(sess, self.save_dir)
            print("Model trained and saved")

    def generate_script(self, vocab_to_int, int_to_vocab, token_dict):
        """
        Load saved model and generate tv script.

        Args:
            vocab_to_int (dict): key = word, value = embedding layer index
            int_to_vocab (dict): key = embedding layer index, key = word
            token_dict (dict): key = punctuation, value = token
        """
        loaded_graph = tf.Graph()
        with tf.Session(graph=loaded_graph) as sess:
            # Load saved model
            loader = tf.train.import_meta_graph(self.save_dir + '.meta')
            loader.restore(sess, self.save_dir)

            # Get tensors from loaded model
            input_text, initial_state, final_state, probs = self._get_loaded_tensors(loaded_graph)

            # Sentences generation setup
            gen_sentences = [self.prime_word + ':']
            prev_state = sess.run(initial_state, {input_text: np.array([[1]])})

            # Generate sentences
            for n in range(self.gen_length):
                # Dynamic Input
                dyn_input = [[vocab_to_int[word] for word in gen_sentences[-self.seq_length:]]]
                dyn_seq_length = len(dyn_input[0])

                # Get Prediction
                feed = {input_text: dyn_input, initial_state: prev_state}
                probabilities, prev_state = sess.run([probs, final_state], feed)
                pred_word = self._pick_word(probabilities[dyn_seq_length - 1],
                                            int_to_vocab)
                gen_sentences.append(pred_word)

            # Remove tokens
            tv_script = self._remove_tokens(gen_sentences, token_dict)

            print(tv_script)


def main():
    """Load trained model and generate a tv script."""
    sg = ScriptGenerator()
    pp = preprocess.DataPreprocessor()
    int_text, vocab_to_int, int_to_vocab, token_dict = pp.load_preprocess()
    sg.train_neural_net(int_to_vocab, int_text)
    sg.generate_script(vocab_to_int, int_to_vocab, token_dict)


if __name__ == "__main__":
    main()
