# coding=utf-8
import tensorflow as tf
import numpy as np

vocab_size = 246873
embedding_dim = 128
filter_sizes = "3, 4, 5"
num_filters = 128
num_classes = 183


def predict_law(fact):
    vec_file = open("dict/dictionary.txt", 'r', encoding='utf-8')
    vec_dict = {}
    vec_lines = vec_file.readlines()
    num = 0
    for word in vec_lines:
        word = word.strip()
        vec_dict[word] = num
        num += 1

    words_in_fact = fact.split(' ')
    fact_vec = []
    for index in range(200):
        if index < len(words_in_fact):
            if words_in_fact[index] in vec_dict.keys():
                fact_vec.append(vec_dict[words_in_fact[index]])
            else:
                fact_vec.append(0)
        else:
            fact_vec.append(0)
    # print(fact_vec)
    return predict(np.array(fact_vec))


def predict(x_dev):

    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=False)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            learning_rate = 1e-3
            cnn = TextCNN(
                sequence_length=200,
                num_classes=num_classes,
                vocab_size=vocab_size,
                embedding_size=embedding_dim,
                filter_sizes=list(map(int, filter_sizes.split(","))),
                num_filters=num_filters,
                l2_reg_lambda=0.0)
            # Define Training procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(learning_rate)

            # minimize loss
            grads_and_vars = optimizer.compute_gradients(cnn.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            save_file = "model/model-4300"
            saver.restore(sess, save_file)

            def predict_step(x_batch, y_batch):
                # print(x_batch)
                feed_dict = {
                    cnn.input_x: x_batch,
                    cnn.input_y: y_batch,
                    cnn.dropout_keep_prob: 1.0
                }
                step, pred, scores = sess.run(
                    [global_step, cnn.predictions, cnn.scores],
                    feed_dict)

                a = max(scores[0])
                b = min(scores[0])

                standardization = [(i - b) / (a - b) for i in scores[0]]
                result = []
                for i in standardization:
                    if i > 0.98:
                        result.append(standardization.index(i))

                return result

            return predict_step([x_dev], [0])


class TextCNN(object):
    def __init__(
      self, sequence_length, num_classes, vocab_size,
      embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0):

        self.input_x = tf.placeholder(tf.int64, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.int64, (None, ), name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        l2_loss = tf.constant(0.0)

        with tf.name_scope("embedding"):
            self.W = tf.Variable(
                tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                name="W")
            self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                conv = tf.nn.conv2d(
                    self.embedded_chars_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)

        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[num_filters_total, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        with tf.name_scope("loss"):
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, self.input_y)
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")