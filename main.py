"""
    Encoder-Decoder Model to classify e-NABLE Facebook posts

    author: Japheth Adhavan
"""

import tensorflow as tf
import numpy as np

from tensorflow.python.layers import core as layers_core
from preprocess import loadGloVe, loadData

nlp = loadGloVe()

batch_size = 1

num_units = 8

max_gradient_norm = 1

learning_rate = 0.02

epochs = 2

with tf.variable_scope("dense") as denseScope:
    projection_layer = layers_core.Dense(6400, use_bias=False) # 6400 is a number greater than the no of unique vocabulary

# Encoder
with tf.variable_scope("encoder") as encoderScope:
    encoder_inputs = tf.placeholder(dtype=tf.float64, shape=[None, 1, 300])
    # Build RNN cell
    encoder_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units)

    encoder_outputs, encoder_state = tf.nn.dynamic_rnn(cell=encoder_cell, inputs=encoder_inputs, time_major=True, dtype=tf.float64)

# Decoder
with tf.variable_scope("decoder") as decoderScope:

    decoder_lengths = tf.constant(8, shape=[1])  # The sequence length -  An int32 vector tensor.

    decoder_inputs = tf.placeholder(tf.float64, shape=(8, 1, 1), name="decoderInput")

    ## Build RNN cell
    decoder_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units)

    ## Helper
    helper = tf.contrib.seq2seq.TrainingHelper(decoder_inputs, decoder_lengths, time_major=True)

    ## Decoder
    decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell, helper, encoder_state, output_layer=projection_layer)

    ## Dynamic decoding
    outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder, impute_finished=True)

    logits = outputs.rnn_output

# Calculating loss
with tf.variable_scope("loss") as lossScope:

    decoder_outputs = tf.placeholder(tf.int32, name="decoderOutput")

    target_weights = tf.constant(1, shape=[8], dtype=tf.float64)

    # Returns a tensor with shape of decoder_outputs
    crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=decoder_outputs, logits=logits)

    train_loss = (tf.reduce_sum(crossent * target_weights) / batch_size)


# Calculate and clip gradients
with tf.variable_scope("gradient") as gradScope:
    params = tf.trainable_variables()
    gradients = tf.gradients(train_loss, params)
    clipped_gradients, _ = tf.clip_by_global_norm(gradients, max_gradient_norm)


# Optimization
with tf.variable_scope("optimizer") as optimScope:
    optimizer = tf.train.AdamOptimizer(learning_rate)
    update_step = optimizer.apply_gradients(zip(clipped_gradients, params))


with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())

    train = loadData("./data/split_data/train.csv")

    for _ in range(epochs):
        for idx, row in train.iterrows():

            if not row["content"]:
                continue

            labels = [
                        row["Report"],
                        row["Device"],
                        row["Delivery"],
                        row["Progress"],
                        row["becoming_member"],
                        row["attempt_action"],
                        row["Activity"],
                        row["Other"]
                    ]
            x_term = [[token.vector] for token in nlp(row["content"])]
            y_term = np.expand_dims(np.vstack(labels), axis=1)

            feed_dict = {
                encoder_inputs: x_term,
                decoder_inputs: y_term,
                decoder_outputs: y_term
            }

            logitsOp, currentLoss = sess.run([logits, train_loss], feed_dict=feed_dict)

            # Prints first 50 characters of the content with loss
            print(row["content"][:50], " - Loss:", currentLoss)

            # print("Actual", labels, "\nPredicted", logitsOp)

            sess.run(update_step, feed_dict=feed_dict)

            if currentLoss < 2:
                learning_rate = 0.0001

    print("\nModel trained!")

    # # Validation
    # validation = loadData("./data/split_data/validate.csv")
    #
    # accuracy = []
    #
    # for idx, row in validation.iterrows():
    #
    #     if not row["content"]:
    #         continue
    #
    #     y_term = [
    #                 row["Report"],
    #                 row["Device"],
    #                 row["Delivery"],
    #                 row["Progress"],
    #                 row["becoming_member"],
    #                 row["attempt_action"],
    #                 row["Activity"],
    #                 row["Other"]
    #             ]
    #     x_term = [[token.vector] for token in nlp(row["content"])]
    #     y_term = np.expand_dims(np.vstack(y_term), axis=1)
    #
    #     feed_dict = {
    #         encoder_inputs: x_term,
    #         decoder_inputs: y_term,
    #         decoder_outputs: y_term
    #     }
    #
    #     cross_ent, loss = sess.run([logits, train_loss], feed_dict=feed_dict)
    #     print(row["content"][:50], " - Loss:", loss, np.squeeze(cross_ent))





