"""
    Encoder-Decoder Model to classify e-NABLE Facebook posts

    author: Japheth Adhavan
"""

import tensorflow as tf
import numpy as np

from tensorflow.python.layers import core as layers_core
from preprocess import getTrainTest

batch_size = 1

num_units = 8

max_gradient_norm = 1

learning_rate = 0.02

epochs = 2

train, test = getTrainTest()


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
    outputs, _, fs = tf.contrib.seq2seq.dynamic_decode(decoder, impute_finished=True)

    logits = outputs.rnn_output

# Calculating loss
with tf.variable_scope("loss") as lossScope:

    decoder_outputs = tf.placeholder(tf.int32, name="decoderOutput")

    # Returns a tensor with shape of decoder_outputs
    crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=decoder_outputs, logits=logits)

    target_weights = tf.placeholder(tf.float64)
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

    for _ in range(epochs):
        for idx, row in train.iterrows():

            if not row["content"]:
                continue

            enpInp = row["x_term"]
            decInp = np.expand_dims(np.vstack(row["y_term"]), axis=1)
            enpInpLen = len(row["x_term"])

            feed_dict = {
                encoder_inputs: enpInp,
                decoder_inputs: decInp,
                decoder_outputs: decInp,
                target_weights: np.ones(8)
            }
            currentLoss = sess.run(train_loss, feed_dict=feed_dict)

            # Prints first 50 characters of the content with loss
            print(row["content"][:50], " - Loss:", currentLoss)

            sess.run(update_step, feed_dict=feed_dict)

            if currentLoss < 2:
                learning_rate = 0.0001

    #Test
    print("Model trained!")

