"""
    Encoder-Decoder Model to classify e-NABLE Facebook posts

    author: Japheth Adhavan
"""
import sys

import tensorflow as tf
from tensorflow.python.layers import core as layers_core

from preprocess import loadGloVe, loadData
import util

if len(sys.argv) < 2 or sys.argv[1] not in {"train", "test"}:
    print("Usage: python main.py [train|test]")
    sys.exit(0)
else:
    is_train = sys.argv[1] == "train"

nlp = loadGloVe()

batch_size = 1

num_units = 10

max_gradient_norm = 1

learning_rate = 0.02

epochs = 2

with tf.variable_scope("dense") as denseScope:
    projection_layer = layers_core.Dense(10, activation=tf.sigmoid, use_bias=True) # 6400 is a number greater than the no of unique vocabulary

# Encoder
with tf.variable_scope("encoder") as encoderScope:
    encoder_inputs = tf.placeholder(dtype=tf.float64, shape=[None, 1, 300])
    # Build RNN cell
    encoder_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units)

    encoder_outputs, encoder_state = tf.nn.dynamic_rnn(cell=encoder_cell, inputs=encoder_inputs, time_major=True, dtype=tf.float64)

# Decoder
with tf.variable_scope("decoder") as decoderScope:

    decoder_lengths = tf.constant(1, shape=[1])  # The sequence length -  An int32 vector tensor.

    decoder_inputs = tf.placeholder(tf.float64, shape=(1, 1, 10), name="decoderInput")

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

    # Returns a tensor with shape of decoder_outputs
    train_loss = tf.losses.mean_squared_error(
                labels=decoder_outputs,
                predictions=logits,
                weights=1.0,
                scope=None,
                loss_collection=tf.GraphKeys.LOSSES,
                reduction=tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS
            )


# Calculate and clip gradients
with tf.variable_scope("gradient") as gradScope:
    params = tf.trainable_variables()
    gradients = tf.gradients(train_loss, params)
    clipped_gradients, _ = tf.clip_by_global_norm(gradients, max_gradient_norm)


# Optimization
with tf.variable_scope("optimizer") as optimScope:
    optimizer = tf.train.AdamOptimizer(learning_rate)
    update_step = optimizer.apply_gradients(zip(clipped_gradients, params))


# Add ops to save and restore all the variables.
saver = tf.train.Saver()


def get_feed_dict(row_data):
    start_token = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    end_token = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    labels = [
        0,
        row_data["Report"],
        row_data["Device"],
        row_data["Delivery"],
        row_data["Progress"],
        row_data["becoming_member"],
        row_data["attempt_action"],
        row_data["Activity"],
        row_data["Other"],
        0
    ]
    x_term = [[token.vector] for token in nlp(row_data["content"])]
    y_inp_term = [[start_token]]
    y_out_term = [[labels]]

    feed_dict = {
        encoder_inputs: x_term,
        decoder_inputs: y_inp_term,
        decoder_outputs: y_out_term
    }

    return feed_dict, labels[1:-1]


def train_model():
    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())

        train = loadData("./data/split_data/train.csv")

        for _ in range(epochs):
            for idx, row in train.iterrows():

                if not row["content"]:
                    continue

                feed_dict, labels = get_feed_dict(row)

                predicted, currentLoss = sess.run([logits, train_loss], feed_dict=feed_dict)

                # Prints first 50 characters of the content with loss
                print(row["content"][:50], " - Loss:", currentLoss)

                sess.run(update_step, feed_dict=feed_dict)

        print("\nModel trained!")

        save_path = saver.save(sess, "./model_dir/model1/model.ckpt")
        print("Model saved in path: %s" % save_path)


def test_model(test):
    with tf.Session() as sess:

        saver.restore(sess, "./model_dir/model1/model.ckpt")
        print("Model restored.")

        predictions = []
        labels = []

        for idx, row in test.iterrows():

            if not row["content"]:
                continue

            feed_dict, label = get_feed_dict(row)

            predicted_logits = sess.run(logits, feed_dict=feed_dict)

            predicted = util.normalize_predictions(predicted_logits[0][0][1:-1])

            print(row["content"][:50], "\n","Actual:", label, "\nPredicted:", predicted_logits, "\n")

            predictions.append(predicted)
            labels.append(label)

    util.print_summary(labels, predictions)


if is_train:
    train_model()
    cv_test = loadData("./data/split_data/validate.csv")
    test_model(cv_test)
else:
    test = loadData("./data/split_data/test.csv")
    test_model(test)














