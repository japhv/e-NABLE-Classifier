"""
    Encoder-Decoder Model to classify e-NABLE Facebook posts

    author: Japheth Adhavan
"""

import tensorflow as tf
from tensorflow.python.layers import core as layers_core

batch_size = 1

num_units = 25

max_gradient_norm = 1

learning_rate = 0.02

max_encoder_time = 1000 # Shape of the input strings
max_decoder_time = 8 # List of classes


with tf.variable_scope("embedding") as scope:
    # Load glove embeddings matrix
    embedding_encoder = tf.Variable(getEmbeddings())

    # Load the embeddings for the encoder sentence
    encoder_inputs = tf.placeholder(tf.int32, shape=[max_encoder_time, batch_size], name="encoderInput")
    encoder_emb_inp = tf.nn.embedding_lookup(embedding_encoder, encoder_inputs)

    # Load the embeddings for the decoder sentence
    decoder_inputs = tf.placeholder(tf.int32, shape=[max_decoder_time, batch_size], name="decoderInput")
    decoder_emb_inp = tf.nn.embedding_lookup(embedding_encoder, decoder_inputs)

    decoder_outputs = tf.placeholder(tf.int32, name="decoderOutput")

# Dense layer
with tf.variable_scope("dense") as denseScope:
    projection_layer = layers_core.Dense(getVocabSize(), use_bias=False) #TODO: change when we add jobsEmbedding matrix


# Encoder
with tf.variable_scope("encoder") as encoderScope:
    # Build RNN cell
    encoder_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units)

    # This is unnecessary but can be added as a parameter for the dynamic_rnn for verifyability
    #source_sequence_length = tf.Variable(batch_size,dtype=tf.int32)

    # defining initial state
    #encoder_outputs is a tensor with the shape (max_encoder_time, batchSize, num_units)
    #encoder_state is a LSTMStateTuple
    encoder_outputs, encoder_state = tf.nn.dynamic_rnn(cell=encoder_cell, inputs=encoder_emb_inp, time_major=True, dtype=tf.float64)

# Decoder
with tf.variable_scope("decoder") as decoderScope:
    decoder_lengths = tf.placeholder(tf.int32, shape=[1])  # The sequence length -  An int32 vector tensor.

    ##Build RNN cell
    decoder_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units)

    ##Helper
    helper = tf.contrib.seq2seq.TrainingHelper(decoder_emb_inp, decoder_lengths, time_major=True)

    ##Decoder
    decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell, helper, encoder_state, output_layer=projection_layer)

    ##Dynamic decoding
    outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder, impute_finished=True, maximum_iterations=3)
    logits = outputs.rnn_output  # (batchSize, output length (should be 2), vocabulary size aka glove embedding (should be 1193514))

# Calculating loss
with tf.variable_scope("loss") as lossScope:
    crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=decoder_outputs, logits=logits)#Returns a tensor with shape of decoder_outputs

    target_weights = tf.placeholder(tf.float64)#, shape=[1])
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
    print(sess.run(tf.global_variables_initializer()))

    c = 0
    while c < epochs:
        for r in range(getTotalLines()-1):
            enpInp, decInp, declen, decOut, targetWeight = loadNextTweet()
            currentLoss = sess.run(train_loss,feed_dict={encoder_inputs:enpInp, decoder_inputs:decInp, decoder_lengths:declen, decoder_outputs:decOut, target_weights:targetWeight })
            print(currentLoss)
            sess.run(update_step,feed_dict={encoder_inputs:enpInp, decoder_inputs:decInp, decoder_lengths:declen, decoder_outputs:decOut, target_weights:targetWeight })

            if currentLoss < 2:
                learning_rate = 0.0001
            c = c + 1

    #Test
    print("Model trained!")