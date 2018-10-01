#-*- coding:utf-8 -*-
import tensorflow as tf
import config
from Conversation import Conversation


# 대화 Text를 Seq2Seq 로 학습하는
# TensorFlow 모델을 정의한다 (Default : RNN)

class sequence2sequence:

    logits, outputs, train_op, cost = None, None, None, None
    output_keep_prob = 0.5

    def __init__(self, voc_size):

        self.voc_size = voc_size
        self.n_hidden = config.n_hidden
        self.n_layer  = config.n_layer
        self.learning_rate = config.learning_rate

        print("""self.voc_size : {}\nself.n_hidden : {}
                \nself.n_layer : {}\nself.learning_rate : {}""".format(
                    self.voc_size, self.n_hidden, self.n_layer, self.learning_rate))

        self.enc_input  = tf.placeholder(tf.float32,[None,None,self.voc_size])
        self.dec_input  = tf.placeholder(tf.float32,[None,None,self.voc_size])
        self.dec_target = tf.placeholder(tf.int64,[None,None])

        self.weights = tf.Variable(tf.ones([self.n_hidden,self.voc_size]), name = "weights")
        self.bias    = tf.Variable(tf.zeros([self.voc_size]), name = "bias")
        self.global_step = tf.Variable(0, trainable = False, name = "global_step")
        self.Make_model()
        self.saver = tf.train.Saver(tf.global_variables())


    def Make_model(self,output_keep_prob=0.5):

        self.enc_input = tf.transpose(self.enc_input, [1,0,2])
        self.dec_input = tf.transpose(self.dec_input, [1,0,2])

        with tf.variable_scope('encode'):
            enc_cell = tf.contrib.rnn.MultiRNNCell(
                [self.cell(self.n_hidden,output_keep_prob) for _ in range(self.n_layer)])
            outputs, enc_state = tf.nn.dynamic_rnn(
                enc_cell, self.enc_input, dtype = tf.float32)

        with tf.variable_scope('decode'):
            dec_cell = tf.contrib.rnn.MultiRNNCell(
                [self.cell(self.n_hidden,output_keep_prob) for _ in range(self.n_layer)])
            outputs, dec_state = tf.nn.dynamic_rnn(
                dec_cell, self.dec_input, dtype=tf.float32, initial_state=enc_state)

        self.logits, self.cost, self.train_op = self.Make_ops(outputs, self.dec_target)
        self.outputs = tf.argmax(self.logits, 2)


    def cell(self, n_hidden, output_keep_prob):

        # RNNCell을  BasicLSTMCell / LSTMCell 을 사용하여 구현 가능합니다.
        rnn_cell = tf.contrib.rnn.BasicRNNCell(self.n_hidden)
        #rnn_cell = tf.contrib.rnn.BasicLSTMCell(self.n_hidden)

        rnn_cell = tf.contrib.rnn.DropoutWrapper(
            rnn_cell, output_keep_prob = output_keep_prob)
        return rnn_cell


    def Make_ops(self,outputs,dec_target):
        time_step = tf.shape(outputs)[1]
        outputs   = tf.reshape(outputs, [-1, self.n_hidden])

        logits    = tf.matmul(outputs, self.weights) + self.bias
        logits    = tf.reshape(logits, [-1, time_step, self.voc_size])

        cost = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits = logits, labels = dec_target))
        train_op = tf.train.AdamOptimizer(
            learning_rate = self.learning_rate).minimize(cost, global_step = self.global_step)

        tf.summary.scalar('cost',cost)

        return logits, cost,train_op


    def train(self, session, enc_input, dec_input, dec_target):
        return session.run([self.train_op, self.cost],
                          feed_dict={self.enc_input  : enc_input,
                                     self.dec_input  : dec_input,
                                     self.dec_target : dec_target})


    def logs(self, session, writer, enc_input, dec_input, dec_target):
        merge   = tf.summary.merge_all()
        summary = session.run(
            merge, feed_dict = {self.enc_input  : enc_input,
                                self.dec_input  : dec_input,
                                self.dec_target : dec_target})
        writer.add_summary(summary, self.global_step.eval())


    def predict(self, session, enc_input, dec_input):
        return session.run(self.outputs,
                           feed_dict={self.enc_input: enc_input,
                                      self.dec_input: dec_input})
