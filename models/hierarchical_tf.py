# -*- coding: utf-8 -*-
# HierarchicalAttention: 1.Word Encoder. 2.Word Attention. 3.Sentence Encoder 4.Sentence Attention 5.linear classifier. 117-06-13
import json
import logging
import math
import pickle
import numpy as np
import tensorflow as tf
import tensorflow.contrib as tf_contrib
from gensim import corpora
import caculatescore

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)
with open("./pickles/token2id.pickle", "rb")as f:
    token2id = pickle.load(f)
print("len(token2id):", len(token2id))
print("token2id[\"UNK\"]:", token2id["UNK"])
class HierarchicalAttention:
    def __init__(self,  num_classes, learning_rate, decay_steps, decay_rate, embed_size,
                 hidden_size, vocab_size,
                 initializer=tf.random_normal_initializer(stddev=0.1), clip_gradients=5.0):
        """init all hyperparameter here"""
        self.my_tensor_0 = tf.placeholder(tf.int32)
        self.batch_size = tf.shape(self.my_tensor_0)[0]  # 动态初始化
        self.my_tensor_1 = tf.placeholder(tf.int32)
        self.sequence_length = tf.shape(self.my_tensor_1)[0]  # 动态初始化
        self.my_tensor_2 = tf.placeholder(tf.int32)
        self.num_sentences = tf.shape(self.my_tensor_2)[0]  # 动态初始化

        # with open("./pickles/word_vector_ndarray.pickle", "rb") as f:
        #     embedding = pickle.load(f)
        # self.Embedding = tf.Variable(embedding, name="Embedding")
        self.vocab_size = vocab_size
        self.num_classes = num_classes
        self.embed_size = embed_size
        self.learning_rate = tf.Variable(learning_rate, trainable=False, name="learning_rate")
        self.initializer = initializer
        self.hidden_size = hidden_size
        self.clip_gradients = clip_gradients
        self.input_x = tf.placeholder(tf.int32, [None, None, None], name="input_x")  # [None, self.num_sentences, self.sequence_length]
        self.input_y = tf.placeholder(tf.int32, [None, ], name="input_y")
        # [None, 3]
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.global_step = tf.Variable(0, trainable=False, name="Global_Step")
        self.decay_steps, self.decay_rate = decay_steps, decay_rate  # 1000 0.9
        self.instantiate_weights()
        self.logits = self.inference()  # [batch_size, self.num_classes]. main computation graph is here.

        self.predictions = tf.argmax(self.logits, axis=1, name="predictions")  # shape:[batch_size,]
        correct_prediction = tf.equal(tf.cast(self.predictions, tf.int32), self.input_y)
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="Accuracy")  # shape=()
        print("going to use single label loss.")
        self.loss_val = self.loss()
        self.train_op = self.train()


    def attention_word_level(self, hidden_state):
        """
        input:[batch_size*num_sentences,sentence_length,hidden_size*2]
        :return:representation.shape:[batch_size*num_sentences,hidden_size*2]
        """
        hidden_state_2 = tf.reshape(hidden_state, shape=[-1,
                                                          self.hidden_size * 2])
        hidden_representation = tf.nn.tanh(tf.matmul(hidden_state_2,
                                                     self.W_w_attention_word) + self.W_b_attention_word)
        hidden_representation = tf.reshape(hidden_representation, shape=[-1, self.sequence_length,
                                                                         self.hidden_size * 2])
        hidden_state_context_similiarity = tf.multiply(hidden_representation,
                                                       self.context_vecotor_word)
        attention_logits = tf.reduce_sum(hidden_state_context_similiarity,
                                         axis=2)  # [batch_size*num_sentences,sentence_length]
        attention_logits_max = tf.reduce_max(attention_logits, axis=1,
                                             keep_dims=True)  # [batch_size*num_sentences,1]
        p_attention = tf.nn.softmax(
            attention_logits - attention_logits_max)  # [batch_size*num_sentences,sentence_length]
        p_attention_expanded = tf.expand_dims(p_attention, axis=2)  # [batch_size*num_sentences,sentence_length,1]
        sentence_representation = tf.multiply(p_attention_expanded,
                                              hidden_state)  # [batch_size*num_sentences,sentence_length, hidden_size*2]
        sentence_representation = tf.reduce_sum(sentence_representation,
                                                axis=1)  # [batch_size*num_sentences, hidden_size*2]
        return sentence_representation

    def attention_sentence_level(self, hidden_state_sentence):
        """
        input: [batch_size,num_sentences,hidden_size]
        :return:representation.shape:[batch_size,hidden_size*2]
        """
        hidden_state_2 = tf.reshape(hidden_state_sentence,
                                    shape=[-1, self.hidden_size * 2])
        hidden_representation = tf.nn.tanh(tf.matmul(hidden_state_2,
                                                     self.W_w_attention_sentence) + self.W_b_attention_sentence)
        hidden_representation = tf.reshape(hidden_representation, shape=[-1, self.num_sentences,
                                                                         self.hidden_size * 2])

        hidden_state_context_similiarity = tf.multiply(hidden_representation,
                                                       self.context_vecotor_sentence)
        attention_logits = tf.reduce_sum(hidden_state_context_similiarity,
                                         axis=2)
        attention_logits_max = tf.reduce_max(attention_logits, axis=1, keep_dims=True)

        p_attention = tf.nn.softmax(attention_logits - attention_logits_max)
        p_attention_expanded = tf.expand_dims(p_attention, axis=2)
        sentence_representation = tf.multiply(p_attention_expanded,
                                              hidden_state_sentence)
        sentence_representation = tf.reduce_sum(sentence_representation, axis=1)
        return sentence_representation

    def inference(self):
        print("jin lai le")
        input_x = self.input_x
        self.embedded_words = tf.nn.embedding_lookup(self.Embedding, input_x)
        embedded_words_reshaped = tf.reshape(self.embedded_words, shape=[-1, self.sequence_length, self.embed_size])

        hidden_state_forward = self.gru_forward_word_level(embedded_words_reshaped)
        hidden_state_backward = self.gru_backward_word_level(embedded_words_reshaped)
        self.hidden_state = tf.concat([hidden_state_forward, hidden_state_backward], axis=2)

        sentence_representation = self.attention_word_level(self.hidden_state)  # output:[batch_size*num_sentences,hidden_size*2]
        sentence_representation = tf.reshape(sentence_representation, shape=[-1, self.num_sentences, self.hidden_size * 2])
        # shape:[batch_size,num_sentences,hidden_size*2]
        hidden_state_forward_sentences = self.gru_forward_sentence_level(sentence_representation)
        hidden_state_backward_sentences = self.gru_backward_sentence_level(sentence_representation)
        self.hidden_state_sentence = tf.concat([hidden_state_forward_sentences, hidden_state_backward_sentences], axis=2)
        # [batch_size, num_sentences, hidden_size*2]
        document_representation = self.attention_sentence_level(self.hidden_state_sentence)
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(document_representation, keep_prob=self.dropout_keep_prob)  # TODO
        with tf.name_scope("output"):
            logits = tf.matmul(self.h_drop, self.W_projection) + self.b_projection
        return logits

    def loss(self, l2_lambda=0.0001):
        with tf.name_scope("loss"):

            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.input_y,
                                                                    logits=self.logits);
            loss = tf.reduce_mean(losses)  # print("2.loss.loss:", loss) #shape=()

            # l2_losses = tf.add_n(
            #     [tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name]) * l2_lambda
            # loss = loss + l2_losses

        return loss

    def train(self):
        """based on the loss, use SGD to update parameter"""
        self.learning_rate = tf.train.exponential_decay(self.learning_rate, self.global_step, self.decay_steps,
                                                   self.decay_rate)
        train_op = tf_contrib.layers.optimize_loss(self.loss_val, global_step=self.global_step,
                                                   learning_rate=self.learning_rate, optimizer="Adam", clip_gradients=self.clip_gradients)
        return train_op


    def gru_forward_word_level(self, embedded_words):
        """
        :param embedded_words:[batch_size*num_sentences,sentence_length,embed_size]
        :return:[batch_size*num_sentences,sentence_length,hidden_size]
        """
        with tf.variable_scope("gru_weights_word_level_forward"):
            self.wf_cell = tf.nn.rnn_cell.GRUCell(self.hidden_size)
            init_state = self.wf_cell.zero_state(batch_size=self.batch_size*self.num_sentences, dtype=tf.float32)  # [batch_size, hidden_size]
            output, state = tf.nn.dynamic_rnn(self.wf_cell, embedded_words, initial_state=init_state, time_major=False)
        # output: [batch_size*num_sentences,sentence_length,hidden_size]
        # state:  [batch_size*num_sentences, hidden_size]


        # output_splitted = tf.split(output, self.sequence_length,
        #                                    axis=1)  # it is a list,length is sentence_length, each element is [batch_size*num_sentences,1,hidden_size]
        # output_squeeze = [tf.squeeze(x, axis=1) for x in
        #                           output_splitted]  # it is a list,length is sentence_length, each element is [batch_size*num_sentences,hidden_size]

        return output

    def gru_backward_word_level(self, embedded_words):
        """
        :param   embedded_words:[batch_size*num_sentences,sentence_length,embed_size]
        :return: [batch_size*num_sentences,sentence_length,hidden_size]
        """
        embedded_words_reverse = tf.reverse(embedded_words, [2])
        with tf.variable_scope("gru_weights_word_level_backward"):
            self.wb_cell = tf.nn.rnn_cell.GRUCell(self.hidden_size)
            init_state = self.wb_cell.zero_state(batch_size=self.batch_size*self.num_sentences, dtype=tf.float32)  # [batch_size, hidden_size]
            output, state = tf.nn.dynamic_rnn(self.wb_cell, embedded_words_reverse, initial_state=init_state, time_major=False)
        # output: [batch_size*num_sentences,sentence_length,hidden_size]
        # state:  [batch_size*num_sentences, hidden_size]

        # output_splitted = tf.split(output, self.sequence_length,
        #                            axis=1)  # it is a list,length is sentence_length, each element is [batch_size*num_sentences,1,hidden_size]
        # output_squeeze = [tf.squeeze(x, axis=1) for x in
        #                   output_splitted]  # it is a list,length is sentence_length, each element is [batch_size*num_sentences,hidden_size]
        # output_squeeze.reverse()
        output = tf.reverse(output, [2])
        return output

    def gru_forward_sentence_level(self, sentence_representation):
        """
        :param sentence_representation: [batch_size,num_sentences,hidden_size*2]
        :return:[batch_size,num_sentences,hidden_size]
        """
        with tf.variable_scope("gru_weights_sentence_level_forward"):
            self.sf_cell = tf.nn.rnn_cell.GRUCell(self.hidden_size)
            init_state = self.sf_cell.zero_state(batch_size=self.batch_size,
                                         dtype=tf.float32)  # [batch_size, hidden_size]
            output, state = tf.nn.dynamic_rnn(self.sf_cell, sentence_representation, initial_state=init_state, time_major=False)
        return output

    def gru_backward_sentence_level(self, sentence_representation):
        """
        :param sentence_representation: [batch_size,num_sentences,hidden_size*2]
        :return:[batch_size,num_sentences,hidden_size]
        """
        sentence_representation_reverse = tf.reverse(sentence_representation, [2])
        with tf.variable_scope("gru_weights_sentence_level_backward"):
            self.sb_cell = tf.nn.rnn_cell.GRUCell(self.hidden_size)
            init_state = self.sb_cell.zero_state(batch_size=self.batch_size, dtype=tf.float32)  # [batch_size, hidden_size]
            output, state = tf.nn.dynamic_rnn(self.sb_cell, sentence_representation_reverse, initial_state=init_state, time_major=False)
        # output: [batch_size, num_sentences,hidden_size]
        # state:  [batch_size, hidden_size]

        # output_splitted = tf.split(output, self.num_sentences,
        #                            axis=1)  # it is a list,length is num_sentences, each element is [batch_size,1,hidden_size]
        # output_squeeze = [tf.squeeze(x, axis=1) for x in
        #                   output_splitted]  # it is a list,length is num_sentences, each element is [batch_size,hidden_size]
        # output_squeeze.reverse()
        output = tf.reverse(output, [2])
        return output

    def instantiate_weights(self):
        """define all weights here"""
        with tf.name_scope("embedding_projection"):
            self.W_projection = tf.get_variable("W_projection", shape=[self.hidden_size * 2, self.num_classes],
                                                initializer=self.initializer)  # [embed_size,label_size]
            self.Embedding = tf.get_variable("Embedding", shape=[self.vocab_size, self.embed_size],
                                                initializer=self.initializer)
            self.b_projection = tf.get_variable("b_projection", shape=[self.num_classes])

        with tf.name_scope("attention"):
            self.W_w_attention_word = tf.get_variable("W_w_attention_word",
                                                      shape=[self.hidden_size * 2, self.hidden_size * 2],
                                                      initializer=self.initializer)
            self.W_b_attention_word = tf.get_variable("W_b_attention_word", shape=[self.hidden_size * 2])

            self.W_w_attention_sentence = tf.get_variable("W_w_attention_sentence",
                                                          shape=[self.hidden_size * 2, self.hidden_size * 2],
                                                          initializer=self.initializer)
            self.W_b_attention_sentence = tf.get_variable("W_b_attention_sentence", shape=[self.hidden_size * 2])
            self.context_vecotor_word = tf.get_variable("what_is_the_informative_word", shape=[self.hidden_size * 2],
                                                        initializer=self.initializer)
            self.context_vecotor_sentence = tf.get_variable("what_is_the_informative_sentence",
                                                            shape=[self.hidden_size * 2], initializer=self.initializer)


def sentence_padding(sentence, max_length):
    """
    :param sentence: a list
    :param num: 行向量前边补 num 个 token2id["UNK"]
    :return:
    """
    temp_list = []
    if len(sentence) <= max_length:
        for _ in range (max_length-len(sentence)):
            temp_list.append(token2id["UNK"])
        temp_list.extend(sentence)
    else:
        temp_list = sentence[max_length*(-1):]
    return temp_list


def normarlized_input(sentences, label_list, token2id, batch_size):
    max_num_sentences = 150  # while batch_size = 64
    max_sequence_length = 200
    total_train_data_size = len(sentences)
    iter_num = int(math.floor(total_train_data_size / batch_size))  # number of sentences, per batch_size
    extra_number = total_train_data_size % batch_size
    new_input_x = []
    new_label_list = []
    batch_size_list = []
    num_sentences_list = []
    sequence_length_list = []
    for iter_ in range(iter_num):
        temp_sentences = sentences[iter_*batch_size:(iter_+1)*batch_size]
        num_sentences = 0
        sequence_length = 0
        for j in range(len(temp_sentences)):
            count2 = 0
            temp_sentence = temp_sentences[j]
            count1 = list(temp_sentence).count("。")
            if num_sentences < count1:
                num_sentences = count1
            for idx in range(len(temp_sentence)):
                if temp_sentence[idx] == "。":
                    # 0 1 2 3 4 5 6 7 8
                    #     。      。   。
                    if sequence_length < idx - count2 + 1:
                        # print(temp_sentence[count2:idx])
                        sequence_length = idx - count2 + 1
                        # print(sequence_length)
                        if sequence_length >= max_sequence_length:
                            sequence_length = max_sequence_length
                            break
                    count2 = idx + 1

        if num_sentences > max_num_sentences:
            num_sentences = max_num_sentences
        num_sentences_list.append(num_sentences)
        # print("num_sentences:", num_sentences)
        sequence_length_list.append(sequence_length)
        # print("sequence_length:", sequence_length)
        raw_input_x = []
        for sentence in temp_sentences:
            temp_list_1 = []
            for idx in range(len(sentence)):
                if sentence[idx].strip() == "。":
                    temp_list_1.append(idx)
            if len(temp_list_1) > num_sentences:
                begin = temp_list_1[(num_sentences + 1) * -1] + 1
                short_sentence = sentence[begin:]
            else:
                short_sentence = sentence
            # temp_list = [token2id[k] for k in short_sentence]  # num_sentences 个句子
            temp_list = []
            for k in short_sentence:
                if k in token2id:
                    temp_list.append(token2id[k])
                else:
                    temp_list.append(token2id["UNK"])
                    # temp_list.append(-1)
            temp_list_1 = []
            for idx in range(len(short_sentence)):
                if short_sentence[idx].strip() == "。":
                    temp_list_1.append(idx)
            document = []
            for i in range(len(temp_list_1)):  # num_sentences
                if i == 0:
                    document.append(
                        sentence_padding(sentence=temp_list[:temp_list_1[i] + 1], max_length=sequence_length))
                else:
                    document.append(sentence_padding(sentence=temp_list[temp_list_1[i - 1] + 1:temp_list_1[i] + 1],
                                                     max_length=sequence_length))
            if len(document) < num_sentences:
                pad_list = [token2id["UNK"] for _ in range(sequence_length)]
                for _ in range(num_sentences - len(document)):
                    document.insert(0, pad_list)
            # len(document) = num_sentences, len(element) = sequence_length
            # document: [num_sentences, sequence_length]
            raw_input_x.append(document)
        print("raw_input_x:", np.array(raw_input_x).shape)  # [batch_size, num_sentences, sequence_length]

        new_input_x.append(raw_input_x)
        if label_list != None:
            new_label_list.append(label_list[iter_*batch_size:(iter_+1)*batch_size])
        batch_size_list.append(batch_size)
    print("iter_num:", iter_num, "len(new_input_x):", len(new_input_x))
    if extra_number != 0:
        temp_sentences = sentences[-1*extra_number:]
        num_sentences = 0
        sequence_length = 0
        for j in range(len(temp_sentences)):
            count2 = 0
            temp_sentence = temp_sentences[j]
            count1 = list(temp_sentence).count("。")
            if num_sentences < count1:
                num_sentences = count1
            for idx in range(len(temp_sentence)):
                if temp_sentence[idx] == "。":
                    # 0 1 2 3 4 5 6 7 8
                    #     。      。   。
                    if sequence_length < idx - count2 + 1:
                        # print(temp_sentence[count2:idx])
                        sequence_length = idx - count2 + 1
                        # print(sequence_length)
                        if sequence_length >= max_sequence_length:
                            sequence_length = max_sequence_length
                            break
                    count2 = idx + 1
        if num_sentences > max_num_sentences:
            num_sentences = max_num_sentences
        num_sentences_list.append(num_sentences)
        # print("num_sentences:", num_sentences)
        sequence_length_list.append(sequence_length)
        # print("sequence_length:", sequence_length)
        raw_input_x = []
        for sentence in temp_sentences:
            temp_list_1 = []
            for i in range(len(sentence)):
                if sentence[i].strip() == "。":
                    temp_list_1.append(i)
            if len(temp_list_1) > num_sentences:
                begin = temp_list_1[(num_sentences + 1) * -1] + 1
                short_sentence = sentence[begin:]
            else:
                short_sentence = sentence
            # temp_list = [token2id[k] for k in short_sentence]  # num_sentences 个句子
            temp_list = []
            for k in short_sentence:
                if k in token2id:
                    temp_list.append(token2id[k])
                else:
                    temp_list.append(token2id["UNK"])
            temp_list_1 = []
            for i in range(len(short_sentence)):
                if short_sentence[i].strip() == "。":
                    temp_list_1.append(i)
            document = []
            for i in range(len(temp_list_1)):  # num_sentences
                if i == 0:
                    document.append(
                        sentence_padding(sentence=temp_list[:temp_list_1[i] + 1], max_length=sequence_length))
                else:
                    document.append(sentence_padding(sentence=temp_list[temp_list_1[i - 1] + 1:temp_list_1[i] + 1],
                                                     max_length=sequence_length))
            if len(document) < num_sentences:
                pad_list = [token2id["UNK"] for _ in range(sequence_length)]
                for _ in range(num_sentences - len(document)):
                    document.insert(0, pad_list)
            # len(document) = num_sentences, len(element) = sequence_length
            # document: [num_sentences, sequence_length]
            raw_input_x.append(document)
        print("raw_input_x:", np.array(raw_input_x).shape)  # [batch_size, num_sentences, sequence_length]

        new_input_x.append(raw_input_x)
        if label_list != None:
            new_label_list.append(label_list[-1*extra_number:])
        batch_size_list.append(extra_number)
    print("len(new_input_x):", len(new_input_x), len(new_input_x[0]), len(new_input_x[-1]))
    # 补齐输入数据 length:ceil(total_train_data_size / batch_size)
    # [batch_size, num_sentences, sequence_length]
    return new_input_x, new_label_list, batch_size_list, num_sentences_list, sequence_length_list


def test():
    with open("./pickles/train_label_list.pickle", "rb") as f:
        label_list = pickle.load(f)

    # TODO 需要划分dev train
    num_classes = 9
    learning_rate = 0.01
    batch_size = 32
    vocab_size = len(token2id)
    print("vocab_size:", vocab_size)
    epoch_num = 100
    embed_size = 200  #  TODO 训练词向量
    total_train_data_size = len(label_list)
    print("total_train_data_size:", total_train_data_size)
    iter_num = int(math.floor(total_train_data_size / batch_size))  # number of sentences, per batch_size
    print("iter_num:", iter_num)
    decay_steps = int(math.ceil(total_train_data_size / batch_size) * 0.5)
    decay_rate = 0.96
    hidden_size = 60  # ago 50
    dropout_keep_prob = 0.5
    # sequence_length = 50


    textRNN = HierarchicalAttention(num_classes, learning_rate, decay_steps, decay_rate, embed_size,
                                    hidden_size, vocab_size)
# --------------------------------------------------------------------------
    with open("./pickles/train_seg_list.pickle", "rb") as f:
        sentences = pickle.load(f)

    train_input_x, train_input_y, train_batch_size, train_num_sentences, train_sequence_length = normarlized_input(
        sentences=sentences,
        label_list=label_list,
        batch_size=batch_size,
        token2id=token2id)
    print("writing train_data.pickle...")
    with open("./pickles/train_data.pickle", "wb") as f:
        train_data = []
        train_data.append(train_input_x)
        train_data.append(train_input_y)
        train_data.append(train_batch_size)
        train_data.append(train_num_sentences)
        train_data.append(train_sequence_length)
        pickle.dump(train_data, f)
    print("end writing train_data")

    # print("loading train_data.pickle...")
    # with open("./pickles/train_data.pickle", "rb") as f:
    #     train_data = pickle.load(f)
    #     train_input_x = train_data[0]
    #     train_input_y = train_data[1]
    #     train_batch_size = train_data[2]
    #     train_num_sentences = train_data[3]
    #     train_sequence_length = train_data[4]
    # print("end loading train_data")

# --------------------------------------------------------------------------
    with open("./pickles/dev_seg_list.pickle", "rb") as f:
        dev_data = pickle.load(f)
    with open("./pickles/dev_label_list.pickle", "rb")as f:
        dev_label = pickle.load(f)

    dev_input_x, dev_input_y, dev_batch_size, dev_num_sentences, dev_sequence_length = normarlized_input(
        sentences=dev_data,
        label_list=dev_label,
        batch_size=batch_size,
        token2id=token2id)
    print("writing dev_data.pickle...")
    with open("./pickles/dev_data.pickle", "wb") as f:
        dev_data = []
        dev_data.append(dev_input_x)
        dev_data.append(dev_input_y)
        dev_data.append(dev_batch_size)
        dev_data.append(dev_num_sentences)
        dev_data.append(dev_sequence_length)
        pickle.dump(dev_data, f)
    print("end writing dev_data")

    # print("loading dev_data.pickle...")
    # with open("./pickles/dev_data.pickle", "rb") as f:
    #     dev_data = pickle.load(f)
    #     dev_input_x = dev_data[0]
    #     dev_input_y = dev_data[1]
    #     dev_batch_size = dev_data[2]
    #     dev_num_sentences = dev_data[3]
    #     dev_sequence_length = dev_data[4]
    # print("end loading dev_data")
# -------------------------------------------------------------------------------------
    with open("./pickles/test_seg_list.pickle", "rb") as f:
        test_data = pickle.load(f)

    test_input_x, test_input_y, test_batch_size, test_num_sentences, test_sequence_length = normarlized_input\
        (sentences=test_data,
         label_list=None,
         batch_size=1,
         token2id=token2id)

    print("writing test_data.pickle...")
    with open("./pickles/test_data.pickle", "wb") as f:
        test_data = []
        test_data.append(test_input_x)
        test_data.append(test_input_y)
        test_data.append(test_batch_size)
        test_data.append(test_num_sentences)
        test_data.append(test_sequence_length)
        pickle.dump(test_data, f)

    print("end writing test_data")

    # print("loading test_data.pickle...")
    # with open("./pickles/test_data.pickle", "rb") as f:
    #     test_data = pickle.load(f)
    #     test_input_x = test_data[0]
    #     test_input_y = test_data[1]
    #     test_batch_size = test_data[2]
    #     test_num_sentences = test_data[3]
    #     test_sequence_length = test_data[4]
    # print("end loading test_data")
# ----------------------------------------------------------------------------------------------
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in tf.trainable_variables():
            print(i.name)
        for n in range(epoch_num):
            epoch_loss = 0
            epoch_acc = 0
            for i in range(len(train_input_x)):
                print(train_num_sentences[i], train_sequence_length[i])
                learning_rate, loss, acc, predict, W_projection_value, _ = sess.run(
                    [textRNN.learning_rate, textRNN.loss_val, textRNN.accuracy, textRNN.predictions, textRNN.W_projection, textRNN.train_op],
                    feed_dict={textRNN.input_x: train_input_x[i], textRNN.input_y: train_input_y[i],
                               textRNN.dropout_keep_prob: dropout_keep_prob, textRNN.batch_size: train_batch_size[i],
                               textRNN.sequence_length: train_sequence_length[i], textRNN.num_sentences: train_num_sentences[i]})
                epoch_loss = loss
                epoch_acc = acc
                print("epoch:", n + 1, "loss:", epoch_loss, "acc:", epoch_acc, "learning_rate:", learning_rate)
                # if acc == 0:
                #     print(loss)
                #     return
            print("epoch:", n + 1, "loss:", epoch_loss, "acc:", epoch_acc, "learning_rate:", learning_rate)
            dev_predict = []
            print("len(dev_input_x):",len(dev_input_x))
            for i in range(len(dev_input_x)):
                input_x = dev_input_x[i]
                predict = sess.run(
                    textRNN.predictions,
                    feed_dict={textRNN.input_x: input_x,
                               textRNN.dropout_keep_prob: dropout_keep_prob, textRNN.batch_size: dev_batch_size[i],
                               textRNN.sequence_length: dev_sequence_length[i],
                               textRNN.num_sentences: dev_num_sentences[i]})
                print("iter:", i)
                dev_predict.extend(predict.tolist())
            label_size = 8
            # print(dev_label)
            # print(dev_predict)
            print("Micro-Averaged F1: ", caculatescore.micro_avg_f1(dev_predict, dev_label, label_size))
            print("-----------")
        print("predict on test data...")
        with open("./pickles/test_id_list.pickle", "rb") as f:
            id_list = pickle.load(f)
        predict_list = []
        for i in range(len(test_input_x)):
            predict = sess.run(
            textRNN.predictions, feed_dict={textRNN.input_x: test_input_x[i],
                                             textRNN.dropout_keep_prob: 1, textRNN.batch_size: test_batch_size[i],
                                             textRNN.sequence_length: test_sequence_length[i], textRNN.num_sentences: test_num_sentences[i]})
            predict_list.extend(predict.tolist())
        print(len(predict_list))
        f = open("./result.json", 'w', encoding="utf-8")
        for i in range(len(predict_list)):
            temp_dict = {}
            temp_dict["id"] = id_list[i]
            temp_dict["penalty"] = predict_list[i]
            temp_dict["laws"] = [-1]
            in_json = json.dumps(temp_dict)
            f.write(in_json)
            f.write("\n")


        print("save model in ./models/HAN.ckpt...")
        saver = tf.train.Saver(max_to_keep=1)
        saver.save(sess, "./models/HAN.ckpt")


if __name__ == "__main__":
    test()
