#-*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import re, codecs, config

# from konlpy.tag import Okt
# twitter = Okt()


class Conversation():

    voc_list, voc_dict, voc_size, conversation = [], [], [], []
    index_in_epoch = 0

    def Tokens_to_index(self, token):
        index = []
        for word in token:
            if word in self.voc_dict: index.append(self.voc_dict[word])
            else:                     index.append(config.UNK_ID)
        return index


    def Index_to_token(self,index):
        token = [token.append(self.voc_list[word])  for word in index]
        return token


    def pad(self, seq, max_len, start=None, eos=None):
        if start: padded_seq = [config.GO_ID] + seq
        elif eos: padded_seq = seq + [config.EOS_ID]
        else: padded_seq = seq

        if len(padded_seq) < max_len:
            return padded_seq+([config.PAD_ID]*(max_len-len(padded_seq)))
        else:
            return padded_seq


    def pad_left(self, seq, max_len):
        if len(seq) < max_len:
            return ([config.PAD_ID]*(max_len-len(seq)))+seq
        else:
            return seq


    def transform(self, input, output, input_max, output_max):
        enc_input = self.pad(input, input_max)
        dec_input = self.pad(output, output_max, start=True)
        target    = self.pad(output, output_max, eos=True)

        enc_input.reverse()
        enc_input = np.eye(self.voc_size)[enc_input]
        dec_input = np.eye(self.voc_size)[dec_input]
        return enc_input, dec_input, target


    def max_len(self, batch_set):
        max_len_input, max_len_output = 0, 0
        for i in range(0, len(batch_set), 2):
            len_input, len_output = len(batch_set[i]), len(batch_set[i+1])
            if len_input > max_len_input:   max_len_input  = len_input
            if len_output > max_len_output: max_len_output = len_output
        return max_len_input, max_len_output + 1


    def decode(self, index,string=False):
        token=[ [self.voc_list[i] for i in k]   for k in index]
        if string: return self.decode_to_string(token[0])
        else:      return token


    def decode_to_string(self,token):
        txt = ' '.join(token)
        return txt.strip()


    def cut_eos(self, indices):
        eos_idx = indices.index(config.EOS_ID)
        return indices[:eos_idx]


    def is_eos(self, voc_id):
        return voc_id == config.EOS_ID


    def is_defined(self, voc_id):
        return voc_id in config.ALL


    def Load_conversation(self,data_path):
        self.conversation = []
        with open(data_path, 'r', encoding = 'utf-8') as f:
            for line in f:
                # token=twitter.morphs(line.strip())
                # twitter로 token 생성시 아래를 주석 후, 위를 풀어서 실행
                tokens = self.Tokenizer(line.strip())
                index  = self.Tokens_to_index(tokens)
                self.conversation.append(index)


    def Load_voc(self, voc_path):
        front         = config.ALL + []
        self.voc_list = front

        with open(voc_path,'r', encoding='utf-8') as f:
            for word in f:
                self.voc_list.append(word.strip())

        self.voc_dict = { n : i    for i, n in enumerate(self.voc_list) }
        self.voc_size = len(self.voc_list)


    def Tokenizer(self, sentence):
        token = []

        for word in sentence.strip().split():
            token.extend(re.compile("([.,!?\"':;)(])").split(word))

        ret   = [t     for t in token    if t]
        return ret


    def next_batch(self, batch_size):
        enc_input, dec_input, dec_target = [], [], []
        start = self.index_in_epoch

        if self.index_in_epoch + batch_size < len(self.conversation) - 1:
            self.index_in_epoch = self.index_in_epoch + batch_size
        else: self.index_in_epoch = 0

        batch_set = self.conversation[start : start + batch_size]
        batch_set = batch_set+ batch_set[1:] + batch_set[0 : 1]
        max_len_input, max_len_output = self.max_len(batch_set)

        for i in range(0, len(batch_set) - 1, 2):
            enc, dec, tar = self.transform(
                batch_set[i], batch_set[i + 1], max_len_input, max_len_output)
            enc_input.append(enc)
            dec_input.append(dec)
            dec_target.append(tar)

        return enc_input, dec_input, dec_target
