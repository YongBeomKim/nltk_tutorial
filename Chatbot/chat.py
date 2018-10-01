#-*- coding: utf-8 -*-
import os, math, sys
#os.environ["CUDA_VISIBLE_DEVICES"]="4"

from model import sequence2sequence
import config
from Conversation import Conversation

import tensorflow as tf
import numpy as np
# from konlpy.tag import Okt
# twitter = Okt()


class Chat:

    def __init__(self, VOC_PATH, train_dir):
        self.conv  = Conversation()
        self.conv.Load_voc(VOC_PATH)
        self.model = sequence2sequence(self.conv.voc_size)

        self.sess  = tf.Session()
        ckpt       = tf.train.get_checkpoint_state(train_dir)
        self.model.saver.restore(self.sess,ckpt.model_checkpoint_path)


    def decode(self, enc_input, dec_input):
        if type(dec_input) is np.ndarray:
            dec_input = dec_input.tolist()
        input_len = int(math.ceil((len(enc_input)+1) * 1.5))
        enc_input, dec_input, _ = self.conv.transform(
                enc_input, dec_input, input_len, config.max_decode_len)

        return self.model.predict(self.sess, [enc_input], [dec_input])


    def run(self):
        sys.stdout.write(" >> ")
        sys.stdout.flush()
        line = sys.stdin.readline()

        while line:
            print(self.response(line.strip()))
            sys.stdout.write("\n >> ")
            sys.stdout.flush()
            line = sys.stdin.readline()


    def response(self, ipt):

        # enc_input=twitter.morphs(ipt)
        # twitter 사용시 위는 풀고, 아래를 주석처리.
        enc_input = self.conv.Tokenizer(ipt)
        enc_input = self.conv.Tokens_to_index(enc_input)

        cur, dec_input=0, []
        for i in range(config.max_decode_len):
            outputs = self.decode(enc_input, dec_input)
            if self.conv.is_eos(outputs[0][cur]): break
            elif self.conv.is_defined(outputs[0][cur]) is not True:
                dec_input.append(outputs[0][cur])
                cur += 1

        reply = self.conv.decode([dec_input], True)
        return reply


def main():
    chatbot = Chat(config.VOC_PATH, config.train_dir)
    chatbot.run()

if __name__=="__main__":
    main()