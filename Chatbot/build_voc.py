import tensorflow as tf
import numpy as np
import config
from  Conversation import Conversation

# --twitter 사용시 이 함수 주석을 전부 풀어주고 아래는 주석처리
#from konlpy.tag import Okt
#twitter = Okt()
#def Make_voc(data_path,voc_path):
#    words = []
#    with open(data_path, 'r', encoding='utf-8') as content_file:
#        for con in content_file:
#            content = content_file.read()
#            words.extend(twitter.morphs(content))
#        words = list(set(words))
#
#    with open(voc_path, 'w') as vocab_file:
#        for w in words:
#            vocab_file.write(w + '\n')


# 대화문에서 Punctuation 을 제외한 단어 Token을 추출한다
# 대화문에 사용할 one hot encoding 의 Token을 생성한다
def Make_voc(conv, data_path, voc_path):
    words = []
    with open(data_path, 'r') as f:
        lines = f.read()
        words = conv.Tokenizer(lines)
        words = list(set(words))

    with open(voc_path,'w') as wf:
         for w in words:
            wf.write(w + "\n")
    print("vocabluary's Expliction complete")


def main():
    conv = Conversation()
    Make_voc(conv,config.DATA_PATH,config.VOC_PATH)

    # Twitter 모듈 사용시 위를 주석후 이를 활성화
    # Make_voc(config.DATA_PATH, config.VOC_PATH)


if __name__=="__main__":
    main()