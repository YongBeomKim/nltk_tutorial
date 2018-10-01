Chatbot with Sequence-to-sequence Model Based on Recurrent Neural Network
============================================================
by SangMin Lee [Git Hub](https://github.com/phantasmicmeans/ChatbotModel)


>**Development Environment**
> - Ubuntu 16.04
> - Tensorflow-gpu 1.3.0
> - python 3.6.2
    - konlpy 0.4.4
        - JPype1 0.6.2   


# Data preprocessing

1. 현재는 Token by blank

2. Konlpy twitter로 token을 나누어 학습시키려면

   - builb_voc.py에서 주석 삭제,
   - Conversation.py, chat.py에 주석을 삭제.
   - 그리고 다시 아래의 과정 1부터 진행


# Process

1. build_voc

    * data를 바꾸신다면 다시 voc를 빌드하셔야 합니다. 
    * $python build_voc.py


2. train 

    * $python train.py
 
    **Tip**  model을 다시 train 시키려면-> " cd model " -> " rm -rf * " -> python train.py


3. chat

    * $python chat.py




**Conversatin example** 

1.  하이!
    -> 안녕
2.  안녕하세요
    -> 반가워요 만나서. 안녕하세요
3.  넌 누구야?
    -> 난 금융챗봇이야!
4.  이름이 뭐야?
    -> 이름은 아직 없어!
5.  뭐하니?
    -> 아무것도 안해!
6.  독일의 수도는?
    -> 베를린 입니다.
7.  달러 환율이 어떻게 되나요?
    -> 1,093원 입니다.
8.  좋아하는 가수가 누구예요?
    -> 박효신이요.
9.  어떤 음악을 좋아하세요?
    -> 힙합 음악을 좋아해요.
10. 아이스크림 좋아하나요?
    -> 네. 엄청 좋아하죠.
11. 평일에 시간 되나요?
    -> 스케줄 체크 해 볼게요.
12. 어떤 노래 좋아해요?
    -> 발라드요
13. 배가 너무 고플땐 어떻게 하나요?
    -> 그냥 굶어요


#### 나머지 세부사항은 config에서 확인 하실 수 있습니다. ####
#### Please check the config file ####