# 파이썬을 활용한 자연어 분석 기초 


**강의 슬라이드 [SlideShare](https://www.slideshare.net/YBkim2/ss-117904577)**


## 강의목차

1일차 - 1th hour - **자연어 분석기초, Token과 정규식, Stemming/Tagging**<br>
1일차 - 2th hour - **Konlpy, WordCloud(실습)**<br>
1일차 - 3th hour - **StopWord, tf-idf, N-Gram(PMI)**<br>
  
2일차 - 1th hour - **Naive Bayse Classification(실습)**<br>
2일차 - 2th hour - **Word2Vec(실습)**<br>
2일차 - 3th hour - **LSTM(실습), Seq2Seq(실습)**<br>


## NLTK 설치가 안되는 경우

<figure class="align-left">
  <img src="https://i.stack.imgur.com/0F2k3.png" alt="">
  <figcaption></figcaption>
</figure>

과거 버젼에서 사용되었던 `nltk.download()` 를 사용하면 `CERTIFICATE_VERIFY_FAILED] certificate verify failed` 서버연결 오류로 문제가 발생합니다. 이와같은 경우에는 위와 같이 오류 메세지에서 제시하는 대로 따르면 됩니다, 이는 **각각의 Corpus 파일을 다운받는 방법**으로 문제를 해결 가능합니다 


>  pip3 install -U nltk 

> import nltk; nltk.download('punkt')

만약 위의 방법으로도 해당 Corpus 가 설치되지 않고 계속 오류메세지를 출력하는 경우에는 수동으로 각각의 함수에서 요청하는 Corpus를 다운받아 설치하면 되며, 자세한 내용은 [블로그](http://pubdata.tistory.com/154) 또는 [동영상 해결방법](https://programmers.co.kr/learn/courses/21/lessons/946) 을 참고하시면 도움이 될 것입니다




<figure class="align-left">
  <img src="https://static1.squarespace.com/static/538cea80e4b00f1fad490c1b/54668a77e4b00fb778d22a34/54668d8ae4b00fb778d2859c/1416007413413/NLTK.png" alt="">
  <figcaption>nltk 자연어 분석도구</figcaption>
</figure>


# 예제와 코드로 알아보는 파이썬 자연어 분석


## 사용한 모듈

    nltk == 3.3
    konlpy == 0.5.1
    wordcloud == 1.5.0
    scikit-learn == 0.19.2
    scipy == 1.1.0
    gensim == 3.5.0
    tensorflow == 1.9.0
    pywsd == 1.1.7



## 강의 대상

파이썬에 대한 학습 경험이 있는, 학생/ 직장인을 대상으로 **자연어 분석과정 전반에 대한** 내용을 다룹니다.


## 강의 목표

**자연어 분석**에 필요한 **용어 및 개념들**을 익히고, 간단한 사례와 소스코드 실습을 통해서 내용을 익힙니다.