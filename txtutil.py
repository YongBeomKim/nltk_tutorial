# 텍스트를 줄단위로 끊어서 불러온뒤
# Token 단위로, 한글명사들을 추출한다
def txtnoun(filename , skip=False, tags=['Noun'], stem=True, tokens=False):

    try:
        # konlpy 0.4.4 이하인 경우
        from konlpy.tag import Twitter
        twitter   = Twitter()
    except:
        # konlpy 0.4.5 이상인 경우
        from konlpy.tag import Okt
        twitter   = Okt()

    import re

    with open(filename, 'r', encoding='utf-8') as f:
        contents = f.readlines()

    # ['보고서 개요\n',
    #  '삼성전자는 경제·사회·환경적 가치창출의 통합적인 성과를 다양한 이해관계자에게\n',
    #  '투명하게 소통하고자 매년 지속가능경영보고서를 발간하고 있으며,\n',
    #  '2018년 열한 번째 지속가능경영보고서를 발간합니다.\n',
    #  '보고기간\n',

    result = []
    for content in contents:
        texts     = content.replace('\n', '') # 해당줄의 줄바꿈 내용 제거
        tokenizer = re.compile(r'[^ ㄱ-힣]+')   # 한글과 띄어쓰기를 제외한 모든 글자를 선택
        tokens    = tokenizer.sub('', texts)   # 한글과 띄어쓰기를 제외한 모든 부분을 제거
        tokens    = tokens.split(' ')
        sentence  = []

        for token in tokens:
            # skip 대상이 없을 떄
            if skip == False:
                # temp = twitter.nouns(token)

                # twitter 기준별 태그 분석객체 생성
                chk_tok = twitter.pos(token, stem=stem)
                chk_tok = [temp[0]  for temp in chk_tok   if temp[1] in tags]
                ckeck = "".join(chk_tok)

                if len(ckeck) > 1:
                    sentence.append(ckeck)

            # skip 내용이 있을 때
            else:
                if token.strip() in skip.keys():
                    result.append(skip[token.strip()])
                else:
                    # twitter 기준별 태그 분석객체 생성
                    chk_tok = twitter.pos(token, stem=stem)
                    chk_tok = [temp[0] for temp in chk_tok if temp[1] in tags]
                    ckeck = "".join(chk_tok)

                    # 전처리가 끝난 결과가 skip에 해당여부 판단
                    if ckeck.strip() in skip.keys():
                        result.append(skip[ckeck.strip()])
                    elif len(ckeck) > 1:
                        sentence.append(ckeck)

        # 단락별 작업이 끝난 뒤 '\n'를 덧붙여서 작업을 종료
        temp = "".join(sentence)
        if len(temp) > 1:
            sentence = " ".join(sentence)
            sentence += "\n"
            result.append(sentence)

    return " ".join(result)



# 네이버 뉴스 댓글을 자동으로 수집합니다
def Naver_news_rep(url, flat = True):

    from bs4 import BeautifulSoup
    import requests, re, sys, pprint
    oid  = url.split("oid=")[1].split("&")[0]
    aid  = url.split("aid=")[1]
    List, page = [], 1          # 댓글을 수집할 빈 리스트
    header = {
        "User-agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/65.0.3325.181 Safari/537.36",
        "referer":url}
    while True :
        c_url="https://apis.naver.com/commentBox/cbox/web_neo_list_jsonp.json?"+\
        "ticket=news&templateId=default_society&pool=cbox5&_callback="+\
        "jQuery1707138182064460843_1523512042464&lang=ko&country="+\
        "&objectId=news"+oid+"%2C"+aid+\
        "&categoryId=&pageSize=20&indexSize=10&groupId=&listType="+\
        "OBJECT&pageType=more&page="+str(page)+"&refresh=false&sort=FAVORITE"

        # 파싱단계
        r          = requests.get(c_url,headers=header)
        cont       = BeautifulSoup(r.content,"html.parser")
        total_comm = str(cont).split('comment":')[1].split(",")[0]
        match      = re.findall('"contents":([^\*]*),"userIdNo"', str(cont))

        # 댓글을 리스트에 중첩합니다.
        List.append(match)
        # 한번에 댓글이 20개씩 보이기 때문에 한 페이지씩 몽땅 댓글을 긁어 옵니다.
        if int(total_comm) <= ((page) * 20): break
        else :  page+=1

    # flat == True : 위에서 수집한 20개씩의 댓글목록을 1개의 List로 변환
    if flat == True:
        flatList = []
        for elem in List:
            if type(elem) == list:
                for e in elem: flatList.append(e)
            else: flatList.append(elem)
        return flatList

    else:
        return List


# tf-idf 데이터 값들을 Rank 값으로 변환
def table_rank(series):
    rank, result  = {}, {}
    # 순번 데이터 추출하기
    set_values = list(set(series.values))
    set_values.sort(reverse=False)
    for no, i in enumerate(set_values):
        rank[i] = no + 1
    # 원본 데이터에 순번을 적용
    for k, v in series.items():
        result[k] = rank[v]
    import pandas as pd
    result = pd.Series(result)
    return result.sort_values(ascending=False)



# noun_token = []
# for token in tokens:
#     if token in ['갤러시', '가치창출']:
#         noun_token.append(token)
#     else:
#         token_pos = twitter.pos(token)
#         temp      = [txt_tag[0]   for txt_tag in token_pos
#                                   if txt_tag[1] == 'Noun']
#         if len("".join(temp)) > 1:
#             noun_token.append("".join(temp))

# texts = " ".join(noun_token)
# texts[:100]



# https://gist.github.com/himzzz/4105717
# tf-idf 사용자 함수
import re
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk import bigrams, trigrams
import math

# 단어별 빈도수 계산
def freq(word, doc):
    return doc.count(word)

def word_count(doc):
    return len(doc)

def tf(word, doc):
    return (freq(word, doc) / float(word_count(doc)))

def num_docs_containing(word, list_of_docs):
    count = 0
    for document in list_of_docs:
        if freq(word, document) > 0:
            count += 1
    return 1 + count

def idf(word, list_of_docs):
    return math.log(len(list_of_docs) /
            float(num_docs_containing(word, list_of_docs)))

# doc : 가운데가 많아질수록 tfidf값이 커진다
# doc : 분석대상 Text
# list_of_docs : 분석 기준이 되는 token들의 모음
def tf_idf(word, doc, list_of_docs):
    return (tf(word, doc) * idf(word, list_of_docs))