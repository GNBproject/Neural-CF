#만약에 Doc2Vec에서 호환성 문제 있을 경우 다운그레이드 진행
#!pip install jpype1==0.7.0
#!pip install konlpy
#!pip install --upgrade pip 

import numpy as np
import pandas as pd
from gensim.models.doc2vec import TaggedDocument
from tqdm import tqdm
from gensim.models import doc2vec
from gensim.models.doc2vec import Doc2Vec
from gensim.models.keyedvectors import WordEmbeddingsKeyedVectors
from konlpy.tag import Mecab
from datetime import datetime

#공앤박의 전체 파일을 3분할
#각 파일에는 isbn, title, 서평, 소개, toc, 카테고리, 작가, 출판사, 이미지 등의 정보가 있음
#실제로 사용할 정보는 isbn title 서평 소개 toc
gnb_1 = pd.read_csv('withnull1.csv' , sep = ';')
gnb_2 = pd.read_csv('withnull2.csv' , sep = ';')
gnb_3 = pd.read_csv('withnull3.csv' , sep = ';')

#제공 받은 파일의 isbn이 string으로 처리되어서 수정
gnb_1 = gnb_1.dropna(axis=0, subset=['isbn'])
gnb_1 = gnb_1.reset_index(drop=True)
gnb_1['isbn'] = gnb_1['isbn'].astype(int)

#분할된 파일을 하나로 합침
gnb12 = pd.concat([gnb_1, gnb_2], ignore_index = True)
gnb = pd.concat([gnb12, gnb_3], ignore_index = True)
gnb = pd.concat([gnb12, gnb_3], ignore_index = True)
gnb = gnb.dropna(axis=0, subset=['title'])
gnb = gnb.reset_index(drop=True)

#중복된 isbn 첫번째만 남기고 나머지는 삭제
duplicate = gnb[gnb.duplicated(subset='isbn')]
gnb = gnb.drop_duplicates(subset=["isbn"], keep='first')
gnb = gnb.reset_index(drop = True)

#전처리 진행
gnb['introduction'] = gnb['introduction'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","")
gnb['publisher_comment'] = gnb['publisher_comment'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","")
gnb['toc'] = gnb['toc'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","")

#소개 서평을 하나로 합침, 제목 목차를 하나로 합침
gnb["opinion"] = gnb["introduction"].fillna('') + gnb["publisher_comment"].fillna('')
gnb['info'] = gnb["title"].fillna('') + gnb["toc"].fillna('')

# ''을 결측값으로 바꾸기
gnb = gnb.replace('',np.nan)

#na 제거
gnb_opinion = gnb[gnb['opinion'].notna()]
gnb_final = gnb_opinion[gnb_opinion['info'].notna()]

#조금 깔끔해진 테이블
gnb_final = gnb_final.drop(['introduction', 'publisher_comment', 'toc'], axis=1)
gnb_final = gnb_final.reset_index(drop=True)

mecab = Mecab('C:\\mecab\\mecab-ko-dic')
#만약에 리눅스에서 할 경우
#mecab = Mecab()

#각 책의 주요 정보의 단어 딕셔너리 작업하는 시간 확인용, 보통 오래 걸리지 않음
s = datetime.now().strftime('%H:%M:%S')
print(s)

#opinion의 딕셔너리 만드는 작업
opinion_list = []

for index, row in tqdm(gnb_final.iterrows(), total=len(gnb_final)):
  text = row['opinion']
  tag = row['title']
  opinion_list.append(TaggedDocument(tags=[tag], words=mecab.morphs(text)))

print('문서의 수 :', len(opinion_list))

#info
s = datetime.now().strftime('%H:%M:%S')
print(s)

info_list = []

for index, row in tqdm(gnb_final.iterrows(), total=len(gnb_final)):
  text2 = row['info']
  tag = row['title']
  info_list.append(TaggedDocument(tags=[tag], words=mecab.morphs(text2)))

print('문서의 수 :', len(info_list))

#깡통용 임베딩
s = datetime.now().strftime('%H:%M:%S')
print(s)

model = doc2vec.Doc2Vec(vector_size=200, alpha=0.025, min_alpha=0.025, workers=8, window=8)

# Vocabulary 빌드
model.build_vocab(info_list)

# Doc2Vec 학습
model.train(info_list, total_examples=model.corpus_count, epochs=50)

# 모델 저장
model.save('gnb_doc2vec.doc2vec')
print('finish 1')

#info Doc2Vec 임베딩 진행
s = datetime.now().strftime('%H:%M:%S')
print(s)

model2 = doc2vec.Doc2Vec(vector_size=100, alpha=0.025, min_alpha=0.025, workers=8, window=8)
model2.build_vocab(info_list)
model2.train(info_list, total_examples=model2.corpus_count, epochs=50)
model2.save('info.doc2vec')

print('finish 2')

#opinion Doc2Vec 임베딩 진행
s = datetime.now().strftime('%H:%M:%S')
print(s)

model3 = doc2vec.Doc2Vec(vector_size=100, alpha=0.025, min_alpha=0.025, workers=8, window=8)
model3.build_vocab(opinion_list)
model3.train(opinion_list, total_examples=model2.corpus_count, epochs=50)
model3.save('opinion.doc2vec')

print('finish 2')

#각 Doc2Vec 모델로부터 벡터값들을 추출해서 사용할 준비
vector = np.zeros((len(gnb_final), 20)).astype(np.float32)

for i in range(len(gnb_final)):
    vector[i] = model2.docvecs[gnb_final.iloc[i]['title']]

vector2 = np.zeros((len(gnb_final), 20)).astype(np.float32)

for i in range(len(gnb_final)):
    vector2[i] = model3.docvecs[gnb_final.iloc[i]['title']]

#벡터값들을 하나로 concat
vector_merge = np.hstack((vector, vector2))

#Doc2Vec 모델 완성
for i in range(len(gnb_final)):
    slot = model.docvecs.int_index(gnb_final.iloc[i]['title'], 
                               model.docvecs.doctags,
                               model.docvecs.max_rawint)
    model.docvecs.vectors_docs[slot] = vector_merge[i]