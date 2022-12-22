from datetime import datetime
import pandas as pd
import numpy as np
from gensim.models.doc2vec import TaggedDocument
from tqdm import tqdm
from konlpy.tag import Mecab
from gensim.models import doc2vec

#이 데이터에는 책의 주요 정보가 담겨 있어야 함
#author, publisher, title, isbn, Cn_1, Cn_2, Cn_3, Cn_4
data = pd.read_csv('.../3.csv')

#작가, 출판사 카테고리 전처리
data['author'] = data['author'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","")
data['publisher'] = data['publisher'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","")

for i in range(1,5):
  c = "Cn_"+str(i)
  data[c] = data[c].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","")

#전처리 된 데이터프레임
new_data = data.dropna(subset=['Cn_3','Cn_2', 'Cn_1', 'Cn_4'], inplace=False)
new_data = new_data.reset_index(drop=True)

#로그에서 출판사와 작가 모두 가진 행이 많이 없어서 어쩔 수 없이 두 열을 합쳐서 진행
new_data['author_publisher'] = new_data['author'] + new_data['publisher']
new_data = new_data.drop(['author', 'publisher', 'id'], axis=1)
new_data = new_data.dropna()
new_data = new_data.reset_index(drop=True)

#최종적으로 사용할 데이터프레임
df = new_data

#윈도우에서 진행할 경우에 C 드라이브에서 mecab 딕셔너리나 사용자 정의 딕셔너리를 설치해서 불러 올 수 있다.
mecab = Mecab('C:\\mecab\\mecab-ko-dic')
#만약에 리눅스에서 할 경우
#mecab = Mecab()

#각 책의 주요 정보의 단어 딕셔너리 작업하는 시간 확인용, 보통 오래 걸리지 않음
s = datetime.now().strftime('%H:%M:%S')
print(s)

#카테고리 1의 딕셔너리 만드는 작업
depth1_list = []

for index, row in tqdm(df.iterrows(), total=len(df)):
  text = row['Cn_1']
  tag = row['title']
  depth1_list.append(TaggedDocument(tags=[tag], words=mecab.morphs(text)))

print('문서의 수 :', len(depth1_list))


#카테고리 2
s = datetime.now().strftime('%H:%M:%S')
print(s)

depth2_list = []

for index, row in tqdm(df.iterrows(), total=len(df)):
  text4 = row['Cn_2']
  tag = row['title']
  depth2_list.append(TaggedDocument(tags=[tag], words=mecab.morphs(text4)))

print('문서의 수 :', len(depth2_list))

#카테고리 3
s = datetime.now().strftime('%H:%M:%S')
print(s)

depth3_list = []

for index, row in tqdm(df.iterrows(), total=len(df)):
  text5 = row['Cn_3']
  tag = row['title']
  depth3_list.append(TaggedDocument(tags=[tag], words=mecab.morphs(text5)))

print('문서의 수 :', len(depth3_list))


#카테고리 4
s = datetime.now().strftime('%H:%M:%S')
print(s)

depth4_list = []

for index, row in tqdm(df.iterrows(), total=len(df)):
  text6 = row['Cn_4']
  tag = row['title']
  depth4_list.append(TaggedDocument(tags=[tag], words=mecab.morphs(text6)))

print('문서의 수 :', len(depth4_list))

#출판사+작가
s = datetime.now().strftime('%H:%M:%S')
print(s)

tagged_corpus_list2 = []

for index, row in tqdm(df.iterrows(), total=len(df)):
  text2 = row['author_publisher']
  tag2 = row['title']
  tagged_corpus_list2.append(TaggedDocument(tags=[tag2], words=mecab.morphs(text2)))


#책 전체 Doc2Vec 임베딩 진행
#책의 정보를 따로 Doc2Vec으로 학습 시킨 후에 하나로 합침(concat)
#더미용으로 카테고리 1 사용, 사용할 전체 책의 벡터값의 크기 여기서는 100으로 임베딩
#이렇게 진행하는 이유는 따로 임베딩한 후 그 벡터값들을 합친 Doc2Vec 결과가 좋게 나왔음
s = datetime.now().strftime('%H:%M:%S')
print(s)

model = doc2vec.Doc2Vec(vector_size=100, alpha=0.025, min_alpha=0.025, workers=8, window=8)

# Vocabulary 빌드
model.build_vocab(depth1_list)

# Doc2Vec 학습
model.train(depth1_list, total_examples=model.corpus_count, epochs=50)

# 모델 저장
model.save('concat.doc2vec')
print('finish 1')

#카테고리 1 Doc2Vec 임베딩 진행
s = datetime.now().strftime('%H:%M:%S')
print(s)

model2 = doc2vec.Doc2Vec(vector_size=20, alpha=0.025, min_alpha=0.025, workers=8, window=8)
model2.build_vocab(depth1_list)
model2.train(depth1_list, total_examples=model2.corpus_count, epochs=50)
model2.save('d1.doc2vec')

print('finish 2')

#카테고리 2 Doc2Vec 임베딩 진행
s = datetime.now().strftime('%H:%M:%S')
print(s)

model5 = doc2vec.Doc2Vec(vector_size=20, alpha=0.025, min_alpha=0.025, workers=8, window=8)
model5.build_vocab(depth2_list)
model5.train(depth2_list, total_examples=model5.corpus_count, epochs=50)
model5.save('d2.doc2vec')

print('finish 3')


#카테고리 3 Doc2Vec 임베딩 진행
s = datetime.now().strftime('%H:%M:%S')
print(s)

model6 = doc2vec.Doc2Vec(vector_size=20, alpha=0.025, min_alpha=0.025, workers=8, window=8)
model6.build_vocab(depth3_list)
model6.train(depth3_list, total_examples=model6.corpus_count, epochs=50)
model6.save('d3.doc2vec')

print('finish 4')

#카테고리 4 Doc2Vec 임베딩 진행
s = datetime.now().strftime('%H:%M:%S')
print(s)

model7 = doc2vec.Doc2Vec(vector_size=20, alpha=0.025, min_alpha=0.025, workers=8, window=8)
model7.build_vocab(depth4_list)
model7.train(depth4_list, total_examples=model7.corpus_count, epochs=50)
model7.save('d4.doc2vec')

print('finish 5')


#출판사+작가 Doc2Vec 임베딩 진행
s = datetime.now().strftime('%H:%M:%S')
print(s)

model4 = doc2vec.Doc2Vec(vector_size=20, alpha=0.025, min_alpha=0.025, workers=8, window=8)
model4.build_vocab(tagged_corpus_list2)
model4.train(tagged_corpus_list2, total_examples=model4.corpus_count, epochs=50)
model4.save('dart_4.doc2vec')

print('finish 6')

#각 Doc2Vec 모델로부터 벡터값들을 추출해서 사용할 준비
vector = np.zeros((len(df), 20)).astype(np.float32)

for i in range(len(df)):
    vector[i] = model2.docvecs[df.iloc[i]['title']]

vector2 = np.zeros((len(df), 20)).astype(np.float32)

for i in range(len(df)):
    vector2[i] = model5.docvecs[df.iloc[i]['title']]

vector3 = np.zeros((len(df), 20)).astype(np.float32)

for i in range(len(df)):
    vector3[i] = model6.docvecs[df.iloc[i]['title']]

vector4 = np.zeros((len(df), 20)).astype(np.float32)

for i in range(len(df)):
    vector4[i] = model7.docvecs[df.iloc[i]['title']]

vector5 = np.zeros((len(df), 20)).astype(np.float32)

for i in range(len(df)):
    vector5[i] = model4.docvecs[df.iloc[i]['title']]

#벡터값들을 하나로 concat
vector_merge = np.hstack((vector, vector2, vector3, vector4, vector5))

#합쳐진 벡터값들의 임시 데이터프레임
temp_dataframe = pd.DataFrame(vector_merge)

list_1 = []
for i in range(100):
    list_1.append('Book_vec_'+str(i))

temp_dataframe.columns = list_1

#벡턱값들과 책의 정보를 가로로 결합
df_concat = pd.concat([df, temp_dataframe], axis=1)

df_concat.to_csv('1m_vectors_category.csv', index= False, encoding = 'utf-8')