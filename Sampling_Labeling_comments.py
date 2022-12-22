import numpy as np
import pandas as pd
from gensim.models.doc2vec import TaggedDocument
from tqdm import tqdm
from gensim.models import doc2vec
from gensim.models.doc2vec import Doc2Vec
from gensim.models.keyedvectors import WordEmbeddingsKeyedVectors

#긍정 데이터는 공앤박측에서 제공한 wish, click, 구매를 모든 행이 unqiue한 상태로 구축
pos = pd.read_csv('C:/Users/Sang Been Yim/Downloads/positive_click_final.csv')
vector = pd.read_csv('1m_vectors_book.csv')

#데이터 프레임 구축
df = pos[["customer_idx", "isbn"]]

#전처리 진행
v1 = vector.drop_duplicates(subset=["isbn", "title"], keep='first')
testing_2 = pd.merge(df, v1, on = ['isbn'])
testing_3 = testing_2.drop_duplicates(keep='first', ignore_index=True)

#만약에 Doc2Vec 모델에서 특정 책이 없는 경우 역시 제외 해줘야 한다
testing_3.loc[testing_3['title'] == 'Hanok (The Korean House)']
testing_4 = testing_3[testing_3.title != 'Hanok (The Korean House)']

#유니크한 유저별로 제목을 묶는다
click_unique = testing_4.groupby(['customer_idx'])['title'].unique().reset_index()

#최종적으로 사용한 Doc2Vec 모델
model = Doc2Vec.load('new_GNB_concat.doc2vec')

#클릭한 책들의 후보군을 뽑는 과정
for i in range(len(click_unique)):
    customer_idx = click_unique.customer_idx[i]
    title_list = click_unique.title[i]
    for j in range(len(title_list)):
        candidate_list = []
        candidate = []
        candidate_list = model.docvecs.most_similar(title_list[j], topn=10)
        for l in range(len(candidate_list)):
            candidate.append(candidate_list[l][0])
        candidate
        final_candidate = [x for x in candidate if x not in title_list]
        num1 = str(j)
        print(num1)
        for k in range(len(final_candidate)):
            if i == 0 and (j == 0 and k ==0):
                click_0_test = vector[vector['title'] == final_candidate[k]].copy()
                click_0_test['customer_idx'] = customer_idx
                cols = click_0_test.columns.tolist()
                cols = cols[-1:] + cols[:-1]
                click_0_test = click_0_test[cols]
                click_0_test['click'] = 0
            else:
                tmp = vector[vector['title'] == final_candidate[k]].copy()
                tmp['customer_idx'] = click_unique.customer_idx[i]
                cols = tmp.columns.tolist()
                cols = cols[-1:] + cols[:-1]
                tmp = tmp[cols]
                tmp['click'] = 0
                click_0_test = pd.concat([click_0_test, tmp])
    num2 = str(i)
    print("finish "+ num2)

testing_4['click'] = 1
#negative labeling이 진행된 후보군과 positive인 클릭한 책들을 라벨링된 데이터를 구축
df_sample = pd.concat([testing_4, click_0_test])
df_sample.to_csv('user_book_sample.csv', index= False, encoding = 'utf-8')