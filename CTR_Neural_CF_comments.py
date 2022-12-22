from gensim.models import doc2vec
from gensim.models.doc2vec import Doc2Vec
from gensim.models.keyedvectors import WordEmbeddingsKeyedVectors
from tensorflow.keras.layers import Input, Embedding, Flatten, Dot, Dense, Concatenate
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.utils import plot_model
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import keras
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#postive, negative 라벨링이 완료된 책 데이터셋
data = pd.read_csv('.../user_book_sample.csv')

#유저, 책 isbn 인코딩
user_ids = data["customer_idx"].unique().tolist()
user2user_encoded = {x: i for i, x in enumerate(user_ids)}
userencoede2user = {i: x for i, x in enumerate(user_ids)}

book_ids = data["isbn"].unique().tolist()
book2book_encoded = {x: i for i, x in enumerate(book_ids)}
book_encoded2book = {i: x for i, x in enumerate(book_ids)}
data["encoded_idx"] = data["customer_idx"].map(user2user_encoded)
data["encoded_isbn"] = data["isbn"].map(book2book_encoded)

#test train 분리
train, test = train_test_split(data,test_size = 0.2)

#유니크한 유저와 책 수 확인 및 벡터값 항목 접근
number_of_unique_user = len(data.encoded_idx.unique())
number_of_unique_book_id = len(data.encoded_isbn.unique())
list_1 = []
for i in range(100):
    a = 'Book_vec_'+str(i)
    list_1.append(a)

#test, train, 전체 데이터를 keras에서 알 수 있는 데이터 형태로 변환
X = train[list_1].to_numpy()
Y = test[list_1].to_numpy()
Z = data[list_1].to_numpy()

#사용한 model의 하이퍼파라미터
n_latent_factors_user = 32
n_latent_factors_book = 32
n_latent_factors_mf = 8
n_users, n_books = number_of_unique_user, number_of_unique_book_id

book_input = keras.layers.Input(shape=[100],name='Book')

user_input = keras.layers.Input(shape=[1],name='User')
user_vec_mlp = keras.layers.Flatten(name='FlattenUsers-MLP')(keras.layers.Embedding(n_users + 1, n_latent_factors_user,name='User-Embedding-MLP')(user_input))

user_vec_mf = keras.layers.Flatten(name='FlattenUsers-MF')(keras.layers.Embedding(n_users + 1, 100,name='User-Embedding-MF')(user_input))

concat = keras.layers.concatenate([book_input, user_vec_mlp], name='Concat')
dense = keras.layers.Dense(512,name='FullyConnected',activation='relu')(concat)
dense_2 = keras.layers.Dense(256,name='FullyConnected-1',activation='relu')(dense)
dense_3 = keras.layers.Dense(128,name='FullyConnected-2',activation='relu')(dense_2)
dense_4 = keras.layers.Dense(64,name='FullyConnected-3',activation='relu')(dense_3)
dense_5 = keras.layers.Dense(32,name='FullyConnected-4',activation='relu')(dense_4)

pred_mf = keras.layers.Dot(axes=1)([book_input, user_vec_mf])
pred_mlp = keras.layers.Dense(1, activation='relu',name='Activation')(dense_5)


combine_mlp_mf = keras.layers.concatenate([pred_mf, pred_mlp] ,name='Concat-MF-MLP')
result_combine = keras.layers.Dense(128,name='Combine-MF-MLP')(combine_mlp_mf)
deep_combine = keras.layers.Dense(64,name='FullyConnected-6')(result_combine)
deep_combine_2 = keras.layers.Dense(32,name='FullyConnected-7')(result_combine)
deep_combine_3 = keras.layers.Dense(16,name='FullyConnected-8')(deep_combine_2)
deep_combine_4 = keras.layers.Dense(8,name='FullyConnected-9')(deep_combine_3)



result = keras.layers.Dense(1,activation = 'sigmoid', kernel_initializer='lecun_uniform', name='Prediction')(deep_combine_4)


model = keras.Model([user_input, book_input], result)
opt = tf.keras.optimizers.Adam(learning_rate =0.0000001)
model.compile(optimizer='adam',loss= 'mse', metrics=['accuracy', tf.keras.metrics.Precision(),tf.keras.metrics.Recall(), tf.keras.metrics.AUC()])

#best precision, AUC, recall 모델 저장
auc = keras.callbacks.ModelCheckpoint('gb_auc.h5', save_best_only=True, monitor='val_auc', mode='max') 
pre = keras.callbacks.ModelCheckpoint('gb_pre.h5', save_best_only=True, monitor='val_precision', mode='max')
recall = keras.callbacks.ModelCheckpoint('gb_recall.h5', save_best_only=True, monitor='val_recall', mode='max')

#학습 진행
history = model.fit([train.encoded_idx, X],train.click,batch_size = 64,epochs=100,callbacks=[EarlyStopping(monitor='val_auc', mode='max', verbose=1, patience=10), auc, pre, recall], verbose=1,validation_split=0.2) 

#loss
plt.figure(figsize=[8,6])
plt.plot(history.history['loss'],'r',linewidth=3.0)
plt.plot(history.history['val_loss'],'b',linewidth=3.0)
plt.legend(['Training loss', 'Validation Loss'],fontsize=18)
plt.xlabel('Epochs ',fontsize=16)
plt.ylabel('Loss',fontsize=16)
plt.title('Loss Curves',fontsize=1)
plt.show()

# accuracy
plt.figure(figsize=[8,6])
plt.plot(history.history['accuracy'],'r',linewidth=3.0)
plt.plot(history.history['val_accuracy'],'b',linewidth=3.0)
plt.legend(['Training Accuracy', 'Validation Accuracy'],fontsize=18)
plt.xlabel('Epochs ',fontsize=16)
plt.ylabel('Accuracy',fontsize=16)
plt.title('Accuracy Curves',fontsize=1)
plt.show()

#precision
plt.figure(figsize=[8,6])
plt.plot(history.history['precision'],'r',linewidth=3.0)
plt.plot(history.history['val_precision'],'b',linewidth=3.0)
plt.legend(['Training Precision', 'Validation Precision'],fontsize=18)
plt.xlabel('Epochs ',fontsize=16)
plt.ylabel('Precision',fontsize=16)
plt.title('Precision Curves',fontsize=1)
plt.show()

#recall
plt.figure(figsize=[8,6])
plt.plot(history.history['recall'],'r',linewidth=3.0)
plt.plot(history.history['val_recall'],'b',linewidth=3.0)
plt.legend(['Training Recall', 'Validation Recall'],fontsize=18)
plt.xlabel('Epochs ',fontsize=16)
plt.ylabel('Recall',fontsize=16)
plt.title('Recall Curves',fontsize=1)
plt.show()

#val AUC
plt.figure(figsize=[8,6])
plt.plot(history.history['auc'],'r',linewidth=3.0)
plt.plot(history.history['val_auc'],'b',linewidth=3.0)
plt.legend(['Training AUC', 'Validation AUC'],fontsize=18)
plt.xlabel('Epochs ',fontsize=16)
plt.ylabel('AUC',fontsize=16)
plt.title('AUC Curves',fontsize=1)
plt.show()

#필요한 model 로드 방법
model_precision = load_model('gb_pre.h5')
#실제로 사용한 모델
model_recall = load_model('gb_recall_9.h5')
model_auc = load_model('gb_auc.h5')

#각 model test 평가
loss, accuracy, auc, precision, recall = model_precision.evaluate([test.encoded_idx, Y], test.click, verbose=1)
loss, accuracy, auc, precision, recall = model_recall.evaluate([test.encoded_idx, Y], test.click, verbose=1)
loss, accuracy, auc, precision, recall = model_auc.evaluate([test.encoded_idx, Y], test.click, verbose=1)

#사용하는 모델로부터 전체 데이터에 대한 예측값 얻기
prediction = model_recall.predict([data.encoded_idx, Z])

#test sample은 전체 데이터로 진행
test_sample = data
loss, accuracy, auc, precision, recall = model_recall.evaluate([data.encoded_idx, Z], data.click, verbose=1)

#데이터에 예측값 concat
test_sample['prediction'] = prediction

#클릭 기반 Neural CF 개인화 추천용 데이터 구축
user_prediction = test_sample[["customer_idx", "isbn", 'title', 'Cn_1', 'Cn_2', 'Cn_3', 'Cn_4', 'click', 'prediction']]
test_prediction = user_prediction.drop_duplicates(subset= ['customer_idx', 'title'], keep='first')
test_prediction.to_csv('final_prediction.csv', index= False, encoding = 'utf-8')
