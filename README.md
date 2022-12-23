# Neural-CF
Neural Collaborative Filtering Model for personal recommendation


1. 클릭 기반 Neural CF 실험 결과가 있는 ncf_test_results.ipynb
2. colab 환경에서 실험한 구매 기반 Neural CF 실험 결과가 purchase_ncf_results.ipynb에 있다

3. Doc2Vec_comments.py는 Doc2Vec 임베딩 방법과 Doc2Vec로 데이터셋을 만드는 코드 내용이 있으며, 저 코드의 목저은 CTR_Neural CF용 책의 벡터값을 만든다.
4. Sampling_Labeling_comments.py는 Negative Labeling 할 후보군을 뽑고 라벨링해서 데이터셋을 만든는 과정이다
5. CTR_Neural_CF_comments.py는 클릭 기반 Neural CF를 실행 코드를 작성했다. 이 코드로 훈련하고 학습하면 된다.
6. Doc2vec_model_recommend_comment.py는 Doc2Vec 추천 모델을 만드는 코드다, 하지만 실제로 추천 용도로 사용하지 않았으며 후보군을 뽑는 용도로만 사용했다.
7. gb_recall_9.h5은 최종적으로 사용한 클릭 기반 neural cf 모델이다
