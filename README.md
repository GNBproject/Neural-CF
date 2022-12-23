# Neural-CF
Neural Collaborative Filtering Model for personal recommendation


1. 클릭 기반 Neural CF 실험 결과가 있는 ncf_test_results.ipynb
2. colab 환경에서 실험한 구매 기반 Neural CF 실험 결과가 purchase_ncf_results.ipynb에 있다

***주의할 점은 경로 설정을 안했기 때문에 csv 파일을 읽기 위해서 사용할 환경에 맞춰서 경로 설정을 진행하고 실행해주세요.

3. Doc2Vec_comments.py는 Doc2Vec 임베딩 방법과 Doc2Vec로 데이터셋을 만드는 코드 내용이 있으며, 저 코드의 목저은 CTR_Neural CF용 책의 벡터값을 만든다.
  3.1 그래서 꼭 저 방식으로 안해도 되고 다른 책의 정보들을 임베딩해서 벡터값을 만들어 봐도 된다.
4. Sampling_Labeling_comments.py는 Negative Labeling 할 후보군을 뽑고 라벨링해서 데이터셋을 만든는 과정이다
  4.1 주의할 점은, 사용된 Doc2Vec 모델에서는 모든 책이 아니라 소개, 서평, 목차, 제목이 모두 있어야 함으로 있어야지만 유사 책 추천이 가능하다.
4.2 꼭 Doc2Vec 모델이 아니라, 다른 추천 모델을 활용해서 책의 후보군을 뽑아도 된다.
5. CTR_Neural_CF_comments.py는 클릭 기반 Neural CF를 실행 코드를 작성했다. 이 코드로 훈련하고 학습하면 된다.
  5.1 보통 데이터 수가 커지면 파라미터와 layer을 늘리면 성능이 좀 더 좋아진다.
6. Doc2vec_model_recommend_comment.py는 Doc2Vec 추천 모델을 만드는 코드다, 하지만 실제로 추천 용도로 사용하지 않았으며 후보군을 뽑는 용도로만 사용했다.
 
7. gb_recall_9.h5은 최종적으로 사용한 클릭 기반 neural cf 모델이다
