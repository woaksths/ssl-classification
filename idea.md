### Keyword, Key-Phrase extraction
```
attention-based models can identify relavant keywords that are strongly related to the certain catergories
```
- RQ1. BERT + bilstm attn layer + classification layer vs BERT
- RQ2. attention score가 높다고, classification의 단서, Keyword는 아니다. (어떻게 개선시킬지?)  class 당 빈도수 기반의 사전 형성, dict({word:count}) 형태, 클래스당 dictionary size는 얼마나 만들어야하는지. 다른 클래스간 사전에서 겹치는 단어의 경우?
- RQ3. stop words 제거 
- RQ4. bilstm attn layer는 bert output 중 token level output에 추가한다. 
- RQ5. attn_threshold = Percentile((attn_max - att_i), N)
- RQ6. 사전으로 만들어질 단어가 overlap 되는 것과 overlap 되지 않는 것의 의미 분석 
- RQ7. 사전을 만들 때, 가중치를 TF-IDF로 주는 방법 고민해보기
- RQ8. 모델이 클래스를 예측할 떄, 언제 사전에 의존하게 할지(?)
- RQ9. 어텐션 스코어가 높다고, 항상 classification 예측에 중요한 단서가 될 단어가 아니다. 해결방법: 입력 시퀀스에서 어텐션 스코어가 높은 단어 n개를 뽑고, 모든 batch 데이터에 대해서 누적 count 값을 적용. 누적 count를 통해 일반적으로 중요한 단어들을 뽑을 수 있게 됨. 

Reference
1. Progress notes Classification and Keyword Extraction using Attention based Deep Learning Models with BERT


### Mutltask Deep Neural Net 적용
1. 학습 데이터셋이 적을 때, Multitask에 대하여 Finetuning 후,
2. 마지막 레이어만 제외한 뒤, 나머지 모든 레이어(BERT encoder 몸통)를 저장하여 재사용하는 방법

Reference
1.
2.
3.


### Semi-supervised Learning
1. Pseudo labeling
2. MixMatch
3. Virtual Adversarial Training

Reference
1.
2.
3.
