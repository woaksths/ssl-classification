Keyword, Key-Phrase extraction
```
attention-based models can identify relavant keywords that are strongly related to the certain catergories
```
RQ1. BERT + bilstm attn layer + classification layer vs BERT
RQ2. attention score가 높다고, classification의 단서, Keyword는 아니다. (어떻게 개선시킬지?)  class 당 빈도수 기반의 사전 형성, dict({word:count}) 형태, 클래스당 dictionary size는 얼마나 만들어야하는지. 다른 클래스간 사전에서 겹치는 단어의 경우?
RQ3. stop words 제거 
RQ4. bilstm attn layer는 bert output 중 token level output에 추가한다. 
RQ5. attn_threshold = Percentile((attn_max - att_i), N)

Reference
1. Progress notes Classification and Keyword Extraction using Attention based Deep Learning Models with BERT


Mutltask Deep Neural Net 적용(?)
1. 학습 데이터셋이 적을 때, Multitask에 대하여 Finetuning 후,
2. 마지막 레이어만 제외한 뒤, 나머지 모든 레이어(BERT encoder 몸통)를 저장하여 재사용하는 방법

Reference
1.
2.
3.


Semi-supervised Learning
1. Pseudo labeling
2. MixMatch
3. Virtual Adversarial Training

Reference
1.
2.
3.
