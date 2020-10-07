- pseudo-labeling
- Can we build a system capable of requiring minimal amount of supervision which can learn majority of the tasks on its own.
- supervised learning 한계: human annotation to a large dataset can be very hard and expensive.

- Semi-Supervised Learning
- use both labeld data(supervised learning)and unlablelled data(unsupervised learning)
- unlabelled data 사용의 장점

1. 레이블을 하지 않아도 되어 cheap, abundant
2. unlabelled data를 통해, 좀 더 정확한 decision boundary -> model robustness하게 만든다. 

Pseudo Labeling
- Instead of manually labelling the unlabelled data, we give approximate labels on the basis of the labelled data.
- 1. Train the model with labeled data
- 2. Use the trained model to predict labels for the unlabeled data
- 3. retrained the model with pseudo and labeled datasets together. 
- 결국 학습데이터셋은 augmented dataset(pseudo label + labeld dataset)이 됨
- 
