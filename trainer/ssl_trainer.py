import torch
import os

class SSL_Trainer(object):
    
    '''
    TO DO
    
    1. PSEUDO-LABELING 
    2. UPDATE DATASET (1. add labeled data into train data 2. remove labeled data from unlabeled data)
    3. MAKE PERTURBED SAMPLES from labeled data

    4. TRAIN MODULE 
    5. EXTRACT lexicon FROM UNSEEN TRAIN DATA usning ATTENTION
    6. UPDATE LEXICON (1. remove some word  2. add some word)

    7. DEV EVAL (if possible test eval)
    8. SAVING & LOAD MODULE
   
    '''
    
    def __init__(self, ):
        '''
        unlabeled_data
        labeled_data
        test_data (if possible)
        tokenizer
        
        model
        optimizer
        criterion
        
        device
        lexicon_obj
        '''
        pass
    
    

    def train():
        ## outer epoch 
        ## inner epoch 
        
        # inner loop early stopping
        # outer loop early stopping 

        ## Outer Loop 
        '''
        1. outer loop에서 model init, optimizer init with best model
        (만약 처음이라면, supervised learning에서 가장 좋은 모델 가지고 옴, 아니라면 준지도 학습에서 가장 좋았던 모델 에폭 가지고 옴)
        
        2. with the best model, do labeling with unlabeled dataset 
        
        3. after per each epoch, update dataset
        
        4. go to the inner loop 
        '''
        
        
        ## Inner Loop 
        '''
        1. 추가된 학습 데이터를 통해 학습을 진행
        
        2. 각 epoch의 배치마다 매번 perturbed samples을 다르게 생성하도록 구현 
        
        3. after training all the epoch or early stopping, extract lexicon for the total labeled training dataset. 
        
        4. update existing lexicon based on extracted lexicon
        '''
        
