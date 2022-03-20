# Stacked-Ensemble-CBOW-Model:
Implementation of a Stacked Ensemble Continuous Bag of Word Model(i.e. CBOW of Word2Vec), in which the Meta-Learner is made up of three different sub-models: 1) a one-dimensional CNN  2) a Stacked Bidirectional-Gru with Attention and 3) a Stacked Bidirectional-Lstm with Attention. The summary of the model architecture is below: 

# Training Summary:
=================================================================
Layer (type:depth-idx)                   Param #
=================================================================
MetaLearner                              --
├─StackedLSTMAtteionModel: 1-1           --
│    └─Embedding: 2-1                    (1,744,400)
│    └─LSTM: 2-2                         (16,000)
│    └─Linear: 2-3                       (41)
│    └─Sigmoid: 2-4                      --
├─TwoLayerGRUAttModel: 1-2               --
│    └─Embedding: 2-5                    (1,744,400)
│    └─GRU: 2-6                          (12,000)
│    └─Linear: 2-7                       (41)
│    └─Sigmoid: 2-8                      --
├─C_DNN: 1-3                             --
│    └─Embedding: 2-9                    (1,744,400)
│    └─Conv2d: 2-10                      (4,900)
│    └─Linear: 2-11                      (101)
│    └─Sigmoid: 2-12                     --
├─Linear: 1-4                            8
├─Linear: 1-5                            3
├─Sigmoid: 1-6                           --
=================================================================
Total params: 5,266,294
Trainable params: 11
Non-trainable params: 5,266,283
=================================================================

The meta model was trained for 20 epochs using the variable_test_set.csv file and the sub-models were trained on the variable_level_zero.csv. C_Dnn was trained for 27 epochs as seen in training_C-DNN_Model.txt; the two layer GRU with attention model was trained for 7 epochs as seen in training_StackedGRUAttention_Model.txt: and the two layer LSTM with attention model was trained for 7 epochs as seen in training_StackedLSTMAttention_Model.txt. All sub-models used a 32 batch size.

# Testing Results of Meta-Model:


