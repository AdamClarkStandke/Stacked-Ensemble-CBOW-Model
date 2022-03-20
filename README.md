# Stacked-Ensemble-CBOW-Model:
Implementation of a Stacked Ensemble Continuous Bag of Word Model, in which the Meta-Learner is made up of three different sub-models: 1) a one-dimensional CNN  2) a Stacked Bidirectional-Gru with Attention and 3) a Stacked Bidirectional-Lstm with Attention. The summary of the model architecture is below: 

# Pre-processing and CBOW embedding:
The data came from https://github.com/lutzhamel/fake-news/blob/master/data/fake_or_real_news.csv which is a dataset that deals with text classification of articles as being either fake or real. After dealing with this dataset previously and using a sklearn's MultinomialNB  and logistic regression models, I wanted to take a deep learning approch to the dataset after reading the awesome performance as found in the article https://opendatascience.com/deep-learning-finds-fake-news-with-97-accuracy/. Instead of just a CNN model, I decided to implement my own form of  Spacy's Text Ensemble model as found at  https://spacy.io/api/architectures#textcat with a continous bag of word model as found at https://pytorch.org/tutorials/beginner/nlp/word_embeddings_tutorial.html with a context window of 2 and an embedding dimension of 16.  

# Training Summary:
![summary](summary.png)

The meta model was trained for 20 epochs using the variable_test_set.csv file and the sub-models were trained on the variable_level_zero.csv.The C_Dnn model was trained for 27 epochs as seen in training_C-DNN_Model.txt with one layer of 100 filters/kernels of size 3x16, a learning rate of 3e-4, and a tolerence/stopping of 1e-1; the two layer GRU with attention model was trained for 7 epochs as seen in training_StackedGRUAttention_Model.txt with a learning rate of 3e-4,a tolerence/stopping of 1e-1, and 20 hidden units; and the two layer LSTM with attention model was trained for 7 epochs as seen in training_StackedLSTMAttention_Model.txt. All sub-models used a 32 batch size with a learning rate of 3e-4,a tolerence/stopping of 1e-1, and 20 hidden units.

# Testing Results of Meta-Model:
![testing](test_results.png)

# Structure of files in this Repo:
Here is extra information, regarding  structure of this repo:
| Models        | Second Header |
| ------------- | ------------- |
| C-DNN         | https://github.com/aCStandke/Stacked-Ensemble-CBOW-Model/blob/main/C-DNN_Model.ipynb |
|  GRU          |  https://github.com/aCStandke/Stacked-Ensemble-CBOW-Model/blob/main/StackedGruAttentionModel.ipynb |
|  LSTM         | https://github.com/aCStandke/Stacked-Ensemble-CBOW-Model/blob/main/StackedLSTMAttentionModel.ipynb  |
|  Meta         |  https://github.com/aCStandke/Stacked-Ensemble-CBOW-Model/blob/main/StackedEnsembleModel.ipynb |
| Single Threaded CBOW  | https://github.com/aCStandke/Stacked-Ensemble-CBOW-Model/blob/main/SingleThreadWord2VecCBOW.ipynb  |
| Multi Threaded CBOW  | https://github.com/aCStandke/Stacked-Ensemble-CBOW-Model/blob/main/MultiThreadWord2VecCBOW.ipynb  |


| Folders       |  Contains     |
| ------------- | ------------- |
| context_vectors | CBOW  context vectors |
| target_vectors  | CBOW target vectors  |
| dictionary      | contains index to word and word to index dictionaries |
| models          | contains trained weights/bias for the sub-models |
| training        | contains printouts of training results for sub-models  |


# References:
https://machinelearningmastery.com/stacking-ensemble-for-deep-learning-neural-networks/
https://towardsdatascience.com/how-to-use-datasets-and-dataloader-in-pytorch-for-custom-text-data-270eed7f7c00
https://github.com/explosion/spaCy/blob/5adedb8587818741dcd4ee1364ffb3f7d5074e75/spacy/ml/models/textcat.py#L114
https://github.com/prakashpandey9/Text-Classification-Pytorch/blob/master/models/LSTM_Attn.py
https://towardsdatascience.com/this-is-hogwild-7cc80cd9b944
https://pytorch.org/docs/stable/notes/multiprocessing.html
https://www.analyticsvidhya.com/blog/2020/01/first-text-classification-in-pytorch/
https://github.com/lutzhamel/fake-news/blob/master/fake_news_classification.ipynb
https://github.com/cezannec/CNN_Text_Classification
https://pytorch.org/tutorials/beginner/text_sentiment_ngrams_tutorial.html
https://pytorch.org/tutorials/beginner/nlp/word_embeddings_tutorial.html
https://pytorch.org/tutorials/
and Alexis
