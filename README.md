# Agitation_detection

We propose a model for analysing the risk of agitation in people with dementia. The model is a semi-supervised model which combines a self-supervised learning model and a Bayesian ensemble classifican. We train and test the model on a dataset from a clinical study from UK Dementia Research Institut (UK DRI). In UK DRI, we have been developing and deploying in-home monitoring technologies and sensors to support people with dementia. The proposed model outperforms the baseline models in recall and f1-score by 20%. It also has better generalisability compared to the baseline models.

- Add UK DRI data in data folder.
- Run python self-supervised.py to train and test the self-supervised transformation learning model. It contains training of 10 autoencoders and we use the encoders to transform the data and add the psuedo-labels and train a CNN classifier on the psuedo-labelled data.
- Run python main_experiments.py to train and test the Bayesian ensemble model. The model contains the frozen trained CNN from self-supervised part and it contains 4 base classifiers: Naive Bayes, K-Nearest Neighbour(KNN), Support Vector Machine (SVM) and Gaussian Process (GP) Classifiers. It combines the 4 base classifier with BCNNet (Bayesian fusion).
- Run python baseline_experiments.py to train and test the baseline models for comparison. There are LSTM, BiLSTM, VGG, ResNet and Inception.
- The models will be saved in saved_models folder.


