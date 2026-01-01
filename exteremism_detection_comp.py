#The dataset being used is from a competition on kaggle 'Social Media Extremism Detection Challenge'
#Goal: Create a Machine Learning model that classifies social media text as extremist or non extremist!
#Submissions in the competition will be evaluated using classification accuracy on a held-out test set.


import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from collections import Counter
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression


train_path = "data/train.csv"
df = pd.read_csv(train_path)
print(df.head())
print(df.columns)
print(df.isna().sum()) #no missing values 
df.info() #3 columns two of them are objects and ID is int 

#removing the columns which identify every row uniquely (ID)
df = df.drop('ID',axis=1)

#lets check the class balance in target lable 
counts = Counter(df['Extremism_Label'])

plt.bar(counts.keys(),counts.values(),color= 'blue',width=0.5)
plt.xticks(rotation=45)
plt.xlabel('Class')
plt.ylabel('Count')
plt.show() #balanced counts in each class 

#text length distribution
text_lengths = [len(line.strip().split()) for line in df['Original_Message']]

for i in range(len(text_lengths)):
    plt.scatter(text_lengths[i],i,color='blue')

plt.xlabel('Word count')
plt.show() #gives idea about number of tokens, since NB is sensitive to token freq, mostly wordcount is 0-50 others can be considered long texts ig 

#------------
'''Baseline Model'''
#------------

#splitting data into label and feature
X = df['Original_Message']
y = df['Extremism_Label']

#splitting the data into train/validation 
X_train,X_validate,y_train,y_validate = train_test_split(X,y,test_size=0.2,random_state=42)

#pipeline 
pipeline = Pipeline([
    ('tfidf',TfidfVectorizer()), #turns tests into numbers, TF: how many times the word appearns in my text,IDF: How frequent is this word in complete dataset; also here for baseline n_gram=1(default)
    ('model',MultinomialNB())
])

#training and validation 
pipeline.fit(X_train,y_train)
y_pred = pipeline.predict(X_validate)

print('Baseline model accuracy_score: ',accuracy_score(y_validate,y_pred)) #0.8022 

#lets find a text that is miscalssified and see why so 
baseline_validation_results = pd.DataFrame({
    'X_val': X_validate,
    'y_val': y_validate,
    'y_pred': y_pred
})

miscalssified = baseline_validation_results[baseline_validation_results['y_val'] != baseline_validation_results['y_pred']]
print(miscalssified.head(1))

'''men are superior to women in anything physical women are superior to men in empathy and emotional attachment and that whole nurturing concept as they biologically evolved to be motherly some psychological differences make all women mentally ill though'''
'''Above was one of the sentences that was miscalssified as non_entremist but hsould be extremist, 
we can notice something here that this sentence doesnt consist any slangs/no explicit voilent words etc... 
this obsevation concludes that the model needs to see multiple words together, so we introduce bigrams in tuning, this will help us look after words such as "mentally ill" '''

#hyperparameter tuning by activating bigrams only in tf-idf
pipeline = Pipeline([
    ('tfidf',TfidfVectorizer(ngram_range=(2,2))), #0.7644 now only bigrams will be selected no unigrams
    ('model',MultinomialNB())
])

pipeline.fit(X_train,y_train)
y_pred= pipeline.predict(X_validate)
print('Bigram model accuracy_score: ',accuracy_score(y_validate,y_pred)) #0.764 not a terreble drop but it did drop 
#what this shows is that our model cant do good on only uni or bigrams we need both to capture most of the info 
#so training the model on uni and bigrams now 

pipeline = Pipeline([
    ('tfidf',TfidfVectorizer(ngram_range=(1,2))), #both uni and bigrams 
    ('model',MultinomialNB())
])

pipeline.fit(X_train,y_train)
y_pred= pipeline.predict(X_validate)
print('uni and bigram model accuracy_score: ',accuracy_score(y_validate,y_pred)) #0.811 accuracy increased by a lil than the basemodel

'''Seems like NB learned waht it could and now the accuracy paltued at 0.811, it wont help even if we introduced trigrams'''
'''Using the tuned NB model to predict on test dataset and preparing the submission csv later we'll use full train dataset to make predictions on the test dataset 
to make 2nd submission csv cuz kaggle allows multiple submissions'''

#first submission, tuned model trained on X_train
test_path = "data/test.csv"
test_df = pd.read_csv(test_path)
test_df.info()

X_test = test_df['Original_Message']

test_pred = pipeline.predict(X_test) #kaggle public LB score= 0.794

submission = pd.DataFrame({
    "ID": test_df['ID'],
    "Extremism_Label": test_pred
})

### submission.to_csv("submissionNB.csv",index= False)

#second submission; tuned model trained on full training dataset not X_train
pipeline.fit(X,y)
y_pred= pipeline.predict(X_test) #kaggle public LB score= 0.816 which increased hehehehehehehe

submission = pd.DataFrame({
    "ID": test_df['ID'],
    "Extremism_Label": y_pred
})

### submission.to_csv("submissionNB1.csv",index= False)

#-------------
'''Logistic Regression'''
#-------------

pipeline = Pipeline([
    ('tfidf',TfidfVectorizer(ngram_range=(1,2))), #both uni and bigrams 
    ('model',LogisticRegression(n_jobs=-1,random_state=42))
])

pipeline.fit(X_train,y_train)
y_pred = pipeline.predict(X_validate)
print('Logistic_regression model accuracy_score: ',accuracy_score(y_validate,y_pred)) #0.82 it increased 

#lets hypertune the linear regression model to see if it helps in any terms 
C_values = range(1,100)
for c in C_values:
    pipeline = Pipeline([
    ('tfidf',TfidfVectorizer(ngram_range=(1,2))), 
    ('model',LogisticRegression(n_jobs=-1,
                                C=float(c), #small c-->strong regularization/underfit, large c-->weak regularization/overfitting the train data
                                random_state=42))
                        ])
    pipeline.fit(X_train,y_train)
    y_pred = pipeline.predict(X_validate)
    score = accuracy_score(y_validate,y_pred)
    # print(f'Linear_regression at C={c} accuracy_score: ',score)
    plt.scatter(c,score,color='blue')

plt.xlabel('C')    
plt.ylabel('accuracy score')
plt.show() #the sweet spot for c is around 43-49 which give accuraccy around 84.8 ie well generalized not over fitted; after that noise dominates and model starts overfitting 
#locking value of C=45, validation accuracy= 0.848

#final pipeline
pipeline = Pipeline([
    ('tfidf',TfidfVectorizer(ngram_range=(1,2))), 
    ('model',LogisticRegression(n_jobs=-1,
                                C=45,
                                random_state=42))
                        ])

pipeline.fit(X_train,y_train)
y_pred = pipeline.predict(X_test) #public LB score: 0.805, it decreased but its public LB score so we dont know about the private LB yet 

submission = pd.DataFrame({
    "ID": test_df['ID'],
    "Extremism_Label": y_pred
})

### submission.to_csv("submissionLR.csv",index= False)

'''Conclusion:
This project progressed from a Naive Bayes baseline with unigram and bigram features
to a tuned Logistic Regression model. Performance improvements were achieved through
feature representation and regularization tuning, while validation stability was
prioritized over public leaderboard scores to ensure proper generalization.'''
