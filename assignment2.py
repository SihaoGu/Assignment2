# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
from collections import Counter
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.feature_extraction.text import TfidfVectorizer
import string

with open('assignment2/finefoods.txt',encoding ='latin-1') as f:
    data = f.readlines()

# remove '/n' of the list
data = list(map(lambda x:x.strip(),data))

# remove '' in the list
data = list(filter(('').__ne__, data))


data.remove('88 years old. ...')
data.remove('...creative powers b...')
data.remove('School Princi...')
data.remove('School Princi...')
data.remove('I am a voracious reader/li...')
data.remove('School Princi...')
data.remove('...creative powers b...')


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

data2 = list(chunks(data, 8))

title = ['product/productId','review/userId','review/profileName','review/helpfulness',
        'review/score','review/time','review/summary','review/text']

wholedata = []
for l in data2:
    dic1 = dict.fromkeys(title,0)
    for k in l:
        dic1[k.split(':')[0]] = k.split(':')[1].strip()
    wholedata.append(dic1)

# number of users
len(wholedata)

### [1] unigram + TFIDF + remove punctuation
training = wholedata[:180000]
validation = wholedata[180000:360000]
test = wholedata[360000:]

punctuation = set(string.punctuation)
def computeMSE(dataset):
    reviews = []
    for l in dataset:
        r = ''.join([c for c in l['review/text'].lower() if not c in punctuation])
        reviews.append(r)
    X = vectorizer.transform(reviews)
    y = list(map(float,[d['review/score'] for d in dataset] ))
    return X,y,reviews


reviews_train = []
for l in training:
    r = ''.join([c for c in l['review/text'].lower() if not c in punctuation])
    reviews_train.append(r)

vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(reviews_train)
y_train = [d['review/score'] for d in training]
y_train = list(map(float, y_train))

X_valid,y_valid,reviews_valid = computeMSE(validation)

para = [0.01, 0.1, 1, 10, 100]
for i in para:
    clf = linear_model.Ridge(i, fit_intercept=False)
    clf.fit(X_train, y_train)
    theta = clf.coef_
    predictions = clf.predict(X_valid)
    print(mean_squared_error(y_valid, predictions))

#select regularization parameters 1, becasue it has lower MSE

X_test,y_test,reviews_test = computeMSE(test)
clf = linear_model.Ridge(1, fit_intercept=False)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
lt_predictions = list(predictions)
for i in range(len(lt_predictions)):
    if lt_predictions[i] < 0:
        lt_predictions[i] =0
    elif lt_predictions[i] > 5:
        lt_predictions[i] = 5 
new_predictions = []
for l in range(len(lt_predictions)):
    new_predictions.append(round(lt_predictions[l]))
uni_tfidf_re = mean_squared_error(y_test, new_predictions)
print(uni_tfidf_re)


### [2] unigram + TFIDF + remove punctuation + remove stopwords
from nltk.corpus import stopwords
stopwords = stopwords.words('english')
def computeMSE(dataset):
    reviews = []
    for l in dataset:
        r = ''.join([c for c in l['review/text'].lower() if not c in punctuation])
        reviews.append(r)
    X = vectorizer.transform(reviews)
    y = list(map(float,[d['review/score'] for d in dataset] ))
    return X,y,reviews


reviews_train = []
for l in training:
    r = ''.join([c for c in l['review/text'].lower() if not c in punctuation])
    reviews_train.append(r)

vectorizer = TfidfVectorizer(stop_words = stopwords)
X_train = vectorizer.fit_transform(reviews_train)
y_train = [d['review/score'] for d in training]
y_train = list(map(float, y_train))

X_valid,y_valid,reviews_valid = computeMSE(validation)

para = [0.01, 0.1, 1, 10, 100]
for i in para:
    clf = linear_model.Ridge(i, fit_intercept=False)
    clf.fit(X_train, y_train)
    theta = clf.coef_
    predictions = clf.predict(X_valid)
    print(mean_squared_error(y_valid, predictions))

#select regularization parameters 1, becasue it has lower MSE

X_test,y_test,reviews_test = computeMSE(test)
clf = linear_model.Ridge(1, fit_intercept=False)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
lt_predictions = list(predictions)
for i in range(len(lt_predictions)):
    if lt_predictions[i] < 0:
        lt_predictions[i] =0
    elif lt_predictions[i] > 5:
        lt_predictions[i] = 5 
new_predictions = []
for l in range(len(lt_predictions)):
    new_predictions.append(round(lt_predictions[l]))
uni_tfidf_re_stop = mean_squared_error(y_test, new_predictions)
print(uni_tfidf_re_stop)

### [3] bigram + TFIDF + remove punctuation
def computeMSE(dataset):
    reviews = []
    for l in dataset:
        r = ''.join([c for c in l['review/text'].lower() if not c in punctuation])
        reviews.append(r)
    X = vectorizer.transform(reviews)
    y = list(map(float,[d['review/score'] for d in dataset] ))
    return X,y,reviews


reviews_train = []
for l in training:
    r = ''.join([c for c in l['review/text'].lower() if not c in punctuation])
    reviews_train.append(r)

vectorizer = TfidfVectorizer(ngram_range = (2,2))
X_train = vectorizer.fit_transform(reviews_train)
y_train = [d['review/score'] for d in training]
y_train = list(map(float, y_train))

X_valid,y_valid,reviews_valid = computeMSE(validation)

para = [0.01, 0.1, 1, 10, 100]
for i in para:
    clf = linear_model.Ridge(i, fit_intercept=False)
    clf.fit(X_train, y_train)
    theta = clf.coef_
    predictions = clf.predict(X_valid)
    print(mean_squared_error(y_valid, predictions))
    
#select regularization parameters 0.1, becasue it has lower MSE

X_test,y_test,reviews_test = computeMSE(test)
clf = linear_model.Ridge(0.1, fit_intercept=False)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
lt_predictions = list(predictions)
for i in range(len(lt_predictions)):
    if lt_predictions[i] < 0:
        lt_predictions[i] =0
    elif lt_predictions[i] > 5:
        lt_predictions[i] = 5 
new_predictions = []
for l in range(len(lt_predictions)):
    new_predictions.append(round(lt_predictions[l]))
bi_tfidf_re = mean_squared_error(y_test, new_predictions)
print(bi_tfidf_re)   

### [4] bigram + TFIDF + remove punctuation + remove stopwords
def computeMSE(dataset):
    reviews = []
    for l in dataset:
        r = ''.join([c for c in l['review/text'].lower() if not c in punctuation])
        reviews.append(r)
    X = vectorizer.transform(reviews)
    y = list(map(float,[d['review/score'] for d in dataset] ))
    return X,y,reviews


reviews_train = []
for l in training:
    r = ''.join([c for c in l['review/text'].lower() if not c in punctuation])
    reviews_train.append(r)

vectorizer = TfidfVectorizer(ngram_range = (2,2),stop_words = stopwords)
X_train = vectorizer.fit_transform(reviews_train)
y_train = [d['review/score'] for d in training]
y_train = list(map(float, y_train))

X_valid,y_valid,reviews_valid = computeMSE(validation)

para = [0.01, 0.1, 1, 10, 100]
for i in para:
    clf = linear_model.Ridge(i, fit_intercept=False)
    clf.fit(X_train, y_train)
    theta = clf.coef_
    predictions = clf.predict(X_valid)
    print(mean_squared_error(y_valid, predictions))
    
#select regularization parameters 0.1, becasue it has lower MSE

X_test,y_test,reviews_test = computeMSE(test)
clf = linear_model.Ridge(0.1, fit_intercept=False)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
lt_predictions = list(predictions)
for i in range(len(lt_predictions)):
    if lt_predictions[i] < 0:
        lt_predictions[i] =0
    elif lt_predictions[i] > 5:
        lt_predictions[i] = 5 
new_predictions = []
for l in range(len(lt_predictions)):
    new_predictions.append(round(lt_predictions[l]))
bi_tfidf_re_stop = mean_squared_error(y_test, new_predictions)
print(bi_tfidf_re_stop)   


### [5] uni+ bigram + TFIDF + remove punctuation
def computeMSE(dataset):
    reviews = []
    for l in dataset:
        r = ''.join([c for c in l['review/text'].lower() if not c in punctuation])
        reviews.append(r)
    X = vectorizer.transform(reviews)
    y = list(map(float,[d['review/score'] for d in dataset] ))
    return X,y,reviews


reviews_train = []
for l in training:
    r = ''.join([c for c in l['review/text'].lower() if not c in punctuation])
    reviews_train.append(r)

vectorizer = TfidfVectorizer(ngram_range = (1,2))
X_train = vectorizer.fit_transform(reviews_train)
y_train = [d['review/score'] for d in training]
y_train = list(map(float, y_train))

X_valid,y_valid,reviews_valid = computeMSE(validation)

para = [0.01, 0.1, 1, 10, 100]
for i in para:
    clf = linear_model.Ridge(i, fit_intercept=False)
    clf.fit(X_train, y_train)
    theta = clf.coef_
    predictions = clf.predict(X_valid)
    print(mean_squared_error(y_valid, predictions))
    
#select regularization parameters 0.1, becasue it has lower MSE

X_test,y_test,reviews_test = computeMSE(test)
clf = linear_model.Ridge(0.1, fit_intercept=False)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
lt_predictions = list(predictions)
for i in range(len(lt_predictions)):
    if lt_predictions[i] < 0:
        lt_predictions[i] =0
    elif lt_predictions[i] > 5:
        lt_predictions[i] = 5 
new_predictions = []
for l in range(len(lt_predictions)):
    new_predictions.append(round(lt_predictions[l]))
uni_bi_tfidf_re = mean_squared_error(y_test, new_predictions)
print(uni_bi_tfidf_re)   

### [6] uni+ bigram + TFIDF + remove punctuation + remove stopwords
def computeMSE(dataset):
    reviews = []
    for l in dataset:
        r = ''.join([c for c in l['review/text'].lower() if not c in punctuation])
        reviews.append(r)
    X = vectorizer.transform(reviews)
    y = list(map(float,[d['review/score'] for d in dataset] ))
    return X,y,reviews


reviews_train = []
for l in training:
    r = ''.join([c for c in l['review/text'].lower() if not c in punctuation])
    reviews_train.append(r)

vectorizer = TfidfVectorizer(ngram_range = (1,2),stop_words = stopwords)
X_train = vectorizer.fit_transform(reviews_train)
y_train = [d['review/score'] for d in training]
y_train = list(map(float, y_train))

X_valid,y_valid,reviews_valid = computeMSE(validation)

para = [0.01, 0.1, 1, 10, 100]
for i in para:
    clf = linear_model.Ridge(i, fit_intercept=False)
    clf.fit(X_train, y_train)
    theta = clf.coef_
    predictions = clf.predict(X_valid)
    print(mean_squared_error(y_valid, predictions))
    
#select regularization parameters 1, becasue it has lower MSE

X_test,y_test,reviews_test = computeMSE(test)
clf = linear_model.Ridge(1, fit_intercept=False)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
lt_predictions = list(predictions)
for i in range(len(lt_predictions)):
    if lt_predictions[i] < 0:
        lt_predictions[i] =0
    elif lt_predictions[i] > 5:
        lt_predictions[i] = 5 
new_predictions = []
for l in range(len(lt_predictions)):
    new_predictions.append(round(lt_predictions[l]))
uni_bi_tfidf_re_stop = mean_squared_error(y_test, new_predictions)
print(uni_bi_tfidf_re_stop)   

### [7] unigram + TFIDF + remove punctuation + most popularwords
import string
from collections import defaultdict
wordCount = defaultdict(int)
punctuation = set(string.punctuation)
for d in training:
  r = ''.join([c for c in d['review/text'].lower() if not c in punctuation])
  for w in r.split():
        wordCount[w] += 1

counts = [(wordCount[w], w) for w in wordCount]
counts.sort()
counts.reverse()

words = [x[1] for x in counts[:1000]]

def computeMSE(dataset):
    reviews = []
    for l in dataset:
        r = ''.join([c for c in l['review/text'].lower() if not c in punctuation])
        reviews.append(r)
    X = vectorizer.transform(reviews)
    word_index = vectorizer.vocabulary_
    y = list(map(float,[d['review/score'] for d in dataset] ))
    return X,y,reviews,word_index


reviews_train = []
for l in training:
    r = ''.join([c for c in l['review/text'].lower() if not c in punctuation])
    reviews_train.append(r)

vectorizer = TfidfVectorizer(token_pattern=u"(?u)\\b\\w+\\b")
X_train = vectorizer.fit_transform(reviews_train)
import pandas as pd
X_train_df_0 = pd.DataFrame(X_train[0].todense())
for l in range(113890):
    if X_train_df_0.loc[0,i] == 0:
        X_train_df_0.drop(columns = i)
word_index = vectorizer.vocabulary_
X_train_new = []
wordId = dict(zip(words,range(len(words))))

count = 0
for i in range(180000):
    feat = [0]*len(words)
    r = reviews_train[i].split()
    for word in r:
        if word in words:
            feat[wordId[word]] = X_train[count,][(0,word_index[word])]
    count +=1
    X_train_new.append(feat)

y_train = [d['review/score'] for d in training]
y_train = list(map(float, y_train))

X_valid,y_valid,reviews_valid,word_index_valid = computeMSE(validation)
X_valid_new = []
for i in range(180000):
    feat = [0]*len(words)
    r = reviews_valid[i].split()
    for word in r:
        if word in words:
            feat[wordId[word]] = X_valid[i,][(0,word_index_valid[word])]
    X_valid_new.append(feat)
    
    
para = [0.01, 0.1, 1, 10, 100]
for i in para:
    clf = linear_model.Ridge(i, fit_intercept=False)
    clf.fit(X_train_new, y_train)
    theta = clf.coef_
    predictions = clf.predict(X_valid_new)
    print(mean_squared_error(y_valid, predictions))

clf = linear_model.Ridge(1, fit_intercept=False)
clf.fit(X_train_new, y_train)
theta = clf.coef_
predictions = clf.predict(X_valid)
print(mean_squared_error(y_valid, predictions))


### EDA
#time period
time1 = []
for l in wholedata:
    time1.append(l['review/time'])

import time

time2 = []
for t in time1:
    timeStruct2 = time.gmtime(int(t))
    time2.append(time.strftime("%Y-%m", timeStruct2))

print('The time period is from ' + min(time2) + ' to ' + max(time2) + '.')
time2 = sorted(time2)
count_time = dict(Counter(time2))


x = []
y = []
for key,value in count_time.items():
    x.append(key)
    y.append(value)

import pandas as pd
d = {'time': x, 'count': y}
timedf = pd.DataFrame(data = d)

timedf['time'] = pd.to_datetime(timedf['time'])
timedf['year'] = pd.DatetimeIndex(timedf['time']).year

timedf.groupby('year').sum().plot(kind = 'bar')

topreview = timedf.sort_values('count',ascending=False).head(10)
topreview.plot(kind = 'bar', x ='time',y='count')


#Distribution of ratings
ratings = []
for l in training:
    ratings.append(l['review/score'])
ratingsnew = []
for l in ratings:
    ratingsnew.append(float(l))
    
mean_train = sum(ratingsnew)/len(ratingsnew)

y_valid = list(map(float,[d['review/score'] for d in validation]))
count = 0
for l in y_valid:
    count += (l - mean_train)**2
    
MSE_base = count/len(validation)
from collections import Counter

counts = dict(Counter(ratings))


total = sum(counts.values())
x = []
y = []
for key,value in counts.items():
    x.append(key)
    y.append(value*100/total)
    
plt.xlabel('Rating')
plt.ylabel('Percentage')
plt.title('Distribution of ratings')
plt.bar(x,y)


import string
from collections import defaultdict
wordCount = defaultdict(int)
punctuation = set(string.punctuation)
for d in wholedata:
  r = ''.join([c for c in d['review/text'].lower() if not c in punctuation])
  for w in r.split():
    if w not in stopwords:
        wordCount[w] += 1

counts = [(wordCount[w], w) for w in wordCount]
counts.sort()
counts.reverse()

words = [x[1] for x in counts[:1000]]
#wordcloud
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
# Create and generate a word cloud image:
text = ''
for word in words[:100]:
    text = text + ' ' + word

wholereviews = ''    
for d in wholedata:
    r = ''.join([c for c in d['review/text'].lower() if not c in punctuation])
    wholereviews = wholereviews + ' ' + r

text = " ".join(c for c in d['review/text'].lower() if not c in punctuation for d in wholedata)


wordcloud = WordCloud(stopwords =stopwords, background_color="white").generate(text)


# Display the generated image:
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


#plot 
from collections import Counter
y_test = [d['review/score'] for d in test]
y_test = list(map(float, y_test))
counts = dict(Counter(y_test))
total = sum(counts.values())
x1 = []
y1 = []
for key,value in counts.items():
    x1.append(key)
    y1.append(value)
    
plt.subplot(1,2,1)
plt.hist(y_test)
plt.title('Actual')
plt.grid()

lt_predictions = list(predictions)


for i in range(len(lt_predictions)):
    if lt_predictions[i] < 0:
        lt_predictions[i] =0
    elif lt_predictions[i] > 5:
        lt_predictions[i] = 5
        
new_predictions = []
for l in range(len(lt_predictions)):
    new_predictions.append(round(lt_predictions[l]))
plt.subplot(1,2,1)
plt.hist(y_test)
plt.title('Actual')
plt.grid()
plt.subplots_adjust(wspace =0.3, hspace =0)
plt.subplot(1,2,2)
plt.hist(new_predictions)
plt.title('Predictions')
plt.grid()