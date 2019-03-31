
x = np.linspace(-30,0,10)
y_pred = 27.8079 * x + 882.854

index = []
y_real = []

val = input('')

val = val.split(' ')


index.append(int(val[0]))
y_real.append(float(val[1]))
#y_real = [float(input(''))]

import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
import numpy as np

x = np.asarray(x)
index = np.asarray(index)
y_pred = np.asarray(y_pred)
y_real = np.asarray(y_real)


lm = LinearRegression()
lm.fit(index.reshape(-1,1), y_real)
print(lm.intercept_ ,lm.coef_)

plt.figure()
plt.plot(x,lm.coef_ * x + lm.intercept_ )
plt.scatter(index, y_real)
plt.savefig("Compound Gauge Calibration")
plt.show()

# 27.8079, 882.854 Karl's model
# 24.47877994, 759.123952096, Luke's model
import time


# Run this cell as is
import time
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
'''
Gnerate dataset
'''
# Run this cell as is
quotes = [
    "Going outside is highly overrated.",
    "You'd be amazed how much research you can get done when you have no life whatsoever.",
    "One person can keep a secret, but not two.",
    "No one in the world gets what they want and that is beautiful.",
    "A river of words flowed between us.",
    "No matter where you go, there you are.",
    "The farther I went, the more confident I became.",
    "I wasn't some dilettante.",
    "We'll do our best.",
    "We spread across the entire planet like an unstoppable virus.",
    "D&D",
    "Time to make the doughnuts.",
    "I was too weird, even for the weirdos.",]

def prompt_quote():
    random.shuffle(quotes)
    while True:
        start = time.time()
        s = input('%s\n' % quotes[0])
        duration = time.time() - start
        if s == quotes[0]:
            break
        print('not correct, try again')
    return quotes[0], duration

# Run this cell as is
def generate_dataset():
    N = 50
    df = pd.DataFrame(columns=["quote", "duration"])
    for _ in range(N):
        quote, duration = prompt_quote()
        df = df.append({"quote": quote, "duration":  duration}, ignore_index=True)
    return df

# Run this cell as is
df = generate_dataset()

df.to_csv('cps-analysis.csv')

'''
analysis part
'''
df = pd.read_csv('cps-analysis.csv')

#functions not working correctly

def quote_length(quote):
    length = []
    for i in range(len(df)):
        leng = len(df.quote[i])
        length.append(leng)
    return length

def num_vowels(quote):
    sumv = []
    for i in range(len(df)):
        num = list(df.quote[i])
        vowels = list('aeiouAEIOU')
        sum_vowels = sum(np.isin(num, vowels))
        sumv.append(sum_vowels)
    return sumv

def num_consonants(quote):
    sum_consonants = []
    for i in range(len(df)):
        num = list(df.quote[i])
        vowels = list('aeiouAEIOU')
        letters = list(string.ascii_lowercase) + list(string.ascii_uppercase)
        consonants = [x for x in letters if x not in vowels]
        sump = sum(np.isin(num, consonants))
        sum_consonants.append(sump)
    return sum_consonants

def special_keys(quote):
    sum_punc = []
    for i in range(len(df)):
        num = list(df.quote[i])
        punc = list(string.punctuation)
        sump = sum(np.isin(num, punc))
        sum_punc.append(sump)
    return sum_punc

def right_keyboard(quote):
    right = []
    for i in range(len(df)):
        num = list(df.quote[i])
        r = list('yhbujnikmopl')
        rig = sum(np.isin(num, r))
        right.append(rig)
    return right

def left_keyboard(quote):
    left = []
    for i in range(len(df)):
        q = list(df.quote[i])
        l = list('tgvrfcedxwszqa')
        s = np.sum(np.isin(q, l))
        left.append(s)
    return left


df['special_keys'] = pd.DataFrame(special_keys(df.quote))
df['left_keyboard'] = pd.DataFrame(left_keyboard(df.quote))
df['right_keyboard'] = pd.DataFrame(right_keyboard(df.quote))
df['quote_length'] = pd.DataFrame(quote_length(df.quote))
df['num_vowels'] = pd.DataFrame(num_vowels(df.quote))
df['num_consonants'] = pd.DataFrame(num_consonants(df.quote))

df.to_pickle('CPS_1.pkl')
len(df.quote)

X = df.drop(['duration', 'quote', 'Unnamed: 0', 'Unnamed: 0.1'], axis =1) # modify this to make X a dataframe with all the fields in df except "duration"
y = df.duration # modify this to make y a dataframe with just the field "duration"

# this function randomly splits the dataset for training and testing/validation
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=0)

Xtr.head()

x_new = np.asanyarray(Xtr)
ytr_1 = np.asarray(ytr)
y_new = ytr_1.reshape(-1,1)
lm = LinearRegression()
lm.fit(x_new, y_new)
print(lm.intercept_ ,lm.coef_)

xtr_1 = np.asanyarray(Xtr.quote_length)
ytr_1 = np.asarray(ytr)
y_new = ytr_1.reshape(-1,1)
lm = LinearRegression()
lm.fit(xtr_1.reshape(-1,1), y_new)
print(lm.intercept_ ,lm.coef_)

plt.scatter(ytr_1,Xtr.quote_length)
plt.plot(x,  -0.52734637 * x + lm.intercept_ , 'r--')
plt.show()
#get the model coeff and intercepts.

x_new = np.asanyarray(Xte.quote_length)
x_new_test = x_new.reshape(-1,1)
ytr_1_test = np.asarray(yte)
y_new_test = ytr_1_test.reshape(-1,1)
lm = LinearRegression()
lm.fit(x_new_test, y_new_test)
print(lm.intercept_ ,lm.coef_)
[-16.54578507] [[ 5.84321427  5.36894305  4.60684593 -6.28746073 -5.62109229  0.76335249]]
x = np.linspace(0,35,100)

plt.scatter(y_new_test,x_new_test)
plt.plot(x,  0.76335249* x + lm.intercept_ , 'r--')
plt.show()


[-0.34297832] [[ 1.42521573  1.43381365  1.87719906 -0.15917265 -0.68377049 -0.52734637]]

xtr_1 = np.asarray(Xtr.quote_length)
xtr_2 = np.asarray(Xtr.num_vowels)
xtr_3 = np.asarray(Xtr.num_consonants)
xtr_4 = np.asarray(Xtr.special_keys)
xtr_5 = np.asarray(Xtr.right_keyboard)
xtr_6 = np.asarray(Xtr.left_keyboard)
ytr_1 = np.asarray(ytr)

lm = LinearRegression()
lm.fit(ytr_1.reshape(-1,1), xtr_1)
print(lm.intercept_ ,lm.coef_)
print(lm.score(ytr_1.reshape(-1,1), xtr_1))

lm.fit(ytr_1.reshape(-1,1), xtr_2)
print(lm.intercept_ ,lm.coef_)
print(lm.score(ytr_1.reshape(-1,1), xtr_2))

lm.fit(ytr_1.reshape(-1,1), xtr_3)
print(lm.intercept_ ,lm.coef_)
print(lm.score(ytr_1.reshape(-1,1), xtr_3))

lm.fit(ytr_1.reshape(-1,1), xtr_4)
print(lm.intercept_ ,lm.coef_)
print(lm.score(ytr_1.reshape(-1,1), xtr_4))

lm.fit(ytr_1.reshape(-1,1), xtr_5)
print(lm.intercept_ ,lm.coef_)
print(lm.score(ytr_1.reshape(-1,1), xtr_5))

lm.fit(ytr_1.reshape(-1,1), xtr_6)
print(lm.intercept_ ,lm.coef_)
print(lm.score(ytr_1.reshape(-1,1), xtr_6))


x = np.linspace(0,35,100)
plt.scatter(ytr_1,xtr_1)
plt.plot(x, 2.7261493 * x - 2.52387305 , 'r--')
plt.show()



quote = "You're evil, you know that?"

qu = list(quote)
vowels = list('aeiouAEIOU')
letters = list(string.ascii_lowercase) + list(string.ascii_uppercase)
consonants = [x for x in letters if x not in vowels]
punc = list(string.punctuation)
r = list('yhbujnikmopl')
l = list('tgvrfcedxwszqa')

#right_key
x1 = sum(np.isin(qu, r))
#left_key
x2 = np.sum(np.isin(qu, l))
#punctuation
x3 = sum(np.isin(qu, punc))
#consonants
x4 = sum(np.isin(qu, consonants))
#vowels
x5 = sum(np.isin(qu, vowels))
#length_quote
x6 = len(list(quote))

right_key
left_key
punctuation
consonants
vowels
length_quote
#nc,nv,sk,lk,rk,ql

def Pred(x1,x2,x3,x4,x5,x6):
    p = - 0.34297832 + 1.42521573 * x1 + 1.43381365 * x2 + 1.87719906 * x3 - 0.15917265 * x4 - 0.68377049 * x5 - 0.52734637 * x6
    return p
Pred(x1,x2,x3,x4,x5,x6)

while True:
    start = time.time()
    s = input('%s\n' % quote)
    duration = time.time() - start
    if s == quote:
        break
    print('not correct, try again')
print(duration)
