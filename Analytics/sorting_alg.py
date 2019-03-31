
import numpy as np
import pandas as pd
import time
import pickle

bs = []
date = []
typ = []
kind = []
Time = []

Type = ['political', 'social', 'educational', 'personal']


def bullshit():
    t = time.time()
    while t > 0:
        bullshit = input('') #input the data
        if bullshit == 'end': #if input is end terminate Program
            break
        else: #if not continue
            data = bullshit.split('#') #split the string into chuncks
            if len(data) < 1:
                bs.append('')
                date.append(pd.to_datetime(''))
                typ.append('')
            if type(data[0]) != str:
                bs.append('Unkown')
            else:
                bs.append(data[0]) #add the first element of data to bs list
            if len(data) < 2: #if the size of data is under two, one 1 input
                date.append('0') #add zero to date
            else:
                if type(data[2]) != str:
                    date.append(pd.to_datetime('np.nan'))
                else:
                    if data[1] == '': #if the second value in data is an empty string
                        x = pd.to_datetime('') #convert to datetime and store in date list
                        date.append(x)
                    else:
                        y = pd.to_datetime(data[1]) #store the date in the date list
                        date.append(y)
            if len(data) < 3: #if the size of data is under 3 ie 2 inputs
                typ.append('') #add an empty string
            else:
                if type(data[2]) != str:
                    typ.append('Unknown')
                else:
                    if data[2] == '': #if the data is an empty string add that
                        typ.append(data[2])
                    else:
                        if np.isin(data[2], Type) == True:
                            typ.append(data[2]) #just add the data
                        else:
                            typ.append('Unkown')
            if len(data) < 4:
                kind.append('')
            else:
                if data[3] == '':
                    kind.append(data[3])
                else:
                    kind.append(data[3])

    p = input('')
    print('end of %s dataset' % (p))
if __name__ == "__main__":
    bullshit()
