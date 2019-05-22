import numpy as np
import matplotlib.pyplot as plt
import sys

"""
Run the folloqing to get plots: python file.py filename1 filename2 filename3 filename4
"""

def plot(filename1, filename1):
    #read the data from the two files and store data in bins
    #and counts variables for plotting. Set x to the number of lines use
    #in the header of the files... anything that is not a number.
    data1 = np.genfromtxt(filename1, skip_header = x)
    data2 = np.genfromtxt(filename2, skip_header = x)
    #assuming the data is a two column file and genfromtxt understood
    bins1 = data1[:,0] #first column
    counts1 = data1[:,1] #second column
    bins2 = data2[:,0]
    counts2 = data2[:,1]
    #may need to slice data and take out headers and stuff

    #Convert data into floats
    try:
        bins1 = float(bins1)
        counts1 = float(counts1)
        bins2 = float(bins2)
        counts2 = float(counts2)
    except:
        print("Unable to convert to float")

    #open a new figure (use figsize = (x,y) to change size of figure)
    #and creates two bar plots stacked on top of one another
    plt.figure()
    plt.subplot(211)
    plt.bar(bins1, counts1)
    plt.subplot(212)
    plt.bar(bins2, counts2)
    plt.xlabel("x axis")
    plt.ylabel("y axis")
    plt.savefig("fig1.png")
    plt.show()

def main():
    #pass files directly to script
    file1 = sys.argv[1]
    file2 = sys.argv[2]
    file3 = sys.argv[3]
    file4 = sys.argv[4]

    #Plot and show...
    plot(file1,file2)
    plot(file3,file4)



if __name __ == "__main__":
    main()
