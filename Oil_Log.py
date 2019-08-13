import datetime as dt
import os
import sys
import pandas as pd

pd.options.mode.chained_assignment = None

def quit():
    sys.exit()

def log():
    try:
        if os.path.isfile("OIL_LOG.csv") == True:
            f = open("OIL_LOG.csv", 'a')
        else:
            f = open("OIL_LOG.csv", 'w')
            f.write("Bowl,QTY,Desc.,Date in,Est Date out,Date out,User\n")
        bowl = input("Enter Bowl Number: ")
        qty = input("Quantity: ")
        desc = input("Product: ")
        date_in = dt.datetime.now().strftime("%m/%d/%Y")
        date_out = input("Enter the estameted date out: ")
        date_out_real = 'NAN'
        usr = input("Enter Name: ")
        print("{},{},{},{},{},{},{}\n".format(bowl, qty, desc, date_in, date_out, date_out_real, usr))
        f.write("{},{},{},{},{},{},{}\n".format(bowl, qty, desc, date_in, date_out, date_out_real, usr))
        f.close()
    except Exception as e:
        print(e)

def view():
    try:
        data = pd.read_csv("OIL_LOG.csv")
        print("Recent entries are...")
        print(data.head())
    except Exception as e:
        print(e)

def close():
    try:
        bowl = input("Enter the bowl you wish to close out: ")
        date = input("Enter the date the units went in (MM/DD/YYYY): ")
        usr = input("Enter name: ")
        dataraw = pd.read_csv("OIL_LOG.csv")
        data = dataraw[dataraw['Bowl'] == bowl]
        data = data[data["Date in"] == date]
        data = data[data["User"] == usr]
        idx = data.index
        if dataraw["Date out"][idx] != "NAN":
            print("Already closed out")
        else:
            dataraw["Date out"][idx] = dt.datetime.now().strftime("%m/%d/%Y")
            dataraw.to_csv("OIL_LOG.csv",columns = ['Bowl','QTY','Desc.','Date in','Est Date out','Date out','User'], index = False)
            print("Data was succesfully logged")
    except Exception as e:
        print(e)


def main():
    print("***Oil Log***")
    print("Authur: Lucas Jameson")
    print("Version 0.0")
    while True:
        cmd = input("Enter command or type help: ")
        if cmd == 'help':
            print("help")
        elif cmd == 'quit':
            print("bye bye for now :)")
            quit()
        elif cmd == 'log':
            log()
        elif cmd == "close":
            close()
        elif cmd == 'view':
            view()
        else:
            print("bad command")


if __name__ == "__main__":
    main()
