import os
import sys
import time
import datetime
import pyodbc
import smtplib
import SETTINGS
import FUNCTIONS

def CALIBRATE_VACUUM():
    input("Press enter to being")
    REF_lst = []
    ACT_lst = []
    tol = SETTINGS.__VAC_TOL__
    while True:
        cmd = input("Press enter to do another point of q to exit...")
        if cmd.upper() == 'Q':
            break
        else:
            REF = input('Enter reference vacuum (mtorr): ')
            ACT = input('Enter actual vacuum (mtorr): ')
            try:
                if abs(float(REF) - float(ACT)) > tol:
                    pass
                else:
                    pass
            except Exception as e:
                print(e)
    cmd = input("send report to QE?[y/n] ")
    while True:
        if cmd.upper() == 'Y':
            FUNCTIONS.send_report(FILE)
            break
        elif cmd.upper() == 'N':
            break
        else:
            cmd = input("Invalid repsonse. send report to QE?[y/n] ")
