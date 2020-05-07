import os
import sys
import time
import datetime
import pyodbc
import smtplib
import python-docx
import SETTINGS

def CALIBRATE_THERMOMETRY():
    input("Press enter to being")
    REF_lst = []
    ACT_lst = []
    tol = SETTINGS.__TEMP_TOL__
    while True:
        cmd = input("Press enter to do another point of q to exit...")
        if cmd.upper() == 'Q':
            break
        else:
            REF = input('Enter reference temperature (C): ')
            ACT = input('Enter actual temperature (C): ')
            try:
                if abs(float(REF) - float(ACT)) > tol:
                    pass
                else:
                    pass
            except Exception as e:
                print(e)
