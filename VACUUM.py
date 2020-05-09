import os
import sys
import time
import datetime
import pyodbc
import smtplib
import SETTINGS
import FUNCTIONS
"""
This handels the vacuum part of the program
"""


def GET_VAC_VALUE(dev):
    values = []
    smaples = SETTINGS.SAMPLES
    max_val = SETTINGS.MAX_VAC_PRESSURE
    if type(dev) == type([]):
        values_pair = []
        for i in range(len(dev)):
            for _ in range(samples):
                dev.write(''.encode())
                time.sleep(1/samples)
                val = dev.readline().decode('utf-8')
                try:
                    val = float(val)
                    if val < max_val:
                        values_pair.append(val)
                    else:
                        pass
                except Exception as e:
                    print(e)
            values.append(values_pair)
        return values
    elif type(dev) == type(''):
        for _ in range(samples):
            dev.write(''.encode())
            time.sleep(1/samples)
            val = dev.readline().decode('utf-8')
            try:
                val = float(val)
                if val < max_val:
                    values.append(val)
                else:
                    pass
            except Exception as e:
                print(e)
        return values
    else:
        pass

def CALIBRATE_VACUUM_INIT():
    com = 'com1'
    baud = 9600
    bytes = 8
    parity = 'N'
    stopbits = 1
    dev = serial.Serial(com,baud,bytes,parity,stopbits,timeout=1)
    return  dev

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
            ACT = input('Enter actual vacuum (mtorr): ')
            REF = input('Enter reference vacuum (mtorr): ')
            GET_VAC_VALUE(dev)
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
