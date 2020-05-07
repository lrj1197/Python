import os
import sys
import time
import datetime
import pyodbc
import smtplib

import TEMPERATURE
import VACUUM

def main():
    cmd = input("Enter the type of calibration to be done: ")
    if cmd.upper().replace('','_') == "ROTARY_HEAD":
        return
    elif cmd.upper().replace('','_') == "VACUUM":
        VACUUM.CALIBRATE_VACUUM()
    elif cmd.upper().replace('','_') == "TEMPERATURE":
        TEMPERATURE.CALIBRATE_THERMOMETRY()
    else:
        main()
