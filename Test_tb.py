#!/usr/bin/python

"""
ZetCode PyQt5 tutorial

This program creates a menubar. The
menubar has one menu with an exit action.

Author: Jan Bodnar
Website: zetcode.com
"""

import sys
from PyQt5.QtWidgets import QMainWindow, QAction, qApp, QApplication
from PyQt5.QtGui import QIcon


class Example(QMainWindow):

    def __init__(self):
        super().__init__()

        self.initUI()

    def initUI(self):

        exitAct = QAction('&Exit', self)
        exitAct.setShortcut('Ctrl+Q')
        exitAct.setStatusTip('Exit application')
        exitAct.triggered.connect(self.exit)

        saveAct = QAction('&Save', self)
        saveAct.setShortcut('Ctrl+S')
        saveAct.setStatusTip('Save')
        saveAct.triggered.connect(self.save)

        preferencesAct = QAction('&Preferences', self)
        preferencesAct.setShortcut('Ctrl+,')
        preferencesAct.setStatusTip('Open the Preferences window')
        preferencesAct.triggered.connect(self.preferences)

        aboutAct = QAction('&About', self)
        aboutAct.setShortcut('Ctrl+A')
        aboutAct.setStatusTip('About the program')
        aboutAct.triggered.connect(self.about)

        self.statusBar()

        menubar = self.menuBar()
        fileMenu = menubar.addMenu('&File')
        aboutMenu = menubar.addMenu('&Help')
        aboutMenu.addAction(aboutAct)
        fileMenu.addAction(exitAct)
        fileMenu.addAction(saveAct)
        fileMenu.addAction(preferencesAct)

        self.setGeometry(300, 300, 300, 200)
        self.setWindowTitle('Simple menu')
        self.show()

    def exit(self):
        sys.exit()
    def save(self):
        print("save")
    def about(self):
        print("about")
    def preferences(self):
        print("settings")

def main():
    app = QApplication(sys.argv)
    ex = Example()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
