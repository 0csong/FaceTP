import sys
from PyQt5.QtGui import QIcon,QPixmap
from PyQt5 import QtGui
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication
from pythonProject import dd

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        label1 = QLabel(self)
        label1.move(173, 10)
        pixmap = QPixmap('ai.jpg')
        label1.setPixmap(pixmap)
        label1.resize(pixmap.width() + 20, pixmap.height() + 20)

        btn1 = QPushButton('&Start!', self)
        btn1.resize(300,50)
        btn1.move(150,300)
        btn1.clicked.connect(self.opennextscreen)

        self.setWindowTitle('Icon')
        self.setWindowIcon(QIcon('sm.jpg'))
        self.setWindowTitle('SangMyung Graduate UI')
        self.move(500, 300)
        self.resize(600, 400)
        self.show()
    def opennextscreen(self):
        widget.setCurrentIndex(widget.currentIndex()+1)


class PrivateWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
    def initUI(self):
        self.setWindowTitle('Icon')
        self.setWindowIcon(QIcon('sm.jpg'))
        self.setWindowTitle('SangMyung Graduate UI')

        btn1 = QPushButton('백준호', self)
        btn1.resize(60, 60)
        btn1.move(50, 30)

        btn2 = QPushButton('송영도', self)
        btn2.resize(60, 60)
        btn2.move(50, 100)

        btn3 = QPushButton('박신위', self)
        btn3.resize(60, 60)
        btn3.move(50, 170)

        btn4 = QPushButton('서현수', self)
        btn4.resize(60, 60)
        btn4.move(50, 240)

        btn5 = QPushButton('안병선', self)
        btn5.resize(60, 60)
        btn5.move(50, 310)

        label1 = QLabel('누구 코드?', self)
        label1.setAlignment(Qt.AlignCenter)
        label1.setStyleSheet("color:gray;"
                             "background-color:white ")
        label1.move(150,50)
        label1.resize(400,300)
        font=label1.font()
        font.setPointSize(20)
        font.setBold(True)
        label1.setFont(QtGui.QFont("Arial", 14, QtGui.QFont.Black))

        btn1.clicked.connect(lambda: self.code1(label1))
        btn2.clicked.connect(lambda: self.code2(label1))
        btn3.clicked.connect(lambda: self.code3(label1))
        btn4.clicked.connect(lambda: self.code4(label1))
        btn5.clicked.connect(lambda: self.code5(label1))

        self.move(500, 300)
        self.resize(600, 400)
        self.show()

    def code1(self,label):
        label.setText("백준호 코드 불러오는 중...")

    def code2(self, label):
        label.setText("송영도 코드 불러오는 중...")

    def code3(self, label):
        label.setText("박신위 코드 불러오는 중...")

    def code4(self, label):
        label.setText("서현수 코드 불러오는 중...")

    def code5(self, label):
        label.setText("안병선 코드 불러오는 중...")
        dd.dd.dd()

if __name__ == '__main__':
   app = QApplication(sys.argv)
   widget = QtWidgets.QStackedWidget()
   mainWindow=MainWindow()
   privateWindow=PrivateWindow()
   widget.addWidget(mainWindow)
   widget.addWidget(privateWindow)

   widget.setFixedWidth(600)
   widget.setFixedHeight(400)
   widget.show()
   app.exec_()