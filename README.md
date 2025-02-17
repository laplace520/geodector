# geodector
Geodetectors written in Python can be visualized and operated
before using the code , please install the needed library
-----------------------------------------------------------
import warnings
from typing import Sequence
from scipy.stats import f, levene, ncf, ttest_ind
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QFileDialog, QVBoxLayout, QWidget, QMessageBox, QComboBox, QTextEdit, QDialog
from PyQt5.QtCore import Qt 
-------------------------------------------------------------
The first function is differentiation and factor detection
The numbers in the figure represent the q values between the other two variables, and the larger the q, the stronger the joint influence of the two factors on the outcome variables.
The numbers in red indicate that there is a significant difference in the results of ecological testing
The last function is the detection of risk zones
第一个功能是分异及因子探测
图里面的数字表示的是另外两个变量之间的q值,q越大，说明两个因子对结果变量的共同影响比较强.
图中红色的数字表示生态检测结果是有显著性差异的
最后的功能就是风险区探测
