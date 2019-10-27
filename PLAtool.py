import random
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np


class Pladata:
    def __init__(self, pn=10, nn=10, v=2):
        self.posnumber = pn  # 样本数目
        self.negnumber = nn  # 样本数目
        self.v = v  # 特征维数

    def linesplit(self):
        number_positive = 0
        number_negetive = 0
        sample = []
        lable = []
        w = []
        w0 = random.randint(-10, 10)
        for i in range(self.v):
            w.append(random.randint(-10, 10))

        while (number_positive < self.posnumber) or (number_negetive < self.negnumber):
            HX = w0
            x = []
            for i in range(self.v):
                x.append(random.uniform(-20, 20))
            for i in range(self.v):
                HX += w[i] * x[i]
            if HX > 0 and number_positive < self.posnumber:
                sample.append(x)
                lable.append(1)
                number_positive += 1
            elif HX < 0 and number_negetive < self.negnumber:
                sample.append(x)
                lable.append(-1)
                number_negetive += 1
        tw = []
        for j in range(self.v):
            tw1 = []
            tw1.append(w[j])
            tw.append(tw1)
        sample = np.array(sample)
        return sample, lable, tw, w0

    def addnoise(self, labels, change=0.1):
        source = []
        for i in range(len(labels)):
            rate = random.randint(0,100) / 100
            if rate < change:
                source.append(-1*labels[i])
            else:
                source.append(labels[i])
        return source


# 画图描绘
class Picture:
    def __init__(self, data, lable, num, v=2):
        self.fig = plt.figure("wankejia")
        plt.title('Perceptron Learning Algorithm', size=14)
        self.data = data
        self.lable = lable
        self.num = num
        self.v = v
        self.legend = []
        if v == 2:
            self._test2d()
        elif v == 3:
            self._test3d()

    def _test2d(self):
        plt.xlabel('x0', size=20)
        plt.ylabel('x1', size=20)
        for i in range(self.num):
            if self.lable[i] > 0:
                plt.scatter(self.data[i][0], self.data[i][1], s=50)
            else:
                plt.scatter(self.data[i][0], self.data[i][1], s=50, marker='x')

    def _test3d(self):
        self.ax = self.fig.add_subplot(111, projection='3d')
        for i in range(self.num):
            if self.lable[i] > 0:
                self.ax.scatter(self.data[i][0], self.data[i][1], self.data[i][2], color='r', s=50)
            else:
                self.ax.scatter(self.data[i][0], self.data[i][1], self.data[i][2], s=50, color='b', marker='x')
        self.ax.set_xlabel('x0')
        self.ax.set_ylabel('x1')
        self.ax.set_zlabel('x2')

    def _cal3d(self, w, b, l):
        X = np.arange(-20, 20, 0.5)
        Y = np.arange(-20, 20, 0.5)
        xdata, ydata = np.meshgrid(X, Y)
        zdata = (-b - w[0] * xdata - w[1] * ydata) / w[2]
        self.ax.plot_surface(xdata, ydata, zdata, color='g', cmap="coolwarm", antialiased=True)
        self.legend.append(l)

    def _cal2d(self, w, b, l):
        xdata = np.linspace(-20, 20, 100)
        ydata = (-b - w[0] * xdata) / w[1]
        plt.plot(xdata, ydata, label=l)
        self.legend.append(l)

    def AddSplit(self, w, b, l):
        if self.v == 2:
            self._cal2d(w, b, l)
        elif self.v == 3:
            self._cal3d(w, b, l)

    def Show(self):
        if self.v == 2:
            plt.legend(self.legend)
        plt.show()


class DrawLine:
    def __init__(self,l='PDA'):
        self.fig = plt.figure("wankejia")
        plt.title(l, size=14)
        self.legend = []

    def linechart(self,yData,name):
        xData = []
        for i in range(len(yData)):
            xData.append(i+1)
        plt.plot(xData, yData)
        self.legend.append(name)

    def Show(self):
        plt.legend(self.legend)
        plt.show()