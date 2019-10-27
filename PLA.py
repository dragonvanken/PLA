import numpy as np
import PLAtool as ptools
import time
import random
# 训练感知机模型


class Perceptron:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.w = np.zeros((x.shape[1], 1))  # 初始化权重，w1,w2均为0
        self.b = 0
        self.numsamples = self.x.shape[0]

    def _sign(self, w, b, x):
        y = np.dot(x, w) + b
        return int(y)

    def _update(self, label_i, data_i):
        tmp = label_i * data_i
        tmp = tmp.reshape(self.w.shape)
        # 更新w和b
        self.w = tmp + self.w
        self.b = self.b + label_i

    def PLAtrain(self):
        time_start = time.time()
        isFind = False
        account = 0
        count = 0
        while not isFind:
            count = 0
            account += 1
            for i in range(self.numsamples):
                tmpY = self._sign(self.w, self.b, self.x[i, :])
                if tmpY * self.y[i] <= 0:  # 如果是一个误分类实例点
                    count += 1
                    self._update(self.y[i], self.x[i, :])
            if count == 0:
                isFind = True
            if account >= 999:
                break
        time_end = time.time()
        profile = []
        profile.append(account)
        profile.append(time_end - time_start)
        profile.append(1 - count / self.numsamples)
        # print('PLA arg w and b: ', self.w, self.b)
        print('PLA iteration: ', account)
        print('PLA time cost: ', time_end - time_start, 's')
        print('PLA Hit: ', 1 - count / self.numsamples)
        return self.w, self.b, profile

    def _countwrong(self):
        count = 0
        err = []
        err_l = []
        for i in range(self.numsamples):
            tmpY = self._sign(self.w, self.b, self.x[i, :])
            if tmpY * self.y[i] <= 0:  # 如果是一个误分类实例点
                count += 1
                err.append(self.x[i, :])
                err_l.append(self.y[i])
        return count, err, err_l

    def Pockertrain(self, s=1000):
        time_start = time.time()
        max_w = self.w
        max_b = self.b
        less_error = self.numsamples
        for pocket in range(s):
            count, err, err_l = self._countwrong()
            if count < less_error:
                max_w = self.w
                max_b = self.b
                less_error = count
            if count > 0:
                r = random.randint(0, count-1)
                self._update(err_l[r], err[r])
            else:
                break
        time_end = time.time()
        profile = []
        profile.append(pocket)
        profile.append(time_end - time_start)
        profile.append(1 - less_error / self.numsamples)
        # print('Poket arg w and b: ', max_w, max_b)
        print('Poket iteration: ', pocket)
        print('Poket time cost: ', time_end - time_start, 's')
        print('Poket Hit: ', 1 - less_error / self.numsamples)
        return max_w, max_b, profile


if __name__ == '__main__':
    pnum = 100
    nnum = 100
    v = 2
    esipode = []
    ltime = []
    hit = []
    pesipode = []
    pltime = []
    phit = []
    for i in range(1):
        # 获得数据
        data = ptools.Pladata(pnum, nnum, v)
        samples, labels, tw, tb = data.linesplit()
        # 添加噪音
        noise = data.addnoise(labels)
        # 感知机学习
        # plaw, plab ,pro= Perceptron(x=samples, y=labels).PLAtrain()
        pplaw, pplab ,ppro= Perceptron(x=samples, y=noise).Pockertrain(100)
        #esipode.append(pro[0])
        #ltime.append(pro[1])
        # hit.append(pro[2])
        pesipode.append(ppro[0])
        pltime.append(ppro[1])
        phit.append(ppro[2])
        # 绘制
        Picture = ptools.Picture(samples, labels, pnum + nnum, v)
        # Picture.AddSplit(plaw, plab, 'PLA')
        Picture.AddSplit(pplaw, pplab, 'Pocket')
        Picture.Show()

    # 统计绘制

    # chart = ptools.DrawLine('100-epoch time')
    # chart.linechart(ltime, 'PLA')
    # chart.linechart(pltime, 'PocketPLA')
    # chart.Show()

    # chart = ptools.DrawLine('100-epoch hit')
    # chart.linechart(hit, 'PLA')
    # chart.linechart(phit, 'PocketPLA')
    # chart.Show()
