import numpy as np
import random

# 进行GCC_PHAT操作
def gcc_phat(s1, s2):

    f1 = np.fft.fft(s1)
    f2 = np.fft.fft(s2)

    G = np.conj(f1) * f2

    G1 = np.conj(f1) * f1
    G2 = np.conj(f2) * f2

    # 相位变换系数矩阵，即最功率谱中每一项就abs
    # w = 1
    
    # w = 1 / np.abs(G) # phat

    w = 1 / np.sqrt(G1 * G2) # scot

    G = G * w

    R = np.fft.ifft(G)

    # 找到R的最大值对应的索引编号
    max_idx = np.argmax(np.abs(R))

    return np.abs(R), max_idx # 输出相位变化后的互相关函数，和峰值索引值

# ——————————————————————————————————————————————————————————————————————————————————————————————————
# # 测试
# len = 500

# # 随机函数
# a1 = (np.random.random(size=len) - 0.5) * 5

# # 变频正弦函数
# a1 = 2.5 * np.sin(np.linspace(0, 10, len)**3)

# # 时间延迟98，a2的前23位是0，后面是a1，长度也为len
# delay = random.randint(1, len/2-1)
# a2 = np.zeros(len)
# a2[delay:len] = a1[0:(len-delay)]

# # 加随机高斯白噪声
# # a1 += 1 * np.random.normal(size=len)
# # a2 += 1 * np.random.normal(size=len)

# res1,idx1 = gcc_phat(a1, a2)

# # print((res1[idx1-1]), (res1[idx1]), (res1[idx1+1]))

# print(idx1, delay, (idx1 == delay))

# import matplotlib.pyplot as plt

# plt.figure()
# plt.subplot(3,1,1)
# plt.plot(a1, color='green')
# plt.title('Original Signal')

# plt.subplot(3,1,2)
# plt.plot(a2, color='orange')
# plt.axvline(x=delay, color='red', linestyle='--') # 标注delay的位置，红色虚线
# plt.title('Delayed Signal')

# plt.subplot(3,1,3)
# plt.plot((res1))
# plt.axvline(x=idx1, color='red', linestyle='--') # 标注idx1的位置，红色虚线
# plt.title('GCC-PHAT Result')


# res1,idx1 = gcc_phat(a2, a1)

# print(idx1, delay, (idx1 + delay == len))

# import matplotlib.pyplot as plt

# plt.figure()
# plt.subplot(3,1,1)
# plt.plot(a2, color='green')
# plt.title('Original Signal')

# plt.subplot(3,1,2)
# plt.plot(a1, color='orange')
# plt.axvline(x=delay, color='red', linestyle='--') # 标注delay的位置，红色虚线
# plt.title('Delayed Signal')

# plt.subplot(3,1,3)
# plt.plot((res1))
# plt.axvline(x=idx1, color='red', linestyle='--') # 标注idx1的位置，红色虚线
# plt.title('GCC-PHAT Result')

# plt.show()