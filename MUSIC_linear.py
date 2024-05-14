import numpy as np
import matplotlib.pyplot as plt
import datetime

pi = np.pi
sin = np.sin
cos = np.cos

# 参数设置
N = 7  # 阵元个数

SOURCES_theta_angle = np.random.randint(-90, 90, 1) # 来波方向(角度) np.random.randint(1, 5)
# SOURCES_theta_angle = np.array([66]) # 来波方向(角度) np.random.randint(1, 3)
print('声源方向：', SOURCES_theta_angle)
SOURCES_theta = SOURCES_theta_angle * pi / 180  # 来波方向(弧度)

M = len(SOURCES_theta)  # 信源个数
SNR = 10  # 信号信噪比dB
K = 2560  # 总采样点
delta_d = 0.07  # 阵元间距
f = 500  # 信号源频率
c = 340  # 声速
num = 720 # 角度采样点

# 阵列配置
d = np.arange(0, N * delta_d, delta_d)

A = np.exp(1j * 2 * pi * (f / c) * np.outer(d, np.sin(0 - SOURCES_theta)))  # 接收信号方向向量
# print(A.shape)

# 生成信号
S = np.random.randn(M, K)  # 阵列接收到来自声源的信号
X = A @ S  # 最终接收信号，是带有方向向量的信号

X_real = np.real(X) # 对X取实部
X_abs = abs(X) # 对X取幅值
# print(X.shape)

# 在信号中添加高斯噪声
X1 = X + np.random.normal(0, 10**(-SNR/20), X.shape)
# X1= X_real + np.random.normal(0, 10**(-SNR/20), X.shape)
# X1= X_abs + np.random.normal(0, 10**(-SNR/20), X.shape)

t = datetime.datetime.now()

# 协方差矩阵
Rx = (X1 @ X1.conj().T) / K

# 特征值分解
D, Ev = np.linalg.eig(Rx)
EVA = np.sort(np.abs(D))[::-1]  # 将特征值排序
EV = Ev[:, np.argsort(np.abs(D))[::-1]]  # 对应特征矢量排序

# 噪声子空间
En = EV[:, M:N]
# print(En.shape)

# 计算MUSIC谱
p_music = np.zeros(num)
angles = np.linspace(-90, 90, num)  # This includes the endpoint, so the angles will be from -90 to 90 inclusive
for i, angle in enumerate(angles):
    theta_m = angle * np.pi / 180
    a = np.exp(-1j * 2 * np.pi * (f / c) * d * np.sin(theta_m))
    # print(a.shape)
    p_music[i] = 1 / np.abs(a.conj().T @ En @ En.conj().T @ a)

print('运算时间：', datetime.datetime.now() - t)

# 归一化,分贝处理
p_music_db = 10 * np.log10(p_music / np.max(p_music))

# 绘制空间谱
plt.figure()
plt.plot(angles, p_music_db)
plt.xlabel('Angle')
plt.ylabel('dB')

# 标注SOURCES_theta_angle中的坐标线
for u in SOURCES_theta_angle:
    plt.axvline(x = u, color='g', linestyle='--')
plt.title('Linear_MUSIC spatial spectrum')
plt.show()
