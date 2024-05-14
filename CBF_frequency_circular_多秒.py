import numpy as np
import matplotlib.pyplot as plt

def CBF_PC(l, t_detal, fs, f_start, f_end, theta, w_f):
    # 参数设定
    f_CBF_F = np.arange(f_start, f_end + 1, 1 / t_detal)
    f_N = len(f_CBF_F)
    th_N = len(theta)

    out_theta = np.zeros((f_N, th_N), dtype=np.complex128)     # 存储计算结果

    # 如果信号长度小于给定值，在后面补0
    if len(l[0]) < t_detal * fs:
        # l = np.hstack((l, np.zeros((len(l), t_detal * fs - len(l[0])))))
        l = np.concatenate((l, np.zeros((l.shape[0], t_detal * fs - l.shape[1]))), axis=1)
    
    # 取出各个通道在当前时间段内的信号值
    detal = l[:, :int(t_detal * fs)]
    # 做各个通道的FFT
    detal_f = np.fft.fft(detal, axis=1) / detal.shape[1]
    # 取出关注的频段
    detal_f2 = detal_f[:, int(f_start * t_detal):int(f_end * t_detal) + 1]

    # 遍历频率
    # for f_num in range(f_N):
    #     # 遍历角度
    #     for theta_num in range(th_N):
    #         temp = np.dot(w_f[f_num, :, theta_num], detal_f2[:, f_num])
    #         out_theta[f_num, theta_num] = np.dot(temp, temp)

    for f_num in range(f_N): # 加速算法 快很多
        w_temp = w_f[f_num]
        temp = np.dot(detal_f2[:, f_num], w_temp)
        out_theta[f_num, :] = np.abs(temp) ** 2

    # python中需要指定行求和还是列求和，不然就会全部相加为一个标量
    out = np.sum(np.abs(out_theta), axis=0)
    # print(np.shape(out))
    return out


fs = 5000       # 采样频率
time = 2       # 采样时间
t = np.arange(0, fs * time) / fs
N = 6         # 阵元数量
c = 1510        # 声速
f = 700         # 信号的频率
R = 0.1        # 接收器半径
theta0 = 33.48    # 仿真目标角度
print(f'目标角度为：{theta0}')

ArraySignal = np.zeros((N, fs * time))
SNR = 0

for n in range(N):
    distance = R * np.cos(np.deg2rad(theta0) - (n * 2 * np.pi / N))
    delay_time = distance / c
    ArraySignal[n, :] = 100 * np.sin(2 * np.pi * f * (t - delay_time))
    ArraySignal[n, :] += np.random.normal(0, 20**(-SNR/20), fs * time)

f_CBF_start = 100       # CBF起始频率
f_CBF_end = 1000        # CBF终止频率
f_CBF_F = np.arange(f_CBF_start, f_CBF_end + 1)   # CBF频域处理频段
d_alpha = 0.5  # CBF 角度间隔
theta = np.arange(-180, 181, d_alpha)    # 角度范围

CBF_ANS = np.zeros((time, len(theta)))

# 阵列配置
MICS_theta_angle = np.linspace(0, 360, N, endpoint=False, dtype=np.float64) # 均匀圆阵麦克风方向(角度)
MICS_theta = MICS_theta_angle * np.pi / 180  # 麦克风方向(弧度)

# python中要定义数组的形状，因为携带复数所以dtype
w_f = np.zeros((len(f_CBF_F), N, len(theta)), dtype=np.complex128)


for i in range(len(f_CBF_F)):
    # TODO:导向矢量
    for j in range(N):
        # TODO:麦克风方向矢量
        w_f[i, j, :] = np.exp(1j * (2 * np.pi * f_CBF_F[i] * R * np.cos(np.deg2rad(theta) - MICS_theta[j])) / c)

    # w_f[i, :, :] = np.exp(1j * (2 * np.pi * f_CBF_F[i] * R * np.cos(np.deg2rad(theta) - MICS_theta)) / c)

    # A = np.exp(1j * 2 * pi * (f / c) * R * cos(SOURCES_theta_array - MICS_theta_array)) 

# 一次处理1s的数据
for j in range(time):
    print(j)
    CBF_ANS[j, :] = CBF_PC(ArraySignal[:, (j * fs):((j + 1) * fs)], d_alpha, fs, f_CBF_start, f_CBF_end, theta, w_f)

t2 = np.arange(1, time + 1, 1)


# 绘制彩色图
plt.figure(figsize=(8, 3))
plt.pcolor(theta, t2[0:time], CBF_ANS, shading='auto')
plt.title('CBF Freqency Domain DOA (SNR={}dB)'.format(SNR), fontsize=20)
plt.xlabel('Angle(°)')
# plt.ylabel('Time(s)')
plt.yticks([])
plt.axis([np.min(theta), np.max(theta), np.min(t2[0:time]), np.max(t2[0:time])])
# plt.grid(True)
plt.colorbar()
plt.show()

# 绘制第1秒的各个角度能量值
plt.figure()
p = 1
plt.plot(theta, CBF_ANS[p, :])  # 在横坐标上使用角度值 theta
plt.xlabel('Angle(°)')
plt.ylabel('')
plt.title('CBF Freqency Domain DOA (SNR={})'.format(SNR))
# plt.title('第1秒的各个角度能量值')
plt.show()

# 找到最大值及其索引
max_value = np.max(CBF_ANS[p, :])
max_index = np.argmax(CBF_ANS[p, :])

# 获取最大值对应的角度值
max_angle = theta[max_index]

# 显示最大值及其对应的角度值
print(f'对应的角度值为：{max_angle}')