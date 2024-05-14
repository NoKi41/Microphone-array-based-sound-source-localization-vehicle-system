import time
from matplotlib import pyplot as plt
import numpy as np
import pyaudio
import wave

# 设置录音参数
CHUNK = 2048         # 单次录音的帧数
FORMAT = pyaudio.paInt16  # 录音格式
CHANNELS = 8         # 录音通道数
RATE = 48000         # 采样率

# 初始化
p = pyaudio.PyAudio()   

# 打开录音设备
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK,
                input_device_index=0)

# 传感器阵列坐标
# mics = np.array([[ 4.00000000e-02, 0.00000000e+00],     # mic1  # 0
#                 [-2.00000000e-02, -3.46410162e-02],    # mic2 # 4
#                 [ 2.00000000e-02, 3.46410162e-02],     # mic3 # 1
#                 [ 2.00000000e-02, -3.46410162e-02],    # mic4 # 5
#                 [-2.00000000e-02, 3.46410162e-02],     # mic5 # 2
#                 [ 0.00000000e+00, 0.00000000e+00],    # mic6 # 6
#                 [-4.00000000e-02, 0.00000000e+00]      # mic7 # 3
#                 ])

MICS_theta = np.pi * np.array([0, 4, 1, 5, 2, 0, 3]) / 3  # 各麦克风与x轴的夹角
R = 0.04  # 麦克风阵列半径
c = 343  # 声速

f_CBF_start = 500       # CBF起始频率
f_CBF_end = 1000        # CBF终止频率
f_CBF_F = np.arange(f_CBF_start, f_CBF_end + 1)   # CBF频域处理频段
d_alpha = 1  # CBF 角度间隔
theta = np.arange(-180, 180 + d_alpha, d_alpha)    # 角度范围

# 初始化结果矩阵
CBF_ANS = np.zeros((len(theta)))

# 构建导向矢量
w_f = np.ones((len(f_CBF_F), 7, len(theta)), dtype=np.complex128)

for i in range(len(f_CBF_F)):
    for j in range(7):
        if j != 5:
            # TODO:麦克风方向矢量
            w_f[i, j, :] = np.exp(-1j * (2 * np.pi * f_CBF_F[i] * R * np.cos(np.deg2rad(theta) - MICS_theta[j])) / c)

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

    # 归一化处理
    out = out / max(out)
    # 转化为dB
    out = np.log10(out)

    return out

print("初始化完毕...")

# 主循环
while True:
    # 开始录音
    # print("开始录音...")
    frames = []

    for i in range(int(RATE / CHUNK)):
        data = stream.read(CHUNK, exception_on_overflow=False)
        frames.append(data)

    # data初始化为形状的数组(8,1)
    data = np.zeros((8, 0))

    # 数据整形，帧连续但mic索引不连续，分块再转置
    for i in range(len(frames)):
        data = np.concatenate((data, np.frombuffer(frames[i], dtype=np.int16).reshape((CHUNK, -1)).T), axis=1)

    # print(data.shape)

    # time_start = time.time() 
    # 计算CBF_ANS
    CBF_ANS = CBF_PC(data[:7, :RATE], d_alpha, RATE, f_CBF_start, f_CBF_end, theta, w_f)

    # 找到最大值及其索引
    # max_value = np.max(CBF_ANS)
    max_index = np.argmax(CBF_ANS)

    # 获取最大值对应的角度值
    max_angle = theta[max_index]

    # print("计算时间：\t", time.time() - time_start)

    print("预测朝向：\t", max_angle)
    # TODO:串口发送预测朝向

    # plt.figure()
    # ax = plt.subplot(111, projection='polar')  # 设置子图为极坐标形式
    # ax.plot(np.linspace(0, 2*np.pi, len(theta), endpoint=False), CBF_ANS)  # 将角度转换为弧度并绘制极坐标图
    # ax.set_title('Circular_CBF (Frequence-domain) spatial spectrum(Polar plot)')
    # plt.show()
