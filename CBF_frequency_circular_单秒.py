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

    # 归一化处理
    out = out / max(out)
    # 转化为dB
    out = 10 * np.log10(out)

    return out


fs = 48000       # 采样频率
t = np.arange(0, fs) / fs
N = 6         # 阵元数量
c = 1510        # 声速
f = 700         # 信号的频率
R = 0.04        # 接收器半径
theta0 = np.random.uniform(-180, 180)    # 仿真目标角度
print(f'目标角度为：{theta0}')

ArraySignal = np.zeros((N, fs))

for n in range(N):
    distance = R * np.cos(np.deg2rad(theta0) - (n * 2 * np.pi / N))
    delay_time = distance / c
    ArraySignal[n, :] = 100 * np.sin(2 * np.pi * f * (t - delay_time))
    # 添加AWGN噪声
    # ArraySignal[n, :] += np.random.normal(0, 1, fs) * 200

f_CBF_start = 500       # CBF起始频率
f_CBF_end = 1000        # CBF终止频率
f_CBF_F = np.arange(f_CBF_start, f_CBF_end + 1)   # CBF频域处理频段
d_alpha = 0.5  # CBF 角度间隔
theta = np.arange(-180, 181, d_alpha)    # 角度范围

CBF_ANS = np.zeros((len(theta)))

# 阵列配置
MICS_theta_angle = np.linspace(0, 360, N, endpoint=False, dtype=np.float64) # 均匀圆阵麦克风方向(角度)

# MICS_theta = np.pi * np.array([0, 4, 1, 5, 2, 0, 3]) / 3  # 各麦克风与x轴的夹角
# print(f'麦克风角度为：{MICS_theta}')

MICS_theta = MICS_theta_angle * np.pi / 180  # 麦克风方向(弧度)
# print(f'麦克风角度为：{MICS_theta}')


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
CBF_ANS = CBF_PC(ArraySignal[:, :fs], d_alpha, fs, f_CBF_start, f_CBF_end, theta, w_f)


# 找到最大值及其索引
max_value = np.max(CBF_ANS)
max_index = np.argmax(CBF_ANS)

# 获取最大值对应的角度值
max_angle = theta[max_index]

# # 绘制各个角度能量值
# plt.figure(4)
# plt.plot(theta, CBF_ANS)  # 在横坐标上使用角度值 theta
# plt.xlabel('Angle(°)')
# plt.ylabel('')
# # plt.title('第1秒的各个角度能量值')
# plt.show()

plot_polar = False  # 是否绘制极坐标图 

# 绘制空间谱
if plot_polar:        
    plt.figure()
    ax = plt.subplot(111, projection='polar')  # 设置子图为极坐标形式
    ax.plot(np.linspace(0, 2*np.pi, len(theta), endpoint=False), -CBF_ANS)  # 将角度转换为弧度并绘制极坐标图
    ax.set_title('Circular_CBF (Frequence-domain) spatial spectrum(Polar plot)')
    plt.show()

    # 使用极坐标形式
else:
    plt.figure()
    plt.plot(theta, CBF_ANS)
    # 标记最大值处
    plt.plot(max_angle, max_value, 'ro')
    plt.text(max_angle, max_value, f'{max_angle}°', ha='center', va='bottom', fontsize=10)
    plt.xlabel('Angle(°)')
    plt.ylabel('dB')
    plt.xlim(-180, 180)
    plt.title('Circular_CBF (Frequence-domain) spatial spectrum')
    plt.show()

# 显示最大值及其对应的角度值
print(f'对应的角度值为：{max_angle}')