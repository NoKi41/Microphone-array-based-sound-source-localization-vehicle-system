import numpy as np
import matplotlib.pyplot as plt
import datetime
import scipy.signal as signal

pi = np.pi
sin = np.sin
cos = np.cos

# 参数设置
N = 6  # 阵元个数

# SOURCES_theta_angle = np.random.randint(-180, 180, 3) # 来波方向(角度) np.random.randint(1, 3)
SOURCES_theta_angle = np.array([33.48]) # 来波方向(角度) np.random.randint(1, 3)
print('声源方向：', SOURCES_theta_angle)
SOURCES_theta = SOURCES_theta_angle * pi / 180  # 来波方向(弧度)

M = len(SOURCES_theta)  # 信源个数

# SNR = 200  # 信号信噪比dB
K = 48000 # 总采样点
radius = 0.04  # 圆阵半径
f = 500  # 信号源频率
c = 340  # 声速
num = 14400 # 角度采样点

for SNR in [0, 10, 20]:
    # 阵列配置
    print('SNR:', SNR)
    MICS_theta_angle = np.linspace(0, 360, N, endpoint=False, dtype=np.float64) # 均匀圆阵麦克风方向(角度)
    # MICS_theta_angle = np.arange(0, 360, ) # 麦克风方向(角度)
    print('麦克风方向：', MICS_theta_angle)
    MICS_theta = MICS_theta_angle * pi / 180  # 麦克风方向(弧度)

    # 矩阵扩张 N * M
    SOURCES_theta_array = np.tile(SOURCES_theta, (N, 1))
    # print(SOURCES_theta_array)
    MICS_theta_array = np.tile(MICS_theta.reshape(len(MICS_theta), 1), M)
    # print(MICS_theta_array.shape)

    A = np.exp(1j * 2 * pi * (f / c) * radius * cos(SOURCES_theta_array - MICS_theta_array))  # 接收信号方向向量 (f / c = 1 / lamda) 通过频域加权实现时域延迟
    # print(A.shape)

    # 生成信号
    S = np.random.randn(M, 48000)  # 阵列接收到来自声源的信号
    t = np.linspace(0, 1, 48000)  # 时间序列
    S[0] = np.sin(2 * pi * f * t)  # 第一个声源的信号为正弦波
    K = S.shape[1]  # 信号长度

    X = A @ S  # 最终接收信号，是带有方向向量的信号

    # 在信号中添加高斯噪声
    X1 = X + np.random.normal(0, 20**(-SNR/20), X.shape)

    # t = datetime.datetime.now()

    # 协方差矩阵
    Rx = (X1 @ X1.conj().T) / K

    # 特征值分解
    D, Ev = np.linalg.eig(Rx)
    # EVA = np.sort(np.abs(D))[::-1]  # 将特征值排序
    EV = Ev[:, np.argsort(np.abs(D))[::-1]]  # 对应特征矢量排序

    # 噪声子空间
    En = EV[:, M:N] # N, N-M
    # print(En.shape)

    # 计算MUSIC谱
    p_music = np.zeros(num)
    angles = np.linspace(-180, 180, num)  # This includes the endpoint, so the angles will be from -180 to 180 inclusive

    for i, angle in enumerate(angles):
        theta_m = angle * np.pi / 180
        a = np.exp(1j * 2 * pi * (f / c) * radius * cos(theta_m - MICS_theta))
        p_music[i] = 1 / np.abs(a.conj().T @ En @ En.conj().T @ a)

    # print('运算时间：', datetime.datetime.now() - t)

    # 归一化,分贝处理
    p_music_db = 10 * np.log10(p_music / np.max(p_music))

    # 绘制空间谱
    plt.figure(figsize=(8, 8))
    plt.plot(angles, p_music_db)
    plt.xlabel('Angle')
    plt.ylabel('dB')
    # 标注SOURCES_theta_angle中的坐标线
    for u in SOURCES_theta_angle:
        plt.axvline(x = u, color='g', linestyle='--')
    # plt.title('MUSIC spatial spectrum (Frequency: {}Hz)'.format(f), fontsize=20)
    plt.title('MUSIC spatial spectrum (SNR: {}dB)'.format(SNR), fontsize=20)
    plt.show()

    plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, projection='polar')  # 设置子图为极坐标形式
    ax.plot(np.deg2rad(angles), p_music_db.real)  # 将角度转换为弧度并绘制极坐标图
    ax.set_title('MUSIC spatial spectrum (Polar) (Frequency: {}Hz)'.format(f), fontsize=20)
    ax.set_title('MUSIC spatial spectrum (Polar) (SNR: {}dB)'.format(SNR), fontsize=20)    
    plt.show()

    max_index = np.argsort(p_music_db.real)[-M:]
    print('最大值对应角度：', angles[max_index]+360)
    print('相对误差：', np.abs(360 +angles[max_index] - SOURCES_theta_angle)/360)

plt.show()