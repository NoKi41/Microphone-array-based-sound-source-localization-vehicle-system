import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
import Signal_Source_Simulation as sss
import random

pi = np.pi
sin = np.sin
cos = np.cos

def MUSIC_circular(mics_signal_array: np.ndarray, mics_array: np.ndarray, SOURCE_NUM = 1, STEP_NUM = 360, frecuncy_carry = 1, c_speed = 343, plot_polar=False):

    # SOURCE_NUM = 1 # 信号源数目
    # STEP_NUM = 360 # 角度分辨率

    mics_array = mics_array[:-1, :]

    mics_num = mics_array.shape[0]
    signal_length = mics_signal_array.shape[1]
    radius = np.linalg.norm(mics_array[0, :])

    # 扣掉原点mic，舍弃mics_array、mics_signal_array最后一行数据
    mics_signal_array = mics_signal_array[:-1, :]

    # 恢复复数信号
    mics_signal_array_complex = np.zeros(mics_signal_array.shape, dtype=np.complex128)

    # for i in range(mics_signal_array.shape[0]):
    #     mics_signal_array_complex[i] = np.fft.fft(mics_signal_array[i])
    #     # 根据原信号长度，把负频率的部分置零
    #     mics_signal_array_complex[i][:signal_length//2+1] = 0
    #     # 反变换
    #     mics_signal_array_complex[i] = np.fft.ifft(mics_signal_array_complex[i])

    mics_signal_array_complex = signal.hilbert(mics_signal_array, axis=1)
 
    # 根据mics_array(麦克风个数，xy坐标)，得到mics_theta(麦克风角度)
    mics_theta = np.arctan2(mics_array[:, 1], mics_array[:, 0])
    # print(np.rad2deg(mics_theta))

    # 计算协方差矩阵
    Rx = (mics_signal_array_complex @ mics_signal_array_complex.conj().T) / signal_length

    # 特征值分解
    D, Ev = np.linalg.eig(Rx)
    # EVA = np.sort(np.abs(D))[::-1]  # 将特征值排序
    EV = Ev[:, np.argsort(np.abs(D))[::-1]]  # 对应特征矢量排序

    # 噪声子空间
    En = EV[:, SOURCE_NUM:mics_num] # N, N-M

    # 计算MUSIC谱
    p_music = np.zeros(STEP_NUM)
    angles = np.linspace(-180, 180 + 360/STEP_NUM, STEP_NUM)  # This includes the endpoint, so the angles will be from -180 to 180 inclusive

    for i, angle in enumerate(angles):
        theta_m = angle * np.pi / 180
        a = np.exp(1j * 2 * pi * (frecuncy_carry / c_speed) * radius * cos(theta_m - mics_theta))
        p_music[i] = 1 / np.abs(a.conj().T @ En @ En.conj().T @ a)

    # 归一化,分贝处理
    p_music_db = 10 * np.log10(p_music / np.max(p_music))

    # 绘制空间谱
    if plot_polar:        
        plt.figure()
        ax = plt.subplot(111, projection='polar')  # 设置子图为极坐标形式
        ax.plot(np.deg2rad(angles), p_music_db.real)  # 将角度转换为弧度并绘制极坐标图
        ax.set_title('Circular_MUSIC spatial spectrum (Polar)')
        plt.show()

    # 使用极坐标形式
    else:
        plt.figure()
        plt.plot(angles, p_music_db.real)
        plt.xlabel('Angle')
        plt.ylabel('dB')
        plt.title('Circular_MUSIC spatial spectrum')
        plt.show()

    print('Max response angle:', angles[np.argmax(p_music_db)])   

    return angles[np.argmax(p_music_db)]

# 测试样例
# region
    
mics, l_total =  sss.get_array(array_type='circular', d_or_r=0.1, num=7, plot=False)
# print(mics)

SOURCE = np.array([-7,-5])

f = 650
sampling_rate = 44100
t = np.linspace(0, 0.1, sampling_rate)
SIGNAL = np.sin(t*2*np.pi*(f)) * np.sin(t*2*np.pi)

mics_signal_array = sss.get_receiver_signal(source_location=SOURCE, mic_array=mics, mic_array_lamda_total=l_total, signal=SIGNAL, sampling_rate=sampling_rate, c_speed=343, frecuncy_carry=f, AWGN=False, SNR_dB=50, plot=False)

MUSIC_circular(mics_signal_array=mics_signal_array,
               mics_array=mics,
               SOURCE_NUM=1,
               STEP_NUM=1440,
               frecuncy_carry=f,
               c_speed=343,
               plot_polar=True)

# endregion