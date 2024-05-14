import wave
from matplotlib import pyplot as plt
import numpy as np
import Signal_Source_Simulation as sss
import GCC as gcc

def vector_sum_location(mics, mics_signals):
# def vector_sum_location(mics, mics_signals0, mics_signals1, mics_signals2, mics_signals3):
    signal_length = len(mics_signals[0])
    mics_num = mics.shape[0]

    vector_sum= np.array([0.0, 0.0])

    # 计算相对于原点位置时间延迟向量并求和
    for i in range(mics_num-1):
        _, delay = gcc.gcc_phat(mics_signals[i], mics_signals[mics_num-1])
        
        if delay > signal_length/2:
            delay = delay - signal_length

        # print(i, delay)

        # vector_sum[0] += mics[i][0] * delay
        # vector_sum[1] += mics[i][1] * delay
        vector_sum += mics[i] * delay

    # 把vector_sum归一化
    vector_sum = vector_sum / np.linalg.norm(vector_sum)
    # 求俯角
    theta = np.arctan2(vector_sum[1], vector_sum[0])

    # return vector_sum, theta, np.rad2deg(theta)

    if np.rad2deg(theta) < 0:
        theta = theta + 2*np.pi
    return np.rad2deg(theta)


# 测试样例
# region
    
mics, l_total =  sss.get_array(array_type='circular', d_or_r=0.04, num=7, plot=False)
# print(mics)

theta = 311.42
SOURCE = np.array([np.cos(np.deg2rad(theta)), np.sin(np.deg2rad(theta))])

# SIGNAL = np.sin(np.linspace(0, 1, 28800)*4*np.pi)
# SIGNAL = np.random.normal(0, 1, 28800)
sampling_rate = 48000

# wave.read 对象方法
wr_1 = wave.open('./audio_data/voice_boy.wav', 'rb')
# 读取音频数据
SIGNAL = wr_1.readframes(wr_1.getnframes())
SIGNAL = np.frombuffer(SIGNAL, dtype=np.int16)

# import audio2np_array as a2n
# SIGNAL, sampling_rate = a2n.mp3_to_numpy_array('test_audio.MP3')

for i in [20]:
# for i in [20, 30, 100]:

    print("SNR_dB:", i)
    mics_signals = sss.get_receiver_signal(source_location=SOURCE, mic_array=mics, mic_array_lamda_total=l_total, signal=SIGNAL, sampling_rate=sampling_rate, c_speed=343, frecuncy_carry=1000, AWGN=True, SNR_dB=i, plot=False)

    SSL_estimation = vector_sum_location(mics, mics_signals)
    print("预测朝向：", SSL_estimation)
    # plt.figure()
    # ax = plt.subplot(111, projection='polar')  # 设置子图为极坐标形式

    # # 绘制声源点
    # ax.scatter(np.deg2rad(theta), 1, s=100, c='r', marker='x')

    # # 绘制估计声源点
    # ax.scatter(np.deg2rad(SSL_estimation), 1, s=100, c='b', marker='*')

    # plt.title('Vector Sum Location')  # 标题
    # plt.show()

    print("相对误差：", abs(theta - SSL_estimation) / 360 )
# endregion

