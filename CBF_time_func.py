import wave
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as signal
import Signal_Source_Simulation as sss

def CBF_circular_time_domain(mics: np.ndarray, mics_signals: np.ndarray, STEP_NUM=360, frecuncy_carry=500, c_speed=343, plot_polar=False): # 圆形阵列 常规波束形成算法 Delay And Sum
    
    # mics: 所有麦克风的坐标，shape=(mics_num, 2)
    # mics_signals: 各麦克风信号，shape=(mics_num, signal_length)
    # STEP_NUM: 步数，即计算的方向数目（角度分辨率）
    # frecuncy_carry: 载波频率，默认500Hz
    # c_speed: 声速，默认343m/s

    signal_length = mics_signals.shape[1]  # 信号长度

    # 恢复复数信号
    mics_signal_array_complex = np.zeros(mics_signals.shape, dtype=np.complex128)

    # for i in range(mics_signal_array.shape[0]):
    #     mics_signal_array_complex[i] = np.fft.fft(mics_signal_array[i])
    #     # 根据原信号长度，把负频率的部分置零
    #     mics_signal_array_complex[i][:signal_length//2+1] = 0
    #     # 反变换
    #     mics_signal_array_complex[i] = np.fft.ifft(-mics_signal_array_complex[i])

    mics_signal_array_complex = signal.hilbert(mics_signals, axis=1)
    # print(mics_signal_array_complex.shape)

    # plt.figure()
    # plt.plot(abs(np.fft.fft(mics_signals[0])))
    # plt.plot(abs(np.fft.fft(mics_signal_array_complex[0].conj())))
    # plt.show()

    # 自相关矩阵
    R = np.dot(mics_signal_array_complex, mics_signal_array_complex.conj().T) / signal_length

    lamda = c_speed / frecuncy_carry  # 波长
    radius = np.linalg.norm(mics[0])  # 半径
    mics_theta = np.arctan2(mics[:, 1], mics[:, 0])  # 各麦克风与x轴的夹角
    print(mics_theta.shape)

    # 初始化导向矢量矩阵
    direction_vectors = np.ones((mics.shape[0]), dtype=np.complex64)
    # 初始化响应列表
    response_list = np.zeros(STEP_NUM, dtype=np.float64)

    for theta, idx in zip(np.linspace(0, 2*np.pi, STEP_NUM), range(STEP_NUM)):
        # 建立导向矢量矩阵
        
        direction_vectors = np.exp(1j * 2 * np.pi * radius * np.cos(theta - mics_theta) / lamda)
        direction_vectors[-1] = 1
        # print(direction_vectors.shape)
        # print(R.shape)
        response_list[idx] = np.dot(direction_vectors.conj().T, np.dot(R, direction_vectors)).real  # 计算响应值     
        
        # response_list[idx] = max(np.dot(direction_vectors.conj(), mics_signal_array_complex).real)  # 计算响应值
        # print(response_list[idx])
               
    
    # 归一化
    response_list = response_list / np.max(response_list)
    response_list = 20 * np.log10(response_list)  # 转换为dB

    print('Max response angle:', np.rad2deg(np.argmax(response_list) * 2 * np.pi / STEP_NUM))

    # 绘制空间谱
    if plot_polar:        
        plt.figure(figsize=(8, 8))
        ax = plt.subplot(111, projection='polar')  # 设置子图为极坐标形式
        ax.plot(np.linspace(0, 2*np.pi, STEP_NUM), response_list)  # 将角度转换为弧度并绘制极坐标图
        # ax.set_title('Circular_CBF (DAS) spatial spectrum (Polar)')
        plt.show()

    # 使用极坐标形式
    else:
        plt.figure()
        plt.plot(np.linspace(0, 360, STEP_NUM), response_list)
        plt.xlabel('Angle')
        plt.ylabel('dB')
        # plt.title('Circular_CBF (DAS) spatial spectrum')
        plt.show()

    return np.rad2deg(np.argmax(response_list) * 2 * np.pi / STEP_NUM)  # 返回最大响应方向角

# 测试样例（圆形阵列）
# region
sampling_rate = 48000
mics, l_total =  sss.get_array(array_type='circular', d_or_r=0.04, num=7, plot=False)
# print(mics)

# theta = 49
# SOURCE = np.array([np.cos(theta/180*np.pi), np.sin(theta/180*np.pi)]) * 25
# SOURCE = np.array([20, 24])
# theta = np.rad2deg(np.arctan2(SOURCE[1], SOURCE[0]))
theta = 311.42
SOURCE = np.array([np.cos(theta/180*np.pi), np.sin(theta/180*np.pi)]) * 25

# # wave.read 对象方法
# wr_1 = wave.open('./audio_data/voice_boy.wav', 'rb')
# # 读取音频数据
# SIGNAL = wr_1.readframes(wr_1.getnframes())
# SIGNAL = np.frombuffer(SIGNAL, dtype=np.int16)

f = 700
t = np.linspace(0, 1, sampling_rate)
SIGNAL = np.sin(t*2*np.pi*(f))
# + 0.5*np.sin(t*2*np.pi*(f+50)) + 0.5*np.sin(t*2*np.pi*(f-50))

for i in [100]:
    mics_signal_array = sss.get_receiver_signal(source_location=SOURCE, mic_array=mics, mic_array_lamda_total=l_total, signal=SIGNAL, sampling_rate=sampling_rate, c_speed=343, frecuncy_carry=f, AWGN=True, SNR_dB=i, plot=False)

    angle = CBF_circular_time_domain(mics=mics, mics_signals=mics_signal_array, STEP_NUM=1440, frecuncy_carry=f, c_speed=343, plot_polar=True)
    print(theta, angle, abs(theta-angle)/360)

# endregion
