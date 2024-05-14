import numpy as np
import matplotlib.pyplot as plt
import Signal_Source_Simulation as SSM

# 超级简陋的DAS算法，仅用于仿真，对采样频率要求很高（取整形缩进的舍入误差），精度很一般

def CBF_circular_time_basis(mics: np.ndarray, mics_signals: np.ndarray, STEP_NUM=360, frecuncy_carry=500, c_speed=343, sampling_rate=48000, plot_polar=False): # 圆形阵列 常规波束形成算法 Delay And Sum
    
    # mics: 所有麦克风的坐标，shape=(mics_num, 2)
    # mics_signals: 各麦克风信号，shape=(mics_num, signal_length)
    # STEP_NUM: 步数，即计算的方向数目（角度分辨率）
    # frecuncy_carry: 载波频率，默认500Hz
    # c_speed: 声速，默认343m/s

    # lamda = c_speed / frecuncy_carry  # 波长
    # radius = np.linalg.norm(mics[0])  # 半径
    # mics_theta = np.arctan2(mics[:, 1], mics[:, 0])  # 各麦克风与x轴的夹角
    signal_length = mics_signals.shape[1]
    # print(mic_theta)

    # 初始化响应列表
    response_list = np.zeros(STEP_NUM, dtype=np.float64)

    for theta, idx in zip(np.linspace(0, 2*np.pi, STEP_NUM), range(STEP_NUM)):
        
        vector_source_theta = np.array([np.cos(theta), np.sin(theta)])

        # projection_distance_O = np.dot(mics[-1], vector_source_theta)
        # print('Projection distance:', projection_distance_O)

        # 延迟信号矩阵初始化
        signal_delayed = np.zeros([mics.shape[0], signal_length], dtype=np.float64)
        # plt.figure()
        
        for i in range(mics.shape[0] - 1): # 原点
            projection_distance_i = np.dot(mics[i], vector_source_theta)
            # projection_distance = projection_distance_i - projection_distance_O
            time_delay = projection_distance_i / c_speed
            snap_delay = int(time_delay * sampling_rate)
            if snap_delay >= 0: # 把第i个麦克风信号延迟delay_snaps_matrix[i]个采样点，空出的部分用0填充
                signal_delayed[i] = np.concatenate((mics_signals[i][snap_delay:], np.zeros(snap_delay)))
            else: # 把第i个麦克风信号提前delay_snaps_matrix[i]个采样点，空出的部分用0填充
                signal_delayed[i] = np.concatenate((np.zeros(-snap_delay), mics_signals[i][:snap_delay]))
                
            # plt.subplot(mics.shape[0], 1, i+1)
            # plt.plot(signal_delayed[i])
            # plt.title('Mic '+ str(i) + 'Delay: ' + str(snap_delay))

        # 计算响应值
        response_list[idx] = max(abs(np.sum(signal_delayed, axis=0) / signal_length))  # 计算响应值
        # response_list[idx] = np.dot(signal_delayed.T, signal_delayed).shape  # 计算响应值
        # print(np.dot(signal_delayed.T, signal_delayed).shape)  # 计算响应值  
        
        # print('Response at angle', np.rad2deg(theta), ':', response_list[idx])
        # plt.show()    
    
    # 归一化
    response_list = response_list / np.max(response_list)
    response_list = -20 * np.log10(response_list)  # 转换为dB

    print('Max response angle:', np.rad2deg(np.argmax(response_list) * 2 * np.pi / STEP_NUM))

    # 绘制空间谱
    if plot_polar:        
        plt.figure()
        ax = plt.subplot(111, projection='polar')  # 设置子图为极坐标形式
        ax.plot(np.linspace(0, 2*np.pi, STEP_NUM), response_list)  # 将角度转换为弧度并绘制极坐标图
        ax.set_title('Circular_CBF (DAS) spatial spectrum (Polar)')
        plt.show()

    # 使用极坐标形式
    else:
        plt.figure()
        plt.plot(np.linspace(0, 360, STEP_NUM), response_list)
        plt.xlabel('Angle')
        plt.ylabel('dB')
        plt.title('Circular_CBF (DAS) spatial spectrum')
        plt.show()

    return np.rad2deg(np.argmax(response_list) * 2 * np.pi / STEP_NUM)  # 返回最大响应方向角

# 测试样例（圆形阵列）
# region
mics, l_total =  SSM.get_array(array_type='circular', d_or_r=1.5, num=7, plot=False)
# print(mics) 

SOURCE = np.array([-20, 0])

f = 500
sampling_rate = 48000
t = np.linspace(0, 0.1, sampling_rate)
SIGNAL = np.sin(t*2*np.pi*(f)) * np.exp(-t/0.1)*5
# SIGNAL = np.random.randn(sampling_rate)

mics_signal_array = SSM.get_receiver_signal(source_location=SOURCE, mic_array=mics, mic_array_lamda_total=l_total, signal=SIGNAL, sampling_rate=sampling_rate, c_speed=343, frecuncy_carry=f, AWGN=False, SNR_dB=40, plot=False)

angle = CBF_circular_time_basis(mics=mics, mics_signals=mics_signal_array, STEP_NUM=720, frecuncy_carry=f, c_speed=343, plot_polar=True)

# endregion