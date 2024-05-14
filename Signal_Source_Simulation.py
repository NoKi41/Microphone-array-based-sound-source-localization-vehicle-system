import numpy as np
import matplotlib.pyplot as plt

def get_array(array_type = 'circular' or 'linear', d_or_r = 0.1, num = 7, plot = False): # -> lamda_total, mic_array
  
    if plot == True:
        plt.figure()
    
    if array_type == 'linear':
        # 以0为对称中心，步长为d_or_r的num个元素的列表
        array_x = np.arange(-d_or_r * (num - 1) / 2, d_or_r * (num - 1) / 2 + d_or_r, d_or_r)
        # 以array_x为横坐标，0为纵坐标构建二维数组
        array_y = np.zeros(num)
        # 画出np.array([array_x, array_y]).T
        if plot == True:
            plt.plot(array_x, array_y, 'o')
            for i in range(num):
                plt.text(array_x[i], array_y[i], str(i), size=20)

            # 画框比例1：1
            plt.axis('equal')
            plt.show()
        return np.array([array_x, array_y]).T, (num-1)*d_or_r
    
    elif array_type == 'circular':

        array = np.zeros([num,2])

        for i in range(num-1):
            array[i,:] = [d_or_r * np.cos(2 * np.pi * i / (num-1)), d_or_r * np.sin(2 * np.pi * i / (num-1))]

        # 圆心有一元
        array[num-1,:] = [0, 0]

        # 画出array
        if plot == True:
            plt.plot(array[:,0], array[:,1], 'o')
            # 标注每个元素的索引
            for i in range(num):
                plt.text(array[i,0], array[i,1], str(i), size=20)

            # 画出圆
            theta = np.linspace(0, 2*np.pi, 100)
            plt.plot(d_or_r*np.cos(theta), d_or_r*np.sin(theta))
            # 画框比例1：1
            plt.axis('equal')
            plt.show()

        return array, 2*d_or_r
    
    else:
        print('Unknown array type.')
        return False

def get_receiver_signal(source_location : np.ndarray, mic_array : np.ndarray, mic_array_lamda_total : float, signal : np.ndarray, sampling_rate : int, c_speed : float, frecuncy_carry : float, AWGN : bool, SNR_dB : float, plot : bool): # -> receiver_signal

    # 判断近场或远场情景
    signal_length = len(signal)
    lamda = c_speed / frecuncy_carry
    # print("波长：", lamda)
    distance_threshold = 2 * (mic_array_lamda_total**2) / lamda
    # print("判定门限：", distance_threshold)
    source_distance = np.linalg.norm(source_location)
    signal_amplitude = max(signal)

    receiver_signal = np.zeros([mic_array.shape[0], signal_length], dtype=np.float64)
        
    if plot:
        plt.figure()

    if source_distance <= distance_threshold: # 近场模型下认为是球面波传播
        # print('近场模型')
        for i in range(mic_array.shape[0]):
            # mic_array[i]到source_location的距离
            distance = np.linalg.norm(mic_array[i] - source_location)
            time_delay = distance / c_speed
            snaps_delay = int(time_delay * sampling_rate)
            if snaps_delay >= signal_length:
                print('Error: snaps_delay >= len(signal)')
            # 让对应的接收信号前snaps_dealy位为0（延迟）
            receiver_signal[i] = np.pad(signal, (snaps_delay, 0), 'constant')[: signal_length]
            
            if AWGN:
                receiver_signal[i] = receiver_signal[i] + np.random.normal(0, signal_amplitude*np.exp(-SNR_dB/20), signal.shape)
            if plot:
                plt.subplot(mic_array.shape[0],1,i+1)
                plt.plot(receiver_signal[i])
                plt.title('MIC{} [dis:{:.4f}, time:{:.4f}, snaps:{}]'.format(i, distance, time_delay, snaps_delay))
                # 标准snaps_delay的位置
                plt.axvline(x=snaps_delay, color='r', linestyle='--')
                # 隐藏刻度
                plt.xticks([])
                plt.yticks([1, 0,- 1])
        
        if plot:
            plt.show()

        return receiver_signal
            
    else: # 远场模型下认为是平面波传播
        # print('远场模型')
        vector_source_theta = source_location / np.linalg.norm(source_location)
        # if np.arctan2(vector_source_theta[1], vector_source_theta[0]) * 180 / np.pi < 0:
        #     print('声源朝向：', 360 + np.arctan2(vector_source_theta[1], vector_source_theta[0]) * 180 / np.pi)
        # else:
        #     print('声源朝向：', np.arctan2(vector_source_theta[1], vector_source_theta[0]) * 180 / np.pi)
        
        for i in range(mic_array.shape[0]):
            # mic_array[i]到source_location的距离
            vector_mic_2_source = source_location - mic_array[i]
            projection_distance = np.dot(vector_mic_2_source, vector_source_theta)
            time_delay = projection_distance / c_speed
            snaps_delay = int(time_delay * sampling_rate)
            if snaps_delay >= signal_length:
                print('Error: snaps_delay >= len(signal)')
            # 让对应的接收信号前snaps_dealy位为0（延迟）
            receiver_signal[i] = np.pad(signal, (snaps_delay, 0), 'constant')[: signal_length]

            if AWGN:
                receiver_signal[i] = receiver_signal[i] + np.random.normal(0, signal_amplitude*np.exp(-SNR_dB/20), signal.shape)
            if plot:
                plt.subplot(mic_array.shape[0],1,i+1)
                plt.plot(receiver_signal[i])
                plt.title('MIC{} [pro_dis:{:.4f}, time:{:.4f}(ms), snaps:{}]'.format(i, projection_distance, time_delay*1000, snaps_delay))
                # 标准snaps_delay的位置
                plt.axvline(x=snaps_delay, color='r', linestyle='--')
                # 隐藏刻度
                plt.xticks([])
                # plt.yticks([1, 0,- 1])

        if plot:    
            plt.show()

        return receiver_signal

# 测试样例
# region
    
# mics, l_total =  get_array(array_type='circular', d_or_r=0.5, num=7, plot=True)
# print(mics)

# SOURCE = np.array([1, np.sqrt(3)]) * 2
# SIGNAL = np.sin(np.linspace(0, 20, 1024)*2*np.pi)

# get_receiver_signal(source_location=SOURCE,
#                     mic_array=mics,
#                     mic_array_lamda_total=l_total,
#                     signal=SIGNAL,
#                     sampling_rate=44000,
#                     c_speed=343,
#                     frecuncy_carry=500,
#                     AWGN=False,
#                     SNR_dB=40,
#                     plot=True)

# endregion