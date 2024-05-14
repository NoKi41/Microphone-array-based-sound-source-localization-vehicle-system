from matplotlib import pyplot as plt
import numpy as np
import pyaudio
import wave
import scipy.signal as signal

# 设置录音参数
CHUNK = 2048         # 单次录音的帧数
FORMAT = pyaudio.paInt16  # 录音格式
CHANNELS = 8         # 录音通道数
RATE = 48000         # 采样率
RECORD_SECONDS = 1   # 录音时间
# WAVE_OUTPUT_FILENAME = "output.wav"  # 保存的文件名

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
mics = np.array([[ 4.00000000e-02, 0.00000000e+00],     # mic1
                [-2.00000000e-02, -3.46410162e-02],    # mic2
                [ 2.00000000e-02, 3.46410162e-02],     # mic3
                [ 2.00000000e-02, -3.46410162e-02],    # mic4
                [-2.00000000e-02, 3.46410162e-02],     # mic5
                [ 0.00000000e+00, 0.00000000e+00],    # mic6
                [-4.00000000e-02, 0.00000000e+00]      # mic7
                ])

def CBF_circular(mics: np.ndarray, mics_signals: np.ndarray, threshold=1000, STEP_NUM=360, frecuncy_carry=500, c_speed=343, plot_polar=False): # 圆形阵列 常规波束形成算法
    
    # mics: 所有麦克风的坐标，shape=(mics_num, 2)
    # mics_signals: 各麦克风信号，shape=(mics_num, signal_length)
    # STEP_NUM: 步数，即计算的方向数目（角度分辨率）
    # frecuncy_carry: 载波频率，默认500Hz
    # c_speed: 声速，默认343m/s

    if max(mics_signals[0]) < threshold:
        return None
    
    mics_signals_complex = signal.hilbert(mics_signals, axis=1)

    # 自相关矩阵
    R = np.dot(mics_signals_complex, mics_signals_complex.conj().T) / mics_signals.shape[1]

    lamda = c_speed / frecuncy_carry  # 波长
    radius = np.linalg.norm(mics[0])  # 半径
    mics_theta = np.arctan2(mics[:, 1], mics[:, 0])  # 各麦克风与x轴的夹角
    # print(mic_theta)

    # 初始化导向矢量矩阵
    direction_vectors = np.ones((mics.shape[0]), dtype=complex)
    # 初始化响应列表
    response_list = np.zeros(STEP_NUM, dtype=complex)

    for theta, idx in zip(np.linspace(0, 2*np.pi, STEP_NUM), range(STEP_NUM)):
        # 建立导向矢量矩阵
        
        for i in range(mics.shape[0]): 
            if i != 5: # 原点，最后一个方向矢量为1
                direction_vectors[i] = np.exp(1j * 2 * np.pi * radius * np.cos(theta - mics_theta[i]) / lamda)

        # print(direction_vectors.shape)
        # print(mics_signals.shape)

        # response_list[idx] = np.dot(direction_vectors.conj().T, np.dot(R, direction_vectors)).real  # 计算响应值     
        
        response_list[idx] = max(np.dot(direction_vectors.conj(), mics_signals_complex).real)  # 计算响应值
        # print(response_list[idx])
    
    # 归一化
    response_list = response_list / np.max(response_list)
    response_list = 20 * np.log10(response_list)  # 转换为dB

    angle = np.rad2deg(np.argmax(response_list) * 2 * np.pi / STEP_NUM)  # 最大响应方向角
    # 大于180度，取反
    if angle > 180:
        angle = angle - 360

    # print('Max response angle:', angle)

    # # 绘制空间谱
    # if plot_polar:        
    #     plt.figure()
    #     ax = plt.subplot(111, projection='polar')  # 设置子图为极坐标形式
    #     ax.plot(np.linspace(0, 2*np.pi, STEP_NUM), response_list)  # 将角度转换为弧度并绘制极坐标图
    #     ax.set_title('Circular_MUSIC spatial spectrum (Polar)')
    #     plt.show()

    # # 使用极坐标形式
    # else:
    #     plt.figure()
    #     plt.plot(np.linspace(0, 360, STEP_NUM), response_list)
    #     plt.xlabel('Angle')
    #     plt.ylabel('dB')
    #     plt.title('Circular_MUSIC spatial spectrum')
    #     plt.show()

    return angle  # 返回最大响应方向角

while True:
    # 开始录音
    # print("开始录音...")
    frames = []

    for i in range(int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK, exception_on_overflow=False)
        frames.append(data)

    # print(np.frombuffer(frames[0], dtype=np.int16).shape)

    # # 关闭录音设备
    # print("录音结束...")
    # stream.stop_stream()
    # stream.close()
    # p.terminate()

    # # 保存录音文件
    # print("开始保存文件...")
    # wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    # wf.setnchannels(CHANNELS)
    # wf.setsampwidth(p.get_sample_size(FORMAT))
    # wf.setframerate(RATE)
    # wf.writeframes(b''.join(frames))
    # wf.close()
    # print("文件保存成功！") 

    # _____________________________________________________
    # data初始化为形状的数组(8,1)
    data = np.zeros((8, 0))

    # 数据整形，帧连续但mic索引不连续，分块再转置
    for i in range(len(frames)):
        data = np.concatenate((data, np.frombuffer(frames[i], dtype=np.int16).reshape((CHUNK, -1)).T), axis=1)

    # print(data.shape)

    # plt.figure()
    # # 分别画8路data数据
    # for i in range(8):
    #     plt.subplot(8, 1, i+1)
    #     plt.plot(data[i])
    #     plt.ylim(-5000, 5000)
    #     plt.title("Channel %d" % (i+1))

    # plt.show()    

    print("预测朝向：\t", CBF_circular(mics, data[:7, :], threshold=10))
    # 串口发送预测朝向
