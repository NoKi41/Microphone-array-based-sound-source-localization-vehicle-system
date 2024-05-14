from matplotlib import pyplot as plt
import numpy as np
import pyaudio
import wave
import GCC as gcc

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


def vector_sum_location(mics: np.ndarray, mics_signals: np.ndarray, threshold=1000):
# def vector_sum_location(mics, mics_signals0, mics_signals1, mics_signals2, mics_signals3):
    if max(mics_signals[0]) < threshold:
        return None
    
    signal_length = len(mics_signals[0]) # 信号长度
    mics_num = mics.shape[0] # 麦克风数目

    vector_sum= np.array([0.0, 0.0]) # 相对时间延迟向量的初始化

    # 计算相对于原点位置时间延迟向量并求和
    for i in range(mics_num):
        if i != 5: # 5号麦克风为参考麦克风，不参与计算    
            _, delay = gcc.gcc_phat(mics_signals[i], mics_signals[5])  # 计算5号麦克风到其他麦克风的相对时间延迟
            
            if delay > signal_length/2:  # 时间延迟超过信号长度的一半，说明信号延迟到了另一端，需要减去信号长度
                delay = delay - signal_length 

            # print(i, delay)

            vector_sum[0] += mics[i][0] * delay  # 计算x方向的相对时间延迟向量
            vector_sum[1] += mics[i][1] * delay  # 计算y方向的相对时间延迟向量

    # 把vector_sum归一化
    vector_sum = vector_sum / np.linalg.norm(vector_sum)
    # 求俯角
    theta = np.arctan2(vector_sum[1], vector_sum[0])

    # return vector_sum, theta, np.rad2deg(theta)

    # if np.rad2deg(theta) < 0:
    #     theta = theta + 2*np.pi
    return np.rad2deg(theta)

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

    print("预测朝向：\t", vector_sum_location(mics, data[:7, :], threshold=1000))
    # 串口发送预测朝向
