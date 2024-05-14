import struct
from matplotlib import pyplot as plt
import numpy as np
import pyaudio
import wave
import TDOA.GCC as gcc
import serial
import time
import threading
import queue

# 设置录音参数
CHUNK = 2048  # 单次录音的帧数
FORMAT = pyaudio.paInt16  # 录音格式
CHANNELS = 8  # 录音通道数
RATE = 48000  # 采样率
RECORD_SECONDS = 1  # 录音时间

# 声音阈值
SOUND_THRESHOLD = 500

# 创建队列用于存储声音数据
sound_queue = queue.Queue()


# 声源定位算法---GCC
def vector_sum_location(mics, mics_signals):
    signal_length = len(mics_signals[0])  # 信号长度
    mics_num = mics.shape[0]  # 麦克风数目

    vector_sum = np.array([0.0, 0.0])  # 相对时间延迟向量的初始化

    # 计算相对于原点位置时间延迟向量并求和
    for i in range(mics_num):
        if i != 5:  # 5号麦克风为参考麦克风，不参与计算
            _, delay = gcc.gcc_phat(mics_signals[i], mics_signals[5])  # 计算5号麦克风到其他麦克风的相对时间延迟

            if delay > signal_length / 2:  # 时间延迟超过信号长度的一半，说明信号延迟到了另一端，需要减去信号长度
                delay = delay - signal_length

            vector_sum[0] += mics[i][0] * delay  # 计算x方向的相对时间延迟向量
            vector_sum[1] += mics[i][1] * delay  # 计算y方向的相对时间延迟向量

    # 把vector_sum归一化
    vector_sum = vector_sum / np.linalg.norm(vector_sum)
    # 求俯角
    theta = np.arctan2(vector_sum[1], vector_sum[0])

    return np.rad2deg(theta)


# 初始化
p = pyaudio.PyAudio()
# 打开串口，注意在命令行里确定dev的设备名称，timeout指停止位1秒
# ser = serial.Serial('/dev/ttyAMA0', 9600, timeout=1)

# 打开录音设备
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK,
                input_device_index=1)  # 设备号注意


# 数据整形，帧连续但mic索引不连续，分块再转置
def reshape_data(frames):
    data = np.zeros((8, 0))
    for i in range(len(frames)):
        data_chunk = np.frombuffer(frames[i], dtype=np.int16).reshape((CHUNK, -1)).T
        data = np.concatenate((data, data_chunk), axis=1)
    return data


# 从录音设备中读取声音数据，并存入队列中
def record_sound(stream):
    while True:
        frames = []
        for _ in range(int(RATE / CHUNK * RECORD_SECONDS)):
            data_chunk = stream.read(CHUNK, exception_on_overflow=False)
            frames.append(data_chunk)

        # 将声音数据放入队列
        sound_data = reshape_data(frames)
        sound_queue.put(sound_data)


# 当声音超过设定的阈值时进行声源定位
def locate_sound():
    # 在def里面对外面的数据进行操作时，需要先声明为全局变量！！！
    global sound_queue
    # 传感器阵列坐标
    mics = np.array([[4.00000000e-02, 0.00000000e+00],  # mic1
                     [-2.00000000e-02, -3.46410162e-02],  # mic2
                     [2.00000000e-02, 3.46410162e-02],  # mic3
                     [2.00000000e-02, -3.46410162e-02],  # mic4
                     [-2.00000000e-02, 3.46410162e-02],  # mic5
                     [2.00000000e-02, -3.46410162e-02],  # mic6
                     [-4.00000000e-02, 4.89858720e-18]  # mic7
                     ])

    while True:
        # 从队列中获取声音数据
        sound_data = sound_queue.get()

        # 每读一次就要清空一次队列避免数据过大
        sound_queue = queue.Queue()

        # 如果声音超过阈值，则进行声源定位
        if np.max(sound_data[0]) > SOUND_THRESHOLD:
            # 计算声源角度
            angle = vector_sum_location(mics, sound_data[:7, :])
            angle = int(angle)

            try:
                b = struct.pack("bbhb", 0x2C, 0x12, angle, 0x5B)
                # ser.write(b)
                print("Data sent:", int(angle))
                time.sleep(0.01)  # 每隔1秒发送一次数据
            except Exception as e:
                print("Error:", e)


# 创建并启动录音线程
record_thread = threading.Thread(target=record_sound, args=(stream,))
record_thread.daemon = True
record_thread.start()

# 创建并启动声源定位线程
locate_thread = threading.Thread(target=locate_sound)
locate_thread.daemon = True
locate_thread.start()

try:
    while True:
        pass  # 主线程保持运行

except KeyboardInterrupt:
    print("Program stopped by user.")

finally:
    # 关闭串口
    # ser.close()
    print("Serial port closed.")
