import time
import wave
import GCC as gcc
import numpy as np
import matplotlib.pyplot as plt

def srp():
    # 设置声源坐标
    SOURCE = np.array([2.4, 2.0])
    pi = np.pi
    sin = np.sin
    cos = np.cos
    theta = 2
    # SOURCE = 6 * np.array([cos(theta), sin(theta)])

    # 设置麦克风阵列，每个麦克风的二维坐标，单位为米
    M = 7 

    # 地图分辨率(-5,5)*(-5,5)
    step = 0.05

    MICS = np.array([
        [cos(2*1*pi/6), sin(2*1*pi/6)],
        [cos(2*2*pi/6), sin(2*2*pi/6)],
        [cos(2*3*pi/6), sin(2*3*pi/6)],
        [cos(2*4*pi/6), sin(2*4*pi/6)],
        [cos(2*5*pi/6), sin(2*5*pi/6)],
        [cos(2*6*pi/6), sin(2*6*pi/6)],
        [0, 0]
        ], dtype=np.float64) * 2
    
    # MICS = np.array([
    #     [0, 1],
    #     [-0.866, -0.5],
    #     [0.866, -0.5],
    #     [0, 0]
    #     ], dtype=np.float64)

       
    # 假定信号传播速度为340m每秒，采样频率为4410Hz
    SPEED = 340
    SAMPLING_RATE = 48000

    # 设定原始信号长度
    LEN = 1146880
    # t = np.linspace(0, int(LEN/SAMPLING_RATE), LEN)
    # Signal = np.random.randint(low=-10, high=10, size=LEN)

    # wave.read 对象方法
    wr_1 = wave.open('./audio_data/voice_boy.wav', 'rb')
    # 读取音频数据
    Signal = wr_1.readframes(wr_1.getnframes())
    Signal = np.frombuffer(Signal, dtype=np.int16)
       

    # 定义每个麦克风接收信号的矩阵，第一个索引为麦克风编号
    Signal_MIC = np.zeros((M, LEN))

    # 根据几何关系，推算各个麦克风处接收到的信号，不考虑传播中的损耗，只考虑传播延迟
    for i in range(M):

        # 先计算从信号源传播过来的时间
        delta_t = np.linalg.norm(SOURCE - MICS[i]) / SPEED

        # 由时间，推算需要延迟的码元数
        delay = int(delta_t * SAMPLING_RATE)

        # if delay > LEN:
        #     print('Error')

        # 在原始信号基础上，进行延迟
        Signal_MIC[i] = np.pad(Signal, (delay, 0), 'constant')[:LEN] + 10 * np.random.normal(0, 1, LEN)
        
        # print(len(Signal_MIC[i]))
        # plt.plot(Signal_MIC[i])

    # plt.show()

    time_start = time.time()

    # 初始化互相关函数矩阵
    R = np.zeros((M, M, LEN), dtype=np.float64)

    # 计算每一对麦克风之间的接收信号的gcc_phat
    for i in range(M):
        for j in range(i+1, M):
            R[i, j, :], _ = gcc.gcc_phat(Signal_MIC[i, :], Signal_MIC[j, :])
            
            # plt.plot(R[i, j, :])

    # plt.show()

    # 设定一个空间范围，即声源可能在的空间区域
    x_range = np.arange(-5, 5+step, step)
    y_range = np.arange(-5, 5+step, step)

    # 初始化地图矩阵，用来存放响应值
    map_matrix = np.zeros((len(x_range), len(y_range)), dtype=np.float64)

    count = 0

    # 遍历空间范围内的每一个站点
    for ix, x in enumerate(x_range):
        for iy, y in enumerate(y_range):
        
            # 用以累计响应值
            sum = 0

            # 遍历每一对麦克风组合
            for p in range(0,M):
                for q in range(p+1, M):
                    
                    # 计算站点位置对麦克风对p,q的时间差
                    delta_t = abs((np.linalg.norm([y, x] - MICS[p]) - np.linalg.norm([y, x] - MICS[q])) / SPEED) # TODO:norm[x,y] -> norm[y,x]
                    
                    # 延迟时间对应的延迟码元数
                    delay = int(delta_t * SAMPLING_RATE)

                    # if delay < 0: # 把delay的abs打掉，解决对称问题
                    #     delay = delay + LEN

                    # if np.linalg.norm([x, y] - MICS[p]) > np.linalg.norm([x, y] - MICS[q]):
                    #     delay = 255 - delay

                    # delay2 = 255 - delay

                    # if delay >= LEN:
                        # print("Error")

                    # 累加响应值
                    sum += R[p, q, delay]
                    # sum = sum + R[p, q, delay] + R[p, q, delay2]
            
            # 把响应值存在地图矩阵的对应位置上
            map_matrix[-iy, ix] = sum # TODO:[ix,iy] -> norm[-iy,ix]

            count += 1

        # 打印遍历进度
        if count % 10000 == 0:
            print('Progress: {:.2f}%'.format(count/len(x_range)/len(y_range)*100))

    # max_index = np.unravel_index(np.argmax(map_matrix, axis=None), map_matrix.shape)

    # 找到map_matrix中最大值的索引
    max_index = np.unravel_index(np.argmax(map_matrix, axis=None), map_matrix.shape)
    time_end = time.time()
    print('Time used: {:.2f}s'.format(time_end-time_start))

    # 可视化结果
    plt.figure('3-5\ SRP\ PHAT\ Map\ Step\ =\ {}m'.format(step), figsize=(8, 8.5))
    plt.imshow(map_matrix, extent=[x_range.min(), x_range.max(), y_range.min(), y_range.max()])
    # plt.colorbar()
    plt.xticks([])
    plt.yticks([])
    
    # 标记最大值的位置
    plt.scatter(x_range[max_index[1]], - y_range[max_index[0]], c='blue', marker='*', label='Maximum')

    # 标记真实声源位置
    plt.scatter(SOURCE[1], SOURCE[0], c='red', marker='x', label='Source') # TODO: SOURCE[0], SOURCE[1] -> SOURCE[1], SOURCE[0]

    print(x_range[max_index[1]], - y_range[max_index[0]])

    # 标记麦克风位置
    for mic in MICS:
        plt.scatter(mic[1], mic[0], c='white', marker='o', label='Mic' if 'Mic' not in plt.gca().get_legend_handles_labels()[1] else "") # TODO: mic[0], mic[1] -> mic[1], mic[0]

    plt.title('SRP-PHAT Map: Step = {}m'.format(step))
    plt.xlabel('X')
    plt.ylabel('Y')
    # 图例在左下角
    plt.gca().legend(loc='lower left')   
    plt.show()

srp()