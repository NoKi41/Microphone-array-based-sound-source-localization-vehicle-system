import scipy.signal as signal
import numpy as np
import matplotlib.pyplot as plt

# 定义信号
a = 30
t = np.linspace(0, 1, 48000)

# x = np.cos(2*np.pi*5*t) + 2 * np.cos(2*np.pi*10*t) + np.cos(2*np.pi*15*t)

x = np.sin(2*np.pi*5*t) + 2 * np.sin(2*np.pi*10*t) + np.sin(2*np.pi*15*t)

# 定义解析信号
x_hlibert = signal.hilbert(x)

# 绘制信号及滤波器
plt.figure()
plt.subplot(2, 2, 1)
plt.plot(t, x)
plt.title('Original Signal', fontsize=a)
plt.xlabel('Time', fontsize=12)
plt.ylabel('Amplitude', fontsize=12)

plt.subplot(2, 2, 2)
plt.plot(t, x_hlibert.real)
# plt.title('Analytic Signal (Imaginary Part)')
plt.title('Hilbert Transform Signal (Real Part)', fontsize=a)
plt.xlabel('Time', fontsize=12)
plt.ylabel('Amplitude', fontsize=12)

# 绘制频谱的幅值
x_fft = np.fft.fft(x)
hlibert_fft = np.fft.fft(x_hlibert)

freq = np.fft.fftfreq(len(t), d=t[1]-t[0])
# print(freq)

plt.subplot(2, 2, 3)
plt.title('FFT of Analytic Signal (Absolute)', fontsize=a)
plt.plot(freq, np.abs(x_fft))
plt.axvline(0, color='r', linestyle='--')
plt.xlim(-100, 100)
plt.xlabel('Frequency', fontsize=12)
plt.ylabel('Amplitude', fontsize=12)

plt.subplot(2, 2, 4)
plt.title('FFT of Analytic Signal (Absolute)', fontsize=a)
plt.plot(freq, np.abs(hlibert_fft))
plt.axvline(0, color='r', linestyle='--')
plt.xlim(-100, 100)
plt.xlabel('Frequency', fontsize=12)
plt.ylabel('Amplitude', fontsize=12)

plt.show()
