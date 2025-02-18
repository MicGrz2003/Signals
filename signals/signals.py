import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy import signal as sig
from skimage.metrics import peak_signal_noise_ratio, mean_squared_error
from PyEMD import EMD
import pandas as pd  
from matplotlib.widgets import Slider
import tkinter as tk
from tkinter import ttk, filedialog

# Dane z biblioteki scipy

def signaltonoise(a, axis=0, ddof=0):
    a = np.asanyarray(a)
    m = a.mean(axis)
    sd = a.std(axis=axis, ddof=ddof)
    return np.where(sd == 0, 0, m/sd)
 

# Wizualizacja

def zad_1(amplituda, frekwencja, czest_prob, t_trwania, n):
    fig, ((ax1), (ax2), (ax3)) = plt.subplots(3, 1, figsize=(16, 8))

    fig.subplots_adjust(left=0.1, bottom=0.35, top=0.95, right=0.95, hspace=1.1, wspace=0.25)

    t = np.linspace(0, t_trwania, int(czest_prob * t_trwania), endpoint=False)
    signal = amplituda * sig.chirp(t, f0=frekwencja, t1=t_trwania, f1=frekwencja * 10)
    szum = np.random.normal(0, n, len(signal))
    signal_with_noise = signal + szum

    ax1.plot(t, signal, 'b')
    ax1.set_title('Sygnał')

    ax2.plot(t, szum, 'g')
    ax2.set_title('Szum')

    ax3.plot(t, signal_with_noise, 'r')
    ax3.set_title('Sygnał + Szum')

    ax_amplituda = plt.axes([0.1, 0.15, 0.8, 0.03], facecolor='lightgoldenrodyellow')
    ax_frekwencja = plt.axes([0.1, 0.10, 0.8, 0.03], facecolor='lightgoldenrodyellow')
    ax_czest_prob = plt.axes([0.1, 0.08, 0.8, 0.03], facecolor='lightgoldenrodyellow')
    ax_t_trwania = plt.axes([0.1, 0.05, 0.8, 0.03], facecolor='lightgoldenrodyellow')
    ax_n = plt.axes([0.1, 0.025, 0.8, 0.03], facecolor='lightgoldenrodyellow')

    amplitude_slider = Slider(ax_amplituda, 'Amplituda', 0.1, 2.0, valinit=amplituda, valstep=0.1)
    frequency_slider = Slider(ax_frekwencja, 'Częstotliwość', 1, 100, valinit=frekwencja, valstep=1)
    sample_rate_slider = Slider(ax_czest_prob, 'Częstotliwość próbkowania', 100, 2000, valinit=czest_prob, valstep=100)
    duration_slider = Slider(ax_t_trwania, 'Czas trwania', 1, 10, valinit=t_trwania, valstep=1)
    noise_slider = Slider(ax_n, 'Szum', 1, 50, valinit=n, valstep=1)

    def update(val):
        amplituda = amplitude_slider.val
        frekwencja = frequency_slider.val
        czest_prob = sample_rate_slider.val
        t_trwania = duration_slider.val
        n = noise_slider.val

        t = np.linspace(0, t_trwania, int(czest_prob * t_trwania), endpoint=False)
        signal = amplituda * sig.chirp(t, f0=frekwencja, t1=t_trwania, f1=frekwencja * 10)
        szum = np.random.normal(0, n, len(signal))
        signal_with_noise = signal + szum

        ax1.clear()
        ax1.plot(t, signal, 'b')
        ax1.set_title('Sygnał')

        ax2.clear()
        ax2.plot(t, szum, 'g')
        ax2.set_title('Szum')

        ax3.clear()
        ax3.plot(t, signal_with_noise, 'r')
        ax3.set_title('Sygnał + Szum')

        fig.canvas.draw_idle()

    amplitude_slider.on_changed(update)
    frequency_slider.on_changed(update)
    sample_rate_slider.on_changed(update)
    duration_slider.on_changed(update)
    noise_slider.on_changed(update)

    plt.show()

zad_1(1, 10, 1000, 2, 10)

def zadd_1(amplituda, frekwencja, czest_prob, t_trwania, n):
    t = np.linspace(0, t_trwania, int(czest_prob * t_trwania), endpoint=False)
    signal = amplituda * sig.chirp(t, f0=frekwencja, t1=t_trwania, f1=frekwencja * 10)
    szum = np.random.normal(0, n, len(signal))
    signal_power = np.sum(np.square(signal))
    noise_power = np.sum(np.square(szum))

    # SNR
    snr = 20 * np.log10(signal_power / noise_power)
    print("SNR:", snr)

    # MSE
    mse = np.mean((signal - szum) ** 2)
    print("MSE:", mse)

    # PSNR
    psnr = 20 * np.log10(np.abs(signal).max() / np.sqrt(mse))
    print("PSNR:", psnr)

zadd_1(1, 10, 1000, 2, 10)

print("\nZadanie 1 i 2\n")

def zad_2(amplituda, frekwencja, czest_prob, t_trwania, n):
    t = np.linspace(0, t_trwania, int(czest_prob * t_trwania), endpoint=False)
    signal = amplituda * sig.chirp(t, f0=frekwencja, t1=t_trwania, f1=frekwencja * 10)
    szum = np.random.normal(0, n, len(signal))

    signal_power = np.sum(signal ** 2)
    noise_power = np.sum(szum ** 2)

    # SNR
    snr = 20 * np.log10(signal_power / noise_power)

    # MSE
    mse = np.mean((signal - szum) ** 2)

    # PSNR
    psnr = 20 * np.log10(np.abs(signal).max() / np.sqrt(mse))

    return snr, mse, psnr

def compare_metrics(amplituda, frekwencja, czest_prob, t_trwania, n):
    snr_zad1, mse_zad1, psnr_zad1 = zad_2(amplituda, frekwencja, czest_prob, t_trwania, n)

    t = np.linspace(0, t_trwania, int(czest_prob * t_trwania), endpoint=False)
    signal = amplituda * sig.chirp(t, f0=frekwencja, t1=t_trwania, f1=frekwencja * 10)
    szum = np.random.normal(0, n, len(signal))

    snr_builtin = signaltonoise(signal + szum)
    mse_builtin = mean_squared_error(signal, signal + szum)
    psnr_builtin = peak_signal_noise_ratio(signal, signal + szum, data_range=np.abs(signal).max())

    print("SNR (zad1):", snr_zad1)
    print("SNR (biblioteka):", snr_builtin)
    print("MSE (zad1):", mse_zad1)
    print("MSE (biblioteka):", mse_builtin)
    print("PSNR (zad1):", psnr_zad1)
    print("PSNR (biblioteka):", psnr_builtin)

compare_metrics(1, 10, 1000, 2, 10)

def zad_4(szum_syg, t, snr_db, oryg):
    dsignal = sig.wiener(szum_syg)

    plt.figure(figsize=(10, 6))

    plt.subplot(3,1,1)
    plt.plot(t, szum_syg, label='zaszumiony sygnal')
    plt.xlabel('czas')
    plt.ylabel('Amplituda')
    plt.title('swiergotliwy z zadania 3 zaszumiony (SNR = {} dB)'.format(snr_db))
    plt.legend()
    plt.grid(True)

    plt.subplot(3,1,2)
    plt.plot(t, oryg, label='czysty sygnal')
    plt.xlabel('czas')
    plt.ylabel('Amplituda')
    plt.title('swiergotliwy z zadania 3 niezaszumiony')
    plt.legend()
    plt.grid(True)

    plt.subplot(3,1,3)
    plt.plot(t, dsignal, label='odszumiony sygnal')
    plt.xlabel('czas')
    plt.ylabel('Amplituda')
    plt.title('swiergotliwy z zadania 3 odszumianie za pomoca wienera (SNR = {} dB)'.format(snr_db))
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def zad_5(szum_syg, t, snr_db, oryg):
    window_length = 51  
    polyorder = 3
    d = sig.savgol_filter(szum_syg, window_length, polyorder)

    plt.figure(figsize=(10, 6))

    plt.subplot(3,1,1)
    plt.plot(t, szum_syg, label='zaszumiony sygnal')
    plt.xlabel('czas')
    plt.ylabel('Amplituda')
    plt.title('swiergotliwy z zadania 3 zaszumiony (SNR = {} dB)'.format(snr_db))
    plt.legend()
    plt.grid(True)

    plt.subplot(3,1,2)
    plt.plot(t, oryg, label='czysty sygnal')
    plt.xlabel('czas')
    plt.ylabel('Amplituda')
    plt.title('swiergotliwy z zadania 3 niezaszumiony')
    plt.legend()
    plt.grid(True)

    plt.subplot(3,1,3)
    plt.plot(t, d, label='odszumiony sygnal')
    plt.xlabel('czas')
    plt.ylabel('Amplituda')
    plt.title('swiergotliwy z zadania 3 odszumianie za pomoca savitzky-golay (SNR = {} dB)'.format(snr_db))
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

def zad_6(szum_syg, t, snr_db, oryg):
    emd = EMD()
    IMFs = emd(szum_syg)

    kept_IMFs = range(min(5, len(IMFs)))

    syngal_rek = np.sum(IMFs[kept_IMFs], axis=0)

    plt.figure(figsize=(10, 6))

    plt.subplot(3,1,1)
    plt.plot(t, szum_syg, label='zaszumiony sygnal')
    plt.xlabel('czas')
    plt.ylabel('Amplituda')
    plt.title('swiergotliwy z zadania 3 zaszumiony (SNR = {} dB)'.format(snr_db))
    plt.legend()
    plt.grid(True)

    plt.subplot(3,1,2)
    plt.plot(t, oryg, label='czysty sygnal')
    plt.xlabel('czas')
    plt.ylabel('Amplituda')
    plt.title('swiergotliwy z zadania 3 niezaszumiony')
    plt.legend()
    plt.grid(True)

    plt.subplot(3,1,3)
    plt.plot(t, syngal_rek, label='odszumiony sygnal')
    plt.xlabel('czas')
    plt.ylabel('Amplituda')
    plt.title('swiergotliwy z zadania 3 odszumianie za pomoca emd i rekonstrukcji (SNR = {} dB)'.format(snr_db))
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

def zad_3(amplitude, frequency, sampling_rate, duration, white_noise_std, brown_noise_std, snr_db):
    t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)

    swiergot = amplitude * sig.chirp(t, f0=frequency, t1=duration, f1=frequency * 10)

    white = np.random.normal(0, white_noise_std, len(t))

    brown = np.cumsum(np.random.normal(0, brown_noise_std, len(t)))
    brown -= np.mean(brown)
    brown *= (brown_noise_std / np.std(brown))

    signal_power = np.sum(np.square(swiergot))
    desired_noise_power = signal_power / (10 ** (snr_db / 10))
    current_noise_power = np.sum(np.square(white + brown))
    scaling_factor = np.sqrt(desired_noise_power / current_noise_power)
    white *= scaling_factor
    brown *= scaling_factor

    szum_syg = swiergot + white + brown

    plt.figure(figsize=(10, 6))
    plt.subplot(2,2,1)
    plt.plot(t, szum_syg, label='zaszumiony sygnal')
    plt.xlabel('czas')
    plt.ylabel('Amplituda')
    plt.title('swiergotliwy z szumami browna i bialym (SNR = {} dB)'.format(snr_db))
    plt.legend()
    plt.grid(True)

    plt.subplot(2,2,2)
    plt.plot(t, swiergot, label='czysty sygnal')
    plt.xlabel('czas')
    plt.ylabel('Amplituda')
    plt.title('swiergotliwy')
    plt.legend()
    plt.grid(True)

    plt.subplot(2,2,3)
    plt.plot(t, white, label='White szum', c='red')
    plt.xlabel('czas')
    plt.ylabel('Amplituda')
    plt.title('bialy szum')
    plt.legend()
    plt.grid(True)

    plt.subplot(2,2,4)
    plt.plot(t, brown, label='Brown szum', c='brown')
    plt.xlabel('czas')
    plt.ylabel('Amplituda')
    plt.title('szum browna')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    zad_4(szum_syg, t, snr_db, swiergot)
    zad_5(szum_syg, t, snr_db, swiergot)
    zad_6(szum_syg, t, snr_db, swiergot)

def create_task3_gui(root):
    for widget in root.winfo_children():
        widget.destroy()

    label_amplitude = tk.Label(root, text='Amplituda:')
    label_amplitude.grid(row=0, column=0)
    slider_amplitude = tk.Scale(root, from_=1.0, to=10.0, orient=tk.HORIZONTAL, resolution=0.1)
    slider_amplitude.grid(row=0, column=1)

    label_frequency = tk.Label(root, text='Częstotliwość:')
    label_frequency.grid(row=1, column=0)
    slider_frequency = tk.Scale(root, from_=1.0, to=20.0, orient=tk.HORIZONTAL, resolution=0.1)
    slider_frequency.grid(row=1, column=1)

    label_sampling_rate = tk.Label(root, text='Częstotliwość próbkowania:')
    label_sampling_rate.grid(row=2, column=0)
    slider_sampling_rate = tk.Scale(root, from_=100, to=1000, orient=tk.HORIZONTAL, resolution=10)
    slider_sampling_rate.grid(row=2, column=1)

    label_duration = tk.Label(root, text='Czas trwania:')
    label_duration.grid(row=3, column=0)
    slider_duration = tk.Scale(root, from_=1, to=10, orient=tk.HORIZONTAL, resolution=1)
    slider_duration.grid(row=3, column=1)

    label_white_noise_std = tk.Label(root, text='Odchylenie standardowe białego szumu:')
    label_white_noise_std.grid(row=4, column=0)
    slider_white_noise_std = tk.Scale(root, from_=0.1, to=10.0, orient=tk.HORIZONTAL, resolution=0.1)
    slider_white_noise_std.grid(row=4, column=1)

    label_brown_noise_std = tk.Label(root, text='Odchylenie standardowe szumu Browna:')
    label_brown_noise_std.grid(row=5, column=0)
    slider_brown_noise_std = tk.Scale(root, from_=0.1, to=10.0, orient=tk.HORIZONTAL, resolution=0.1)
    slider_brown_noise_std.grid(row=5, column=1)

    label_snr_db = tk.Label(root, text='SNR (dB):')
    label_snr_db.grid(row=6, column=0)
    slider_snr_db = tk.Scale(root, from_=1, to=30, orient=tk.HORIZONTAL, resolution=1)
    slider_snr_db.grid(row=6, column=1)

    button_generate = ttk.Button(root, text="Generuj sygnał", command=lambda: zad_3(slider_amplitude.get(), slider_frequency.get(), slider_sampling_rate.get(), slider_duration.get(), slider_white_noise_std.get(), slider_brown_noise_std.get(), slider_snr_db.get()))
    button_generate.grid(row=7, columnspan=2, pady=10)

def z_7():
    df_signal = pd.read_csv('moj_sygnal_2.csv')
    df_ns = pd.read_csv('noisy_signal.csv')

    t = df_signal['Time'].values[:100]
    signal = df_signal['Signal'].values[:100]
    szum_syg = df_ns['NoisySignal'].values[:100]

    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.plot(t, szum_syg, label='zaszumiony sygnal')
    plt.xlabel('czas')
    plt.ylabel('Amplituda')
    plt.title('swiergotliwy z szumami')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(t, signal, label='czysty sygnal')
    plt.xlabel('czas')
    plt.ylabel('Amplituda')
    plt.title('swiergotliwy')
    plt.legend()
    plt.grid(True)


    plt.tight_layout()
    plt.show()

    zad_4(szum_syg, t, 5, signal)
    zad_5(szum_syg, t, 5, signal)
    zad_6(szum_syg, t, 5, signal)

# Funkcja tworząca główne GUI
def create_main_gui():
    root = tk.Tk()
    root.title("Wybór Zadania")

    label = tk.Label(root, text="Wybierz zadanie do uruchomienia:")
    label.pack(pady=10)

    button_task3 = ttk.Button(root, text="Zadanie 3", command=lambda: create_task3_gui(root))
    button_task3.pack(pady=5)

    button_task7 = ttk.Button(root, text="Zadanie 7", command=z_7)
    button_task7.pack(pady=5)

    root.mainloop()

create_main_gui()