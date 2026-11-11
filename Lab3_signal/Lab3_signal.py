import numpy as np
import time

def dft(x):
    N = len(x)
    X = np.zeros(N, dtype=complex)
    additions = 0
    multiplications = 0
    
    for k in range(N):
        for n in range(N):
            W = np.exp(-2j * np.pi * k * n / N)
            X[k] += x[n] * W
            multiplications += 1  
            if n > 0:
                additions += 1    
    return X, additions, multiplications

def fft(x, counters=None):
    if counters is None:
        counters = {'additions': 0, 'multiplications': 0}
        
    N = len(x)
    if N <= 1:
        return x, counters

    even, _ = fft(x[0::2], counters)
    odd, _ = fft(x[1::2], counters)

    T = np.zeros(N // 2, dtype=complex)
    for k in range(N // 2):
        W = np.exp(-2j * np.pi * k / N)
        T[k] = W * odd[k]
        counters['multiplications'] += 1  

    X = np.zeros(N, dtype=complex)
    for k in range(N // 2):
        X[k] = even[k] + T[k]
        X[k + N // 2] = even[k] - T[k]
        counters['additions'] += 2  

    return X, counters

def run_lab_comparison():
    print("     Порівняння ДПФ та ШПФ")
    
    N_original = 28
    np.random.seed(42) 
    base_signal = np.random.rand(N_original) + 1j * np.random.rand(N_original)
    
    N_padded = 1 << (N_original - 1).bit_length()
    
    padded_signal = np.pad(base_signal, (0, N_padded - N_original), 'constant')

    print(f" Оригінальний розмір вибірки: N = {N_original} (Для ДПФ)")
    print(f" Доповнений розмір вибірки:   N = {N_padded} (Для ШПФ)")
    
    start_time_dft = time.perf_counter()
    X_dft, dft_add, dft_mult = dft(base_signal)
    time_dft = time.perf_counter() - start_time_dft

    start_time_fft = time.perf_counter()
    X_fft, fft_counters = fft(padded_signal)
    time_fft = time.perf_counter() - start_time_fft
    
    fft_add = fft_counters['additions']
    fft_mult = fft_counters['multiplications']

    print(f"{'Показник':<25} | {'ДПФ':<15} | {'ШПФ':<15}")
    print("-" * 65)
    print(f"{'Час обчислення (сек)':<25} | {time_dft:<15.6f} | {time_fft:<15.6f}")
    print(f"{'Кількість додавань':<25} | {dft_add:<15} | {fft_add:<15}")
    print(f"{'Кількість множень':<25} | {dft_mult:<15} | {fft_mult:<15}")
    print(f"{'Загальна к-ть операцій':<25} | {dft_add+dft_mult:<15} | {fft_add+fft_mult:<15}")

    
    efficiency = (dft_add + dft_mult) / (fft_add + fft_mult)
    print(f"ШПФ потребує у {efficiency:.2f} разів менше математичних операцій.")

    np_dft_check = np.fft.fft(base_signal)
    np_fft_check = np.fft.fft(padded_signal)
    
    dft_correct = np.allclose(X_dft, np_dft_check)
    fft_correct = np.allclose(X_fft, np_fft_check)
    print(f"Коректність ДПФ (з еталоном numpy): {'Так' if dft_correct else 'Ні'}")
    print(f"Коректність ШПФ (з еталоном numpy): {'Так' if fft_correct else 'Ні'}")

if __name__ == "__main__":
    run_lab_comparison()