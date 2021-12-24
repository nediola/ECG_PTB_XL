import numpy as np
import pandas as pd
import neurokit2 as nk
import multiprocessing

def get_intervals(x, y=None):
    if y is not None:
        min_len = min(len(x), len(y))
        return y[:min_len] - x[:min_len]
    elif len(x) > 1:
        return x[1:] - x[:-1]
    return np.array([])

def get_distr_features(x, sr=100):
    if not len(x):
        return 0, 0, 0, 0
    norm_coef = sr if sr else 1.0
    fmax = np.max(x) / norm_coef
    fmin = np.min(x) / norm_coef
    fmean = np.mean(x) / norm_coef
    fstd = np.std(x) / norm_coef
    return fmax, fmin, fmean, fstd

def get_ecg_features(ecg, sr=100):
    try:
        df, info = nk.ecg_process(ecg, sampling_rate=sr)
    except Exception as ex:
        return np.zeros(49)
    
    if df.ECG_Rate.values[0] is not None:
        mean_rate = df.ECG_Rate.mean()
    else:
        mean_rate = 0
    
    # peaks features
    P_max, P_min, P_mean, P_std = get_distr_features(df[df.ECG_P_Peaks == 1].ECG_Clean.values)
    Q_max, Q_min, Q_mean, Q_std = get_distr_features(df[df.ECG_Q_Peaks == 1].ECG_Clean.values)
    R_max, R_min, R_mean, R_std = get_distr_features(df[df.ECG_R_Peaks == 1].ECG_Clean.values)
    S_max, S_min, S_mean, S_std = get_distr_features(df[df.ECG_S_Peaks == 1].ECG_Clean.values)
    T_max, T_min, T_mean, T_std = get_distr_features(df[df.ECG_T_Peaks == 1].ECG_Clean.values)

    # intervals features
    P_peaks = df[df.ECG_P_Peaks == 1].index
    Q_peaks = df[df.ECG_Q_Peaks == 1].index
    R_peaks = df[df.ECG_R_Peaks == 1].index
    S_peaks = df[df.ECG_S_Peaks == 1].index
    T_peaks = df[df.ECG_T_Peaks == 1].index

    RR_max, RR_min, RR_mean, RR_std  = get_distr_features(get_intervals(R_peaks), sr=sr)
    PQ_max, PQ_min, PQ_mean, PQ_std  = get_distr_features(get_intervals(P_peaks, Q_peaks), sr=sr)
    QRS_max, QRS_min, QRS_mean, QRS_std  = get_distr_features(get_intervals(Q_peaks, S_peaks), sr=sr)
    QT_max, QT_min, QT_mean, QT_std = get_distr_features(get_intervals(Q_peaks, T_peaks), sr=sr)

    return np.array([
        mean_rate,
        RR_min/RR_max if RR_max else 0,
        RR_mean/RR_max if RR_max else 0,
        RR_min/RR_mean if RR_mean else 0,
        PQ_min/PQ_max if PQ_max else 0,
        PQ_mean/PQ_max if PQ_max else 0,
        PQ_min/PQ_mean if PQ_mean else 0,
        QRS_min/QRS_max if QRS_max else 0,
        QRS_mean/QRS_max if QRS_max else 0,
        QRS_min/QRS_mean if QRS_mean else 0,
        QT_min/QT_max if QT_max else 0,
        QT_mean/QT_max if QT_max else 0,
        QT_min/QT_mean if QT_mean else 0,
        P_max, P_min, P_mean, P_std,
        Q_max, Q_min, Q_mean, Q_std,
        R_max, R_min, R_mean, R_std,
        S_max, S_min, S_mean, S_std,
        T_max, T_min, T_mean, T_std,
        RR_max, RR_min, RR_mean, RR_std,
        PQ_max, PQ_min, PQ_mean, PQ_std,
        QRS_max, QRS_min, QRS_mean, QRS_std,
        QT_max, QT_min, QT_mean, QT_std
    ]);

def get_N_ecg_features(X, sr=100):
    return np.apply_along_axis(get_ecg_features, 1, X, 100)
    
def parallel_apply_on_samples(func, X):
    pool = multiprocessing.Pool()
    chunks = [sub_arr for sub_arr in np.array_split(X, int(multiprocessing.cpu_count()/2))]
    individual_results = pool.map(func, chunks)
    pool.close()
    pool.join()
    return np.concatenate(individual_results)

def calc_features(X, path, channels=12):
    # saves parallel-processed parts
    for i in range(0, channels):
        print(f'Start process {i}-th channel')
        X_meta = parallel_apply_on_samples(get_N_ecg_features, X[:, :, i])
        np.save(f'{path}_{i}.npy', X_meta)
    return

def load_features(path, channels=12):
    # loads & aggregates parts
    X_meta = np.load(f'{path}_0.npy')
    X_meta = X_meta[..., None]
    for i in range(1, channels):
        X_meta_i = np.load(f'{path}_{i}.npy')
        X_meta_i = X_meta_i[..., None]
        X_meta = np.concatenate((X_meta, X_meta_i), axis=2)
    X_meta = X_meta.transpose(0, 2, 1)
    X_meta = X_meta.reshape(X_meta.shape[0], -1) # Now it contains channels features one after another
    return X_meta