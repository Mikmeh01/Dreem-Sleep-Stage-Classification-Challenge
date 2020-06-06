import numpy as np
import tqdm
import pandas as pd

from FFT import get_fft_values
from PSD import get_psd_values
from auto_correlation import get_autocorr_values

from detect_peaks import detect_peaks

import tsfresh as ts
import tsfresh.feature_extraction.feature_calculators as calc
from statsmodels.tsa.stattools import acf
from scipy.stats import iqr,kurtosis


def get_first_n_peaks(x, y, no_peaks=5):
    """Returns 2 lists containing the first 'no_peaks' elements of the lists x, y
    if the length of x or y is less than 'no_peaks' then it replaces the last elements by 0 in the lists"""
    x_, y_ = list(x), list(y)
    if len(x_) >= no_peaks:
        return x_[:no_peaks], y_[:no_peaks]
    else:
        missing_no_peaks = no_peaks - len(x_)
        return x_ + [0] * missing_no_peaks, y_ + [0] * missing_no_peaks


def get_features(x_values, y_values, mph, no_peaks=5):
    """Returns a list of coordinates of the first no_peaks, for example:
    [x1, x2, x3, x4, x5, y1, y2, y3, y4, y5]"""
    indices_peaks = detect_peaks(y_values, mph=mph)
    peaks_x, peaks_y = get_first_n_peaks(x_values[indices_peaks], y_values[indices_peaks], no_peaks)
    return peaks_x + peaks_y


def extract_features_labels(dataset, labels, T, N, f_s, denominator):
    percentile = 5
    list_of_features = []
    list_of_labels = []
    print("length dataset:", len(dataset))
    for signal_no in tqdm.tqdm(range(len(dataset))):
        features = []
        list_of_labels.append(labels[signal_no])
        # for signal_comp in list_columns:

        signal = dataset[signal_no]
        signal_min = np.nanpercentile(signal, percentile)
        signal_max = np.nanpercentile(signal, 100 - percentile)
        # ijk = (100 - 2*percentile)/10
        mph = signal_min + (signal_max - signal_min) / denominator
        features += get_features(*get_fft_values(signal, T, N, f_s), mph)
        features += get_features(*get_psd_values(signal, T, N, f_s), mph)
        features += get_features(*get_autocorr_values(signal, T, N, f_s), mph)
        list_of_features.append(features)
    return np.array(list_of_features), np.array(list_of_labels)


# Creation of features names
base = ["max_fft_x_1", "max_fft_x_2", "max_fft_x_3", "max_fft_x_4", "max_fft_x_5",
        "max_fft_y_1", "max_fft_y_2", "max_fft_y_3", "max_fft_y_4", "max_fft_y_5",
        "max_psd_x_1", "max_psd_x_2", "max_psd_x_3", "max_psd_x_4", "max_psd_x_5",
        "max_psd_y_1", "max_psd_y_2", "max_psd_y_3", "max_psd_y_4", "max_psd_y_5",
        "max_cor_x_1", "max_cor_x_2", "max_cor_x_3", "max_cor_x_4", "max_cor_x_5",
        "max_cor_y_1", "max_cor_y_2", "max_cor_y_3", "max_cor_y_4", "max_cor_y_5"]

list_sensors = ['accelerometer_x', 'accelerometer_y', 'accelerometer_z', 'pulse_oximeter_infrared',
                'eeg_1', 'eeg_2', 'eeg_3', 'eeg_4', 'eeg_5', 'eeg_6', 'eeg_7']


def creation_features_names(sensor, base):
    features_names = []
    for name in base:
        new_name = name + "_" + sensor
        features_names.append(new_name)
    return features_names


def features_data(dataset, sensors):
    """create new features for training data"""
    index_array = np.array(range(0, 1000))
    result = pd.DataFrame({'ID': index_array})
    for sensor in sensors:
        file_train_sensor = dataset[sensor]
        if len(file_train_sensor[0]) == 1500:
            t_n = 30
            N = 1500
            T = t_n / N
            f_s = 1 / T
        elif len(file_train_sensor[0]) == 300:
            t_n = 30
            N = 300
            T = t_n / N
            f_s = 1 / T
        else:
            print("Problem in choice of T and N")
        x_train_, y_train_ = extract_features_labels(file_train_sensor, dataset['y'], T, N, f_s, denominator=10)
        x_train_ = pd.DataFrame(x_train_)
        x_train_.columns = creation_features_names(sensor, base)
        result = pd.concat([result, x_train_], axis=1, join_axes=[x_train_.index])
    y_train_ = dataset['y']
    result = pd.concat([result, y_train_], axis=1, join_axes=[result.index])
    result.to_csv("driss_features.csv")

def auto_corr_ind(x):
	"""Get the indice of the biggest auto_correlation lag"""
	return acf(x)[1:].argmax()

def gene_features(hdfs,nbr):
	#Function to generate 'nbr' of first features from a HDFS file
    df_train = pd.DataFrame()
    args = (2,3)
    for key in tqdm.tqdm(list(hdfs)[:11]):
        temp = np.stack(hdfs[key][0:nbr])
        #df_train['Mean %s'%key] = np.mean(temp, axis=1)
        df_train['Abs_Mean %s'%key] = np.mean(np.abs(temp), axis=1)
        df_train['Median %s'%key] = np.median(np.abs(temp), axis=1)
        #df_train['Min %s'%key] = np.min(temp, axis =1)
        df_train['Abs_Min %s'%key] = np.min(np.abs(temp), axis =1)
        #df_train['Max %s'%key] = np.max(temp, axis = 1)
        df_train['Abs_Max %s'%key] = np.max(np.abs(temp), axis = 1)
        df_train['Std %s'%key] = np.std(temp, axis = 1) 
        df_train['Energy %s'%key] = np.sum(np.square(temp),axis =1)
        df_train['Entropy %s'%key] = np.apply_along_axis(calc.approximate_entropy, 1, temp,*args)
        df_train['kurtosis_%s'%key] = kurtosis(temp,axis=1)
        df_train['IQR_%s'%key] = iqr(temp,axis=1)
    df_train.to_excel('Data/feat_%s.xlsx' %key)   
    
    	
