import numpy as np

def autocorr(x):
    result = np.correlate(x, x, mode='full')
    return result[len(result) // 2:]


def get_autocorr_values(y_values, T, N, f_s):
    """ Calculates the correlation of a signal with a time-delayed version of itself"""
    autocorr_values = autocorr(y_values)
    x_values = np.array([T * jj for jj in range(0, N)])
    return x_values, autocorr_values


t_n = 10
N = 1000
T = t_n / N
f_s = 1 / T

# t_values, autocorr_values = get_autocorr_values(composite_y_value, T, N, f_s)
#
# plt.plot(t_values, autocorr_values, linestyle='-', color='blue')
# plt.xlabel('time delay [s]')
# plt.ylabel('Autocorrelation amplitude')
# plt.show()