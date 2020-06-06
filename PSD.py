from scipy.signal import welch


def get_psd_values(y_values, T, N, f_s):
    """Returns two arrays, the first contains the frequency values, and the second the Power Spectral Density"""
    f_values, psd_values = welch(y_values, fs=f_s)
    return f_values, psd_values


t_n = 10
N = 1000
T = t_n / N
f_s = 1 / T

# f_values, psd_values = get_psd_values(composite_y_value, T, N, f_s)
#
# plt.plot(f_values, psd_values, linestyle='-', color='blue')
# plt.xlabel('Frequency [Hz]')
# plt.ylabel('PSD [V**2 / Hz]')
# plt.show()