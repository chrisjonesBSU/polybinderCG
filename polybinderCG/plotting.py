import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

def plot_distribution(
        data,
        label=None,
        plot=True,
        xlim=(None, None),
        line_plot=True,
        gaussian_fit=False,
        normalize=False,
        normalize_angles=False
        ):
    """
    """
    bin_heights, bin_borders = np.histogram(data, bins='auto')
    if normalize is True:
        bin_heights = [float(i)/sum(bin_heights) for i in bin_heights]
    bin_widths = np.diff(bin_borders)
    bin_centers = bin_borders[:-1] + bin_widths / 2
    if normalize_angles is True:
        bin_heights = [h/np.sin(bin_centers[i]) for i,h in enumerate(bin_heights)]

    if plot:
        plt.bar(bin_centers, bin_heights, width=bin_widths, label=label)
        plt.xlim(xlim)
        if label is not None:
            plt.legend()
        if line_plot:
            smoothed = np.convolve(bin_heights, np.ones(5), "same") / 5 
            plt.plot(bin_centers, smoothed, "-", linewidth=5)

    if line_plot:
        smoothed = np.convolve(bin_heights, np.ones(5), "same") / 5 
        plt.plot(bin_centers, smoothed, "-", linewidth=5)

    if gaussian_fit:
        popt, _ = curve_fit(gaussian, bin_centers, bin_heights, p0=[1., 0., 1.])
        x_fit = np.linspace(bin_borders[0], bin_borders[-1], 10000)
        y_fit = gaussian(x_fit, *popt)
        if plot:
            plt.plot(x_fit, y_fit, "red")
        return bin_centers, bin_heights, x_fit, y_fit

    return bin_centers, bin_heights

def gaussian(x, mean, amplitude, standard_deviation):
        return amplitude * np.exp( - (x - mean)**2 / (2*standard_deviation ** 2))
