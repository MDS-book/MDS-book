import matplotlib.pyplot as plt
import numpy as np


def pmfplot(ax, random_variable, probability, **kwargs):
    """Create a plot of discrete probabilities
    """
    color = kwargs.get('color', kwargs.get('c', 'C0'))
    alpha = kwargs.get('alpha', 0.5)
    linewidth = kwargs.get('linewidth', kwargs.get('lw', 4))
    mec = kwargs.get('markeredgecolor', kwargs.get('mec', color))
    mfc = kwargs.get('markerfacecolor', kwargs.get('mfc', '1'))
    mlw = float(kwargs.get('markerlinewidths', kwargs.get('mlw', '0.5')))
    marker = kwargs.get('marker', kwargs.get('m', 'o'))
    markersize = kwargs.get('markersize', kwargs.get('s', 20))
    linestyle = kwargs.get('linestyle', kwargs.get('ls', '-'))
    label = kwargs.get('label', '')
    ax.vlines(random_variable, 0, probability, colors=color, lw=linewidth, ls=linestyle, alpha=alpha, zorder=20)
    ax.scatter(random_variable, probability, color=color, marker=marker, s=markersize, edgecolors=mec, facecolors=mfc, linewidths=mlw, zorder=21, label=label )
    