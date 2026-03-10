import numpy as np
import math
from collections import OrderedDict
from os.path import join

# Plotting
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import rcParams, colors
from matplotlib.colors import LinearSegmentedColormap
import cartopy.crs as ccrs
import cartopy.feature as cf

from src.utilities import utils
project_dir, config = utils.setup()

# Other font details
rcParams['font.size'] = config['label_fontsize']*config['scale']
rcParams['axes.titlepad'] = config['title_pad']

from matplotlib.font_manager import findfont, FontProperties
font = findfont(FontProperties(family=['sans-serif']))


def color(k, cmap='CMRmap', lut=10):
    c = plt.cm.get_cmap(cmap, lut=lut)
    return colors.to_hex(c(k))


def get_figsize(aspect, rows, cols, **fig_kwargs):
    # Set default kwarg values
    max_width = fig_kwargs.get(
        'max_width', 
        config['base_width']*config['scale']
    )*cols
    max_height = fig_kwargs.get(
        'max_height', 
        config['base_height']*config['scale']
    )*rows

    # Get figsize
    if aspect > 1: # width > height
        figsize = (max_width,
                   max_width/aspect)
    else: # width < height
        figsize = (max_height*aspect,
                   max_height)
    return figsize


def get_aspect(rows, cols, aspect=None,
               maps=False, lats=None, lons=None):
    if maps:
        aspect = np.cos(np.mean([np.min(lats), np.max(lats)])*np.pi/180)
        xsize = np.ptp([np.max(lons), np.min(lons)])*aspect
        ysize = np.ptp([np.max(lats), np.min(lats)])
        aspect = xsize/ysize
    return aspect*cols/rows


def make_axes(rows=1, cols=1, aspect=None,
              maps=False, lats=None, lons=None,
              **fig_kwargs):
    aspect = get_aspect(rows, cols, aspect, maps, lats, lons)
    figsize = get_figsize(aspect, rows, cols, **fig_kwargs)
    kw = {}
    if maps:
        kw['subplot_kw'] = {'projection' : ccrs.PlateCarree()}
    kw['sharex'] = fig_kwargs.pop('sharex', True)
    kw['sharey'] = fig_kwargs.pop('sharey', False)
    kw['width_ratios'] = fig_kwargs.pop('width_ratios', None)
    kw['height_ratios'] = fig_kwargs.pop('height_ratios', None)
    fig, ax = plt.subplots(rows, cols, figsize=figsize, **kw)
    return fig, ax


def add_cax(fig, ax, cbar_pad_inches=0.25, horizontal=False):
    # TODO: should be updated to infer cbar width and cbar_pad_inches
    if not horizontal:
        try:
            axis = ax[-1, -1]
            height = ax[0, -1].get_position().y1 - ax[-1, -1].get_position().y0
            ax_width = ax[0, -1].get_position().x1 - ax[0, 0].get_position().x0
        except IndexError:
            axis = ax[-1]
            height = ax[0].get_position().y1 - ax[-1].get_position().y0
            ax_width = ax[-1].get_position().x1 - ax[0].get_position().x0
        except TypeError:
            axis = ax
            height = ax.get_position().height
            ax_width = ax.get_position().width

        # x0
        fig_width = fig.get_size_inches()[0]
        x0_init = axis.get_position().x1
        x0 = (fig_width*x0_init + cbar_pad_inches*config['scale'])/fig_width

        # y0
        y0 = axis.get_position().y0

        # Width
        width = 0.1*config['scale']/fig_width
    else:
        try:
            axis = ax[-1, 0]
            width = ax[-1, -1].get_position().x1 - ax[-1, 0].get_position().x0
        except IndexError:
            axis = ax[0]
            width = ax[-1].get_position().x1 - ax[0].get_position().x0
        except TypeError:
            axis = ax
            width = ax.get_position().width

        # x0
        x0 = axis.get_position().x0

        # y0
        fig_height = fig.get_size_inches()[1]
        y0_init = axis.get_position().y0
        y0 = (fig_height*y0_init - cbar_pad_inches*config['scale'])/fig_height

        # Height
        height = 0.1*config['scale']/fig_height

    # Make axis
    cax = fig.add_axes([x0, y0, width, height])

    return cax


def get_figax(rows=1, cols=1, aspect=4,
              maps=False, lats=None, lons=None,
              figax=None, **fig_kwargs):
    if figax is not None:
        fig, ax = figax
    else:
        fig, ax = make_axes(rows, cols, aspect, maps, lats, lons, **fig_kwargs)

        if (rows > 1) or (cols > 1):
            for axis in ax.flatten():
                axis.set_facecolor('0.98')
                if maps:
                    axis = format_map(axis, lats=lats, lons=lons)
        else:
            ax.set_facecolor('0.98')
            if maps:
                ax = format_map(ax, lats=lats, lons=lons)

    return fig, ax


def add_labels(ax, xlabel, ylabel, **label_kwargs):
    # Set default values
    label_kwargs['fontsize'] = label_kwargs.get(
        'fontsize',
        config['label_fontsize']*config['scale']
    )
    label_kwargs['labelpad'] = label_kwargs.get(
        'labelpad',
        config['label_pad']
    )
    labelsize = label_kwargs.pop(
        'labelsize',
        config['tick_fontsize']*config['scale']
    )

    # Set labels
    ax.set_xlabel(xlabel, **label_kwargs)
    ax.set_ylabel(ylabel, **label_kwargs)
    ax.tick_params(axis='both', which='both', labelsize=labelsize)
    return ax


def add_legend(ax, **legend_kwargs):
    legend_kwargs['frameon'] = legend_kwargs.get('frameon', False)
    legend_kwargs['fontsize'] = legend_kwargs.get(
        'fontsize',
        config['label_fontsize']*config['scale']
    )

    # Remove duplicates from legend
    try:
        handles = legend_kwargs.pop('handles')
        labels = legend_kwargs.pop('labels')
    except:
        handles, labels = ax.get_legend_handles_labels()
    labels = OrderedDict(zip(labels, handles))
    handles = labels.values()
    labels = labels.keys()

    ax.legend(handles=handles, labels=labels, **legend_kwargs)
    return ax


def add_title(ax, title, **title_kwargs):
    title_kwargs['y'] = title_kwargs.get('y', config['title_loc'])
    title_kwargs['pad'] = title_kwargs.get('pad', config['title_pad'])
    title_kwargs['fontsize'] = title_kwargs.get(
        'fontsize',
        config['title_fontsize']*config['scale']
    )
    title_kwargs['va'] = title_kwargs.get('va', 'bottom')
    ax.set_title(title, **title_kwargs)
    return ax


def get_square_limits(xdata, ydata, **kw):
    # Get data limits
    dmin = min(np.min(xdata), np.min(ydata))
    dmax = max(np.max(xdata), np.max(ydata))
    pad = (dmax - dmin)*0.05
    dmin -= pad
    dmax += pad

    try:
        # get lims
        ylim = kw.pop('lims')
        xlim = ylim
        xy = (min(xlim[0], ylim[0]), max(xlim[1], ylim[1]))
    except:
        # set lims
        xlim = ylim = xy = (dmin, dmax)

    return xlim, ylim, xy, dmin, dmax


def format_cbar(cbar, cbar_title='', horizontal=False, **cbar_kwargs):
    if horizontal:
        x = 0.5
        y = cbar_kwargs.pop('y', -4)
        rotation = 'horizontal'
        va = 'top'
        ha = 'center'
    else:
        x = cbar_kwargs.pop('x', 5)
        y = 0.5
        rotation = 'vertical'
        va = 'center'
        ha = 'left'

    cbar.ax.tick_params(axis='both', which='both',
                        labelsize=config['tick_fontsize']*config['scale'])
    cbar.ax.text(x, y, cbar_title, ha=ha, va=va, rotation=rotation,
                 fontsize=config['label_fontsize']*config['scale'],
                 transform=cbar.ax.transAxes)

    return cbar


def save_fig(fig, loc, name, **kwargs):
    fig.savefig(join(loc, name + '.png'),
                bbox_inches='tight', dpi=500,
                transparent=True, **kwargs)
    print('Saved %s' % name + '.png')


def format_plot(fig, ax, nstate, **fig_kwargs):
    # Deal with difference in axis shapes
    if type(ax) == np.ndarray:
        if len(ax.shape) == 1:
            ncols = 1
        else:
            ncols = ax.shape[1]
    else:
        ncols = 1
        ax = [ax]

    # Formatting
    for axis in ax.flatten():
        for i in range(nstate+2):
            # ncols = 1 --> lw = 1      1 - 0*0.25
            # ncols = 2 --> lw = 0.75   1 - math.log2(2)*0.25
            # ncols = 4 --> lw = 0.5    1 - math.log2(4)*0.25
            # ncols = 8 --> lw = 0.25   1 - math.log2(8)*0.25
            axis.axvline(i-0.5, c=color(1), alpha=0.2, ls=':', 
                         lw=1-math.log2(ncols)*0.25)
        axis.set_xticks(np.arange(0, nstate+1, 2))
        axis.set_xlim(0.5, nstate+0.5)
        axis.set_facecolor('white')

        # Adjust aspect
        xs = axis.get_xlim()
        ys = axis.get_ylim()
        axis.set_aspect(0.25*(xs[1]-xs[0])/(ys[1]-ys[0]), adjustable='box')
    if len(ax) == 1:
        return fig, ax[0]
    else:
        return fig, ax


def plot_summary(inv_obj, inv_obj_2=None, ls='-', figax=None):
    if figax is None:
        fig, ax = get_figax(rows=2)
    else:
        fig, ax = figax
    add_title(ax[0], 'Base inversion')
    add_title(ax[1], 'Observations')
    fig.subplots_adjust(hspace=0.5)

    # Plot "true " emissions
    xt_abs = inv_obj.xt_abs
    if inv_obj_2 is not None:
        xt_abs = inv_obj_2.xt_abs - inv_obj.xt_abs
    ax[0].plot(inv_obj.xp, xt_abs, c=color(2), ls=ls, label='Truth')

    # Plot the prior
    xa_abs = inv_obj.xa_abs
    if inv_obj_2 is not None:
        xa_abs = inv_obj_2.xa_abs - inv_obj.xa_abs
    ax[0].plot(
        inv_obj.xp, xa_abs, 
        c=color(4), marker='.', markersize=10, ls=ls,
        label=r'Prior($\pm$ 50%)')
    if inv_obj_2 is None:
        ax[0].fill_between(
            inv_obj.xp, 
            inv_obj.xa_abs - inv_obj.xa_abs*inv_obj.sa**0.5,
            inv_obj.xa_abs + inv_obj.xa_abs*inv_obj.sa**0.5,
            color=color(4), alpha=0.2, zorder=-1
        )

    # Plot the posterior
    xhat_abs = inv_obj.xhat * inv_obj.xa_abs
    if inv_obj_2 is not None:
        xhat_abs = (inv_obj_2.xhat * inv_obj_2.xa_abs - 
                    inv_obj.xhat * inv_obj.xa_abs)
    ax[0].plot(
        inv_obj.xp, xhat_abs, 
        ls=ls, marker='*', markersize=10,
        c=color(6), label=f'Posterior'
    )

    handles_0, labels_0 = ax[0].get_legend_handles_labels()
    ax[0] = add_labels(ax[0], '', 'Emissions\n(ppb/day)')

    # Observations
    xp = inv_obj.xp
    y0 = inv_obj.y0[-inv_obj.nstate:]
    y = inv_obj.y.reshape(inv_obj.nobs_per_cell, inv_obj.nstate).T
    so = inv_obj.so.reshape(inv_obj.nobs_per_cell, inv_obj.nstate).T**0.5
    if inv_obj_2 is not None:
        xp = inv_obj_2.xp
        y0 = inv_obj_2.y0[-inv_obj_2.nstate:]
        y = inv_obj_2.y.reshape(inv_obj_2.nobs_per_cell, inv_obj_2.nstate).T
        so = inv_obj_2.so.reshape(inv_obj_2.nobs_per_cell, 
                                  inv_obj_2.nstate).T**0.5
    ax[1].plot(xp, y0, c='black', ls='-', label='Steady state', zorder=10)
    ax[1].plot(xp, y, c='grey', ls='-', label='Observations', lw=0.5, zorder=9)

    # Error range
    y_err_min = (y - so).min(axis=1)
    y_err_max = (y + so).max(axis=1)
    ax[1].fill_between(xp, y_err_min, y_err_max, color='grey', alpha=0.2)
    handles_1, labels_1 = ax[1].get_legend_handles_labels()
    handles_0.extend(handles_1)
    labels_0.extend(labels_1)

    # Aesthetics
    ax[1] = add_legend(ax[1], handles=handles_0, labels=labels_0,
                        bbox_to_anchor=(0.9, 0.5), loc='center left', ncol=1,
                        bbox_transform=fig.transFigure)
    ax[1] = add_labels(ax[1], 'State vector element', 'XCH4\n(ppb)')

    fig, ax = format_plot(fig, ax, inv_obj.nstate)

    return fig, ax


def plot_difference(orig, delta):
    fig, ax = get_figax(rows=2, cols=2)
    fig, ax[:, 0] = plot_summary(orig, ls='-', figax=[fig, ax[:, 0]])
    fig, ax[:, 1] = plot_summary(orig, delta, ls=':', figax=[fig, ax[:, 1]])

    # Set ylim for emissions plot
    ylim = np.abs(ax[0, 1].get_ylim()).max()
    ax[0, 1].set_ylim(-ylim, ylim)
    ax[0, 1].axhline(0, color='grey', lw=0.5, ls=':')

    # Set ylim for observations plot
    ylim1 = ax[1, 0].get_ylim()
    ylim2 = ax[1, 1].get_ylim()
    ymin = np.min([ylim1[0], ylim2[0]])
    ymax = np.max([ylim1[1], ylim2[1]])
    ax[1, 0].set_ylim(ymin, ymax)
    ax[1, 1].set_ylim(ymin, ymax)

    add_labels(ax[0, 1], '', 'Difference\n(ppb/day)')
    add_labels(ax[1, 1], 'State vector element', '')
    add_title(ax[0, 0], 'Base inversions')
    add_title(ax[0, 1], 'Difference (Inversion - Inversion 1)')
    add_title(ax[1, 0], 'Observations (Inversion 1)')
    add_title(ax[1, 1], 'Observations (Inversion 2)')
    format_plot(fig, ax, orig.nstate)
    return fig, ax
