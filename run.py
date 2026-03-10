import numpy as np
import os
from copy import deepcopy as dc
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Custom packages
from src.utilities import utils
from src.utilities import format_plots as fp
from src.model import inversion as inv

np.set_printoptions(precision=3, linewidth=300, suppress=True)

## -------------------------------------------------------------------------##
# File Locations
## -------------------------------------------------------------------------##
project_dir, config = utils.setup()
plot_dir = f'{project_dir}/plots'
if not os.path.exists(plot_dir):
    os.mkdir(plot_dir)

## -------------------------------------------------------------------------##
# Create initial inversion object and plot the results
## -------------------------------------------------------------------------##
orig = inv.Inversion()

fig, ax = fp.plot_summary(orig)
fp.save_fig(fig, plot_dir, f'summary')

## -------------------------------------------------------------------------##
# Now, test changing basic inversion parameters
## -------------------------------------------------------------------------##
# Try using 100% errors on the prior instead of 50% errors
changed_sa = inv.Inversion(sa=1)
fig, ax = fp.plot_difference(orig, changed_sa)
fp.save_fig(fig, plot_dir, 'changed_sa')

# Try using more or fewer observations
nobs_per_cell = int(orig.nobs_per_cell * 1.5)
changed_nobs = inv.Inversion(nobs_per_cell=nobs_per_cell)
fig, ax = fp.plot_difference(orig, changed_nobs)
fp.save_fig(fig, plot_dir, 'changed_nobs')

# Try using more or less accurate observations
obs_err = config['obs_err'] * 1.5
so = orig.so * 1.5
changed_obs_err = inv.Inversion(obs_err=obs_err, so=so)
fig, ax = fp.plot_difference(orig, changed_obs_err)
fp.save_fig(fig, plot_dir, 'changed_obs_err')

## -------------------------------------------------------------------------##
# Slightly more advanced changes!
## -------------------------------------------------------------------------##
# Try using slower or faster winds
U = orig.U * 0.5
changed_U = inv.Inversion(U=U)
fig, ax = fp.plot_difference(orig, changed_U)
fp.save_fig(fig, plot_dir, 'changed_U')

# Try using sloshing wind speeds
U = np.concatenate([np.arange(7, 3, -1), 
                    np.arange(3, 7, 1)])*24*60*60/1000
var_U = inv.Inversion(U=U)
fig, ax = fp.plot_difference(orig, var_U)
fp.save_fig(fig, plot_dir, 'var_U')

# Try using an incorrect boundary condition / background
BC = orig.BC + 10
changed_BC = inv.Inversion(BC=BC)
fig, ax = fp.plot_difference(orig, changed_BC)
fp.save_fig(fig, plot_dir, 'changed_BC')

# # Try changing the emissions (and the prior, so that it's not more or less 
# # biased than before
# xt_abs = orig.xt_abs[0] * 1.5
# rs = np.random.RandomState(config['random_state'])
# xa_abs = 1.5 * np.abs(rs.normal(loc=25, scale=5, size=(orig.nstate_model,)))
# changed_x = inv.Inversion(xt_abs=xt_abs, xa_abs=xa_abs)
# fig, ax = fp.plot_difference(orig, changed_x)
# fp.save_fig(fig, plot_dir, 'changed_x')