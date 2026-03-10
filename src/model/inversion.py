import numpy as np
import math
from copy import deepcopy as dc
from src.utilities import utils
project_dir, config = utils.setup()

class OSSE:
    def __init__(self, nstate_model, nobs_per_cell, 
                 Cmax, L, U, init_t, total_t,
                 BCt, xt_abs, obs_err, rs):
        '''
        Define an inversion object for an OSSE:
            nstate              ...
            nobs_per_cell       ...
            C 
            L
            U
            init_t
            total_t
            BCt
            xt_abs             True emissions in ppb/day. We assume 
                                that the true emissions are constant across
                                the domain, so this is a scalar. Default 
                                value of 100 ppb/day.
            sa                  Relative errors for the inversion. We assume
                                that errors are constant across the domain. 
                                Default value of 0.5.
            opt_BC              A Boolean corresponding to whether the 
                                inversion optimizes the boundary condition 
                                (BC) or not. Default False.
        '''
        # Define dimensions of the state and observation vectors
        self.nstate_model = nstate_model
        self.nobs_per_cell = nobs_per_cell
        self.nobs_model = self.nstate_model * self.nobs_per_cell

        # Define the grid cell length and wind speed
        self.L = L
        if type(U) != np.ndarray:
            U = np.array([U])
        
        # Define time step and times to sample at
        self.delta_t = np.min(Cmax * self.L / np.abs(U))
        self.t = np.arange(0, init_t + total_t + self.delta_t, self.delta_t)
        self.obs_t = np.linspace(init_t + self.delta_t, init_t + total_t, 
                                 nobs_per_cell)

        # Rescale U to be the length of the time array.
        repeat_factor = math.ceil(len(self.t)/len(U))
        self.U = np.tile(U, repeat_factor)[:len(self.t)]

        # Calculate the Courant number
        self.C = self.U*self.delta_t/self.L

        # Get rate constants
        self.j = (self.U).mean()/self.L
        self.tau = 1/self.j

        # True quantities (ppb)
        self.BCt = BCt
        self.xt_abs = xt_abs*np.ones(self.nstate_model)

        # Initial conditions given by steady state with the true BC
        # (We don't need to use the forward model for these because 
        # every model simulation has a spin-up that will reach steady
        # state.)
        self.y0 = self.BCt + np.cumsum(self.xt_abs/self.j)

        # Pseudo observations
        y_err = np.random.RandomState(rs).normal(
            0, obs_err, (self.nstate_model, self.nobs_per_cell))
        self.y = (self.forward_model(x=self.xt_abs, 
                                     BC=self.BCt*np.ones(len(self.t)))
                  + y_err).T.flatten()

class ForwardModel(OSSE):
    def forward_model(self, x, BC):
        '''
        A function that calculates the mass in each reservoir
        after a given time given the following:
            x         :    vector of emissions (ppb/s)
            y0        :    initial atmospheric condition
            BC        :    boundary condition
            ts        :    times at which to sample the model
            U         :    wind speed
            L         :    length scale for each grid box
            obs_t     :    times at which to sample the model
        '''
        # # Get times
        # if times is None:
        #     times = self.t[self.t >= self.obs_t.min()]
        # if recalc_y0:
        #     y0 = self.BCt + np.cumsum(self.xt_abs[-len(x):]/self.j)
        # else:
        #     y0 = self.y0

        # Create an empty array (grid box x time) for all
        # model output
        ys = np.zeros((len(x), len(self.t)))
        ys[:, 0] = self.y0

        # Iterate through the time steps
        for i, t in enumerate(self.t[1:]):
            # Do advection and emissions using the boundary condition
            # from the previous time step (since Lax Wendroff relies
            # on the concentrations across the entire domain, including
            # the boundary condition, at the previous time step) and 
            # the Courant number from the current time step. 
            # TO DO: there is some confusion about which Courant number
            # I should use.
            ynew = self.do_advection(ys[:, i], BC[i], self.C[i + 1])
            ys[:, i+1] = self.do_emissions(ynew, x)

        # Subset all output for observational times
        t_idx = utils.nearest_loc(self.obs_t, self.t)
        ys = ys[:, t_idx]

        return ys

    def do_emissions(self, y_prev, x):
        y_new = y_prev + x*self.delta_t
        return y_new

    def do_advection(self, y_prev, BC, C):
        '''
        Advection following the Lax-Wendroff scheme
        '''
        # Append the boundary conditions
        y_prev = np.append(BC, y_prev)

        # Calculate the next time step using Lax-Wendroff
        y_new = (y_prev[1:-1]
                 - C*(y_prev[2:] - y_prev[:-2])/2
                 + C**2*(y_prev[2:] - 2*y_prev[1:-1] + y_prev[:-2])/2)

        # Update the last grid cell using upstream
        y_new = np.append(y_new, y_prev[-1] - C*(y_prev[-1] - y_prev[-2]))

        return y_new

class Inversion(ForwardModel):
    def __init__(
            self,
            nstate_model=config['nstate_model'],
            nstate=config['nstate'], 
            nobs_per_cell=config['nobs_per_cell'], 
            Cmax=config['Cmax'], 
            L=config['L'], 
            U=config['U'], 
            init_t=config['init_t'],
            total_t=config['total_t'],
            BCt=config['BCt'], 
            xt_abs=config['xt_abs'], 
            obs_err=config['obs_err'],
            xa_abs=None, 
            sa=config['sa'], 
            sa_BC=config['sa_BC'], 
            so=config['so'], 
            gamma=1, 
            k=None, 
            BC=None,
            opt_BC=False, 
            opt_BC_n=1, 
            buffer=False,
            buffer_size=1,
            sequential=False, 
            rs=config['random_state']):
        # Inherit from the parent class
        OSSE.__init__(self, nstate_model, nobs_per_cell, 
                      Cmax, L, U, init_t, total_t, 
                      BCt, xt_abs, obs_err, rs)

        # Set inversion options
        ## Buffer
        self.buffer = buffer
        if not self.buffer:
            buffer_size = 0
        self.buffer_size = buffer_size
        ## Boundary
        self.opt_BC = opt_BC
        self.opt_BC_n = opt_BC_n
        ## Sequential (i.e., optimize BC only, then fluxes only)
        self.sequential = sequential

        # Set state dimension of the inversion. This is a subset of the 
        # dimension of the model.
        self.nstate = nstate
        self.state_start = self.nstate_model - self.nstate - self.buffer_size
        self.nobs = self.nstate * self.nobs_per_cell
        self.obs_start = self.state_start * self.nobs_per_cell
        if self.buffer:
            self.nstate += 1
            self.nobs += self.buffer_size * self.nobs_per_cell
        if self.opt_BC:
            self.nstate += self.opt_BC_n

        # Define an inversion plotting vector (we exclude the buffer element)
        self.xp = np.arange(1, nstate + 1)

        # Prior (we set this to the full model dimension and will subset it 
        # to nstate later)
        if xa_abs is None:
            self.xa_abs = np.abs(
                np.random.RandomState(rs).normal(
                    loc=25, scale=5, size=(self.nstate_model,)))
        else:
            self.xa_abs = xa_abs

        # We no longer need this because we are now using the same nstate_model
        # for all inversions.
        # if self.buffer:
        #     self.xa_abs = np.append(self.xa_abs[-self.buffer_size:], 
        #                             self.xa_abs[:-self.buffer_size])

        # Relative prior
        self.xa = np.ones(self.nstate)

        # Prior errors (ppb/day)
        if type(sa) in [float, int]:
            self.sa = (sa**2)*np.ones(self.nstate)
        else:
            self.sa = sa

        # Observational errors (ppb)
        if type(so) in [float, int]:
            self.so = (so**2)*np.ones(self.nobs)
        else:
            self.so = so
        self.gamma = gamma

        # Define the inversion boundary condition (of length of self.t)
        if BC is None:
            BC = BCt
        self.BC = BC*np.ones(len(self.t))

        # Boolean for whether we optimize the BC, and expand the previously
        # defined quantities to include BC elements
        if self.opt_BC:
            self.opt_BC_n = opt_BC_n
            self.xa[-self.opt_BC_n:] = 0
            self.sa[-self.opt_BC_n:] = sa_BC**2*np.ones(self.opt_BC_n)

        # Prior model simulation. We simulate over all of the model grid cells,
        # and then subset only over the grid cells that are used in the inv.
        self.ya = self.forward_model(x=self.xa_abs, BC=self.BC).T
        obs_shape = self.ya.shape
        self.ya = self.ya.flatten()

        # Build the Jacobian over the full dimension (not technically necessary,
        # but since it's cheap, it's easier). We then subset it over the 
        # inversion domain.
        if k is None:
            self.build_jacobian()
            self.k = self.k.reshape(
                (obs_shape[0], obs_shape[1], -1)
            )
            self.k = self.k[:, self.state_start:, self.state_start:]
            self.k = self.k.reshape((self.nobs, -1))

        # We do the same with the modeled and observed concentrations
        # ya and y (respectively).
        self.y = self.y.reshape(obs_shape)[:, self.state_start:].flatten()
        self.ya = self.ya.reshape(obs_shape)[:, self.state_start:]
        self.ya = self.ya.flatten()

        # Now, we aggregate the Jacobian and prior if we are using buffers. 
        # Also, estimate the p scaling factor and apply it to the prior error
        # covariance
        full_domain_start = self.state_start + self.buffer_size
        if self.buffer:
            self.xa_abs = np.append(
                self.xa_abs[self.state_start:full_domain_start].sum(), 
                self.xa_abs[full_domain_start:])
            k_buffer = self.k[:, :self.buffer_size].sum(axis=1)
            k_rest = self.k[:, self.buffer_size:]
            self.k = np.append(k_buffer.reshape(self.nobs, -1), k_rest, axis=1)
            self.p = self.estimate_p(sa_BC)
            self.sa[0] *= self.p**2
        else:
            self.xa_abs = self.xa_abs[full_domain_start:]
            self.xt_abs = self.xt_abs[full_domain_start:]

        # Calculate c
        self.c = self.ya - self.k @ self.xa

        # Solve the inversion and get the influence length scale
        self.solve_inversion()
        self.calculate_BC_bias_metrics()
        self.remove_BC_elements()
        self.remove_buffer_elements()


    def build_jacobian(self):
        F = lambda x : self.forward_model(x=x, BC=self.BC).T.flatten()

        # Initialize the Jacobian
        k = np.zeros((self.nobs_model, self.nstate_model))

        # Iterate through the state vector elements
        for i in range(self.nstate_model):
            # Apply the perturbation to the ith state vector element
            x = dc(self.xa_abs)
            x[i] *= 1.5

            # Run the forward model
            ypert = F(x)

            # Save out the result
            k[:, i] = (ypert - self.ya)/0.5

        if self.opt_BC:
            # Add a column for the optimization of the boundary condition
            if (self.opt_BC_n > 1):
                BC_chunks = math.ceil(len(self.BC)/self.opt_BC_n)
                ypert = []
                i = 0
                while i < len(self.BC):
                    BCpert = self.BC.copy()
                    BCpert[i:(i + BC_chunks)] += 10
                    ypert.append(
                        self.forward_model(
                            x=self.xa_abs, 
                            BC=BCpert).T.flatten().reshape(-1, 1))
                    i += BC_chunks
                ypert = np.concatenate(ypert, axis=1)
            else:
                ypert = self.forward_model(
                    x=self.xa_abs, BC=self.BC + 10).T.flatten().reshape(-1, 1)
            k = np.append(k, (ypert - self.ya.reshape(-1, 1))/10, axis=1)
        self.k = k

    def solve_inversion(self):
        # Adjust the magnitude of So using a gamma (for similarity
        # to a real inversion)
        if self.gamma is None:
            self.get_gamma()
        else:
            self.so = self.so/self.gamma
            self._solve_inversion()

    def _solve_inversion(self):
        # Solve the inversion
        if self.sequential: 
            # Solve for BC only
            self._solve_inversion_equations(
                self.y, self.k[:, -self.opt_BC_n:], self.xa[-self.opt_BC_n:],
                self.c, self.sa[-self.opt_BC_n:], self.so)
            print('Sequential BC optimized : ', self.xhat)
            
            # Update parameters with output of first inversion
            self.BC = self.xhat*np.ones(len(self.BC))
            self.xa[-self.opt_BC_n:] = self.xhat
            self.c = self.xhat*np.ones(self.nobs)

            # Solve the inversion with the new paramaters
            self._solve_inversion_equations(
                self.y, self.k[:, :-self.opt_BC_n], self.xa[:-self.opt_BC_n],
                self.c, self.sa[:-self.opt_BC_n], self.so)
            
            # Set self.opt_BC to False because now xhat is dimension nstate
            self.opt_BC = False

        else:
            self._solve_inversion_equations(
                self.y, self.k, self.xa, self.c, self.sa, self.so)
    
    def _solve_inversion_equations(self, y, k, xa, c, sa, so):
        # Get the inverse of sa and so
        if len(sa.shape) == 1:
            sa_inv = np.diag(1/sa)
        else:
            sa_inv = np.linalg.inv(sa)
        
        if len(so.shape) == 1:
            so_inv = np.diag(1/so)
        else:
            so_inv = np.linalg.inv(so)

        # Solve for the inversion
        self.shat = np.linalg.inv(sa_inv + k.T @ so_inv @ k)
        self.g = self.shat @ k.T @ so_inv
        self.a = np.identity(len(xa)) - self.shat @ sa_inv
        self.xhat = (xa + self.g @ (y - k @ xa - c))
        self.yhat = k @ self.xhat + c


    def get_gamma(self, tol=1e-1):
        print('Finding gamma...')
        gamma = 10
        gamma_not_found = True
        so_orig = dc(self.so)
        while gamma_not_found:
            self.so = so_orig/gamma
            self._solve_inversion()
            cost = self.cost_prior()/self.nstate
            print(f'{gamma:.4f}: {cost:.3f}')
            if np.abs(cost - 1) <= tol:
                gamma_not_found = False
            elif cost > 1:
                gamma /= 2
            elif cost < 1:
                gamma *= 1.5
        self.gamma = gamma
        print('Gamma found! Adjusting So.')
        print('-'*70)


    def cost_prior(self):
        return (((self.xhat - self.xa)**2)/self.sa).sum()


    def calculate_BC_bias_metrics(self):
        # First, calculate the contributions from the boundary condition
        # and from the emissions to the model corrrection term, accounting
        # for differences resulting from whether or not the boundary
        # condition is optimized by the inversion.
        if self.opt_BC:
            opt_BC_n = int(self.opt_BC_n)
            self.bc_contrib = self.a[:, -opt_BC_n:] @ self.xa[-opt_BC_n:]
            self.xa_contrib = self.a[:, :-opt_BC_n] @ self.xa[:-opt_BC_n]

        else:
            self.bc_contrib = self.g @ self.c
            self.xa_contrib = self.a @ self.xa[:self.nstate]


    def remove_BC_elements(self):
        if self.opt_BC:
            opt_BC_n = int(self.opt_BC_n)

            self.xhat_BC = self.xhat[-opt_BC_n:]
            self.xhat = self.xhat[:-opt_BC_n]

            self.shat_full = self.shat
            self.shat = self.shat[:-opt_BC_n, :-opt_BC_n]

            self.a_full = self.a
            self.a_BC = self.a[-opt_BC_n:, -opt_BC_n:]
            self.a = self.a[:-opt_BC_n, :-opt_BC_n]

            self.g_BC = self.g[-opt_BC_n:, :]
            self.g = self.g[:-opt_BC_n, :]
            
            self.bc_contrib = self.bc_contrib[:-opt_BC_n]
            self.xa_contrib = self.xa_contrib[:-opt_BC_n]

    def remove_buffer_elements(self):
        if self.buffer:
            self.xhat_buffer = self.xhat[0]
            self.xhat = self.xhat[1:]

            self.shat_full = self.shat
            self.shat = self.shat[1:, 1:]

            self.sa_full = self.sa
            self.sa = self.sa[1:]

            self.a_full = self.a
            self.a_BC = self.a[0, 0]
            self.a = self.a[1:, 1:]

            self.g_BC = self.g[0, :]
            self.g = self.g[1:, :]

    def estimate_D(self, sa_bc, R):
        delta_xhat = np.abs(self.estimate_delta_xhat(sa_bc))
        return np.where(delta_xhat < R*delta_xhat.max())

    def preview_2d(self, sa_bc):
        # Transport
        D = np.arange(self.L, self.L*self.nstate + self.L, self.L)
        j = D/self.L
        U = self.U.max()
        k_i = self.L/U
        k_up = (D/U)[:-1]

        # Prior and prior uncertainty
        sa_i = (self.xa_abs**2*self.sa)
        # sa_up = (np.cumsum(self.xa_abs)**2*self.sa)[:-1]
        sa_up = np.cumsum(self.xa_abs**2 * self.sa)[:-1] # Added in quadrature

        # Observing system errors 
        so_i = ((self.so.mean()/self.nobs_per_cell)**0.5)**2 # individual 
        so_up = (so_i/j)[:-1]
        so_up[so_up < 16] = 16

        # Obs counts
        m_i = self.nobs_per_cell
        m_up = j[:-1] * self.nobs_per_cell

        # Secondary quantitites
        # R_i = so_i/(k_i**2*sa_i)
        # R_up = np.append(0, so_up/(k_up**2*sa_up))
        R_i = k_i**2 * sa_i / so_i # Exclude m_i bc it's built into so_i
        R_up = np.append(0, k_up**2 * sa_up / so_up) # Exclude m_up bc i'ts built into so_up
        # beta = j[1:]**2*sa_up/sa_i + 1
        alpha = np.append(0, R_i[:-1] + (m_i / m_up) + 1)
        
        # # Predict error
        # delta_xhat_0 = (sa_bc/k_i)/(1 + R_i[0])
        # delta_xhat = (sa_bc/k_i)*R_up/(R_up*R_i + R_i + beta*R_up + 1)
        delta_xhat = (R_i / (alpha * R_up + R_i + 1)) * (sa_bc / k_i)

        # While we're at it, we'll estimate an influence length scale using
        # the 1D approximation
        so_mean = (np.mean(so_i**0.5)**2)
        sa = np.mean(self.sa**0.5)**2*np.mean(self.xa_abs)**2

        R = so_mean/(np.mean(k_i)**2*sa)
        print('  R : ', 1/R)

        return - delta_xhat / self.xa_abs, R


    def estimate_p(self, sa_bc):
        try: 
            Umin = self.U.min()
            Umax = self.U.max()
        except:
            Umin = self.U
            Umax = self.U
        tau_min = self.L/Umax
        tau_max = self.L/Umin
        if self.opt_BC:
            sa = self.sa[:-self.opt_BC_n]
        else:
            sa = self.sa
        sa_min = (sa**0.5*self.xa_abs).min()**2
        sa_max = (sa**0.5*self.xa_abs).max()**2
        R_min = tau_min**2*sa_min/self.so.max()*self.nobs_per_cell
        R_max = tau_max**2*sa_max/self.so.max()*self.nobs_per_cell
        R_sqrt_min = np.minimum(np.sqrt(R_min + 2), np.sqrt(R_max + 2))
        R_sqrt_max = np.maximum(np.sqrt(R_min + 2), np.sqrt(R_max + 2))
        p_min = sa_bc/(tau_max*sa_max*R_sqrt_max)
        p_max = sa_bc/(tau_min*sa_min*R_sqrt_min)
        print('  Buffer scale factor : ', p_min, p_max)
        return p_max*10