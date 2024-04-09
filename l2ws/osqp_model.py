from functools import partial

import jax.numpy as jnp
import numpy as np
import osqp
from scipy.sparse import csc_matrix

from l2ws.algo_steps import k_steps_eval_osqp, k_steps_train_osqp, unvec_symm
from l2ws.l2ws_model import L2WSmodel


class OSQPmodel(L2WSmodel):
    def __init__(self, **kwargs):
        super(OSQPmodel, self).__init__(**kwargs)

    def initialize_algo(self, input_dict):
        # self.m, self.n = self.A.shape
        self.algo = 'osqp'
        self.m, self.n = input_dict['m'], input_dict['n']
        self.q_mat_train, self.q_mat_test = input_dict['q_mat_train'], input_dict['q_mat_test']

        self.rho = input_dict['rho']
        self.sigma = input_dict.get('sigma', 1)
        self.alpha = input_dict.get('alpha', 1)
        self.output_size = self.n + self.m

        # custom_loss
        custom_loss = input_dict.get('custom_loss')

        """
        break into the 2 cases
        1. factors are the same for each problem (i.e. matrices A and P don't change)
        2. factors change for each problem
        """
        self.factors_required = True
        self.factor_static_bool = input_dict.get('factor_static_bool', True)
        if self.factor_static_bool:
            self.A = input_dict['A']
            # self.P = input_dict.get('P', None)
            self.P = input_dict['P']
            self.factor_static = input_dict['factor']
            self.k_steps_train_fn = partial(
                k_steps_train_osqp, A=self.A, rho=self.rho, sigma=self.sigma, jit=self.jit)
            self.k_steps_eval_fn = partial(k_steps_eval_osqp, P=self.P,
                                           A=self.A, rho=self.rho, sigma=self.sigma, jit=self.jit, 
                                           custom_loss=custom_loss)
        else:
            self.k_steps_train_fn = self.create_k_steps_train_fn_dynamic()
            self.k_steps_eval_fn = self.create_k_steps_eval_fn_dynamic()
            # self.k_steps_eval_fn = partial(k_steps_eval_osqp, rho=rho, sigma=sigma, jit=self.jit)

            self.factors_train = input_dict['factors_train']
            self.factors_test = input_dict['factors_test']

            

        # self.k_steps_train_fn = partial(k_steps_train_osqp, factor=factor, A=self.A, rho=rho, 
        #                                 sigma=sigma, jit=self.jit)
        # self.k_steps_eval_fn = partial(k_steps_eval_osqp, factor=factor, P=self.P, A=self.A, 
        #                                rho=rho, sigma=sigma, jit=self.jit)
        self.out_axes_length = 6

    def create_k_steps_train_fn_dynamic(self):
        """
        creates the self.k_steps_train_fn function for the dynamic case
        acts as a wrapper around the k_steps_train_osqp functino from algo_steps.py

        we want to maintain the argument inputs as (k, z0, q_bar, factor, supervised, z_star)
        """
        m, n = self.m, self.n

        def k_steps_train_osqp_dynamic(k, z0, q, factor, supervised, z_star):
            nc2 = int(n * (n + 1) / 2)
            q_bar = q[:2 * m + n]
            unvec_symm(q[2 * m + n: 2 * m + n + nc2], n)
            A = jnp.reshape(q[2 * m + n + nc2:], (m, n))
            return k_steps_train_osqp(k=k, z0=z0, q=q_bar,
                                      factor=factor, A=A, rho=self.rho, sigma=self.sigma,
                                      supervised=supervised, z_star=z_star, jit=self.jit)
        return k_steps_train_osqp_dynamic

    def create_k_steps_eval_fn_dynamic(self):
        """
        creates the self.k_steps_train_fn function for the dynamic case
        acts as a wrapper around the k_steps_train_osqp functino from algo_steps.py

        we want to maintain the argument inputs as (k, z0, q_bar, factor, supervised, z_star)
        """
        m, n = self.m, self.n

        def k_steps_eval_osqp_dynamic(k, z0, q, factor, supervised, z_star):
            nc2 = int(n * (n + 1) / 2)
            q_bar = q[:2 * m + n]
            P = unvec_symm(q[2 * m + n: 2 * m + n + nc2], n)
            A = jnp.reshape(q[2 * m + n + nc2:], (m, n))
            return k_steps_eval_osqp(k=k, z0=z0, q=q_bar,
                                     factor=factor, P=P, A=A, rho=self.rho, sigma=self.sigma,
                                     supervised=supervised, z_star=z_star, jit=self.jit)
        return k_steps_eval_osqp_dynamic

    def solve_c(self, z0_mat, q_mat, rel_tol, abs_tol, max_iter=40000):
        # assume M doesn't change across problems
        # static problem data
        m, n = self.m, self.n
        nc2 = int(n * (n + 1) / 2)

        if self.factor_static_bool:
            P, A = self.P, self.A
        else:
            P, A = np.ones((n, n)), np.zeros((m, n))
        P_sparse, A_sparse = csc_matrix(np.array(P)), csc_matrix(np.array(A))


        osqp_solver = osqp.OSQP()
        

        # q = q_mat[0, :]
        c, l, u = np.zeros(n), np.zeros(m), np.zeros(m)  # noqa
        
        rho = 1
        osqp_solver.setup(P=P_sparse, q=c, A=A_sparse, l=l, u=u, alpha=self.alpha, rho=rho, 
                          sigma=self.sigma, polish=False,
                          adaptive_rho=False, scaling=0, max_iter=max_iter, verbose=True, 
                          eps_abs=abs_tol, eps_rel=rel_tol)

        num = z0_mat.shape[0]
        solve_times = np.zeros(num)
        solve_iters = np.zeros(num)
        x_sols = jnp.zeros((num, n))
        y_sols = jnp.zeros((num, m))
        for i in range(num):
            if not self.factor_static_bool:
                P = unvec_symm(q_mat[i, 2 * m + n: 2 * m + n + nc2], n)
                A = jnp.reshape(q_mat[i, 2 * m + n + nc2:], (m, n))
                c, l, u = np.array(q_mat[i, :n]), np.array(q_mat[i, n:n + m]),  np.array(q_mat[i, n + m:n + 2 * m])  # noqa
                
                P_sparse, A_sparse = csc_matrix(np.array(P)), csc_matrix(np.array(A))
                # Px = sparse.triu(P_sparse).data
                # import pdb
                # pdb.set_trace()
                osqp_solver = osqp.OSQP()
                osqp_solver.setup(P=P_sparse, q=c, A=A_sparse, l=l, u=u, alpha=self.alpha, rho=rho, 
                                  sigma=self.sigma, polish=False,
                                  adaptive_rho=False, scaling=0, max_iter=max_iter, verbose=True, 
                                  eps_abs=abs_tol, eps_rel=rel_tol)
                # osqp_solver.update(Px=P_sparse, Ax=csc_matrix(np.array(A)))
            else:
                # set c, l, u
                c, l, u = q_mat[i, :n], q_mat[i, n:n + m], q_mat[i, n + m:n + 2 * m]  # noqa
                osqp_solver.update(q=np.array(c))
                osqp_solver.update(l=np.array(l), u=np.array(u))

            

            # set the warm start
            # x, y, s = self.get_xys_from_z(z0_mat[i, :])
            x_ws, y_ws = np.array(z0_mat[i, :n]), np.array(z0_mat[i, n:n + m])

            # fix warm start
            osqp_solver.warm_start(x=x_ws, y=y_ws)

            # solve
            results = osqp_solver.solve()
            # sol = solver.solve(warm_start=True, x=np.array(x), y=np.array(y), s=np.array(s))

            # set the solve time in seconds
            solve_times[i] = results.info.solve_time * 1000
            solve_iters[i] = results.info.iter

            # set the results
            x_sols = x_sols.at[i, :].set(results.x)
            y_sols = y_sols.at[i, :].set(results.y)

        return solve_times, solve_iters, x_sols, y_sols
