import numpy as np
import logging
import yaml
from jax import vmap
import pandas as pd
import matplotlib.pyplot as plt
import time
import jax.numpy as jnp
import os
import scs
import cvxpy as cp
import jax.scipy as jsp
from l2ws.algo_steps import create_M
from scipy.sparse import csc_matrix
from examples.solve_script import setup_script
from l2ws.launcher import Workspace


plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.size": 16,
    }
)
log = logging.getLogger(__name__)


def run(run_cfg):
    example = "sparse_pca"
    data_yaml_filename = 'data_setup_copied.yaml'

    # read the yaml file
    with open(data_yaml_filename, "r") as stream:
        try:
            setup_cfg = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            setup_cfg = {}

    # set the seed
    np.random.seed(setup_cfg['seed'])
    n_orig = setup_cfg['n_orig']
    k = setup_cfg['k']

    static_dict = static_canon(n_orig, k)

    # we directly save q now
    get_q = None
    static_flag = True
    workspace = Workspace(run_cfg, static_flag, static_dict, example, get_q)

    # run the workspace
    workspace.run()


def multiple_random_sparse_pca(n_orig, k, r, N, seed=42):
    out_dict = static_canon(n_orig, k)
    # c, b = out_dict['c'], out_dict['b']
    P_sparse, A_sparse = out_dict['P_sparse'], out_dict['A_sparse']
    cones = out_dict['cones_dict']
    prob, A_param = out_dict['prob'], out_dict['A_param']
    P, A = jnp.array(P_sparse.todense()), jnp.array(A_sparse.todense())

    # get theta_mat
    A_tensor, theta_mat = generate_A_tensor(N, n_orig, r)
    theta_mat_jax = jnp.array(theta_mat)

    # get theta_mat
    m, n = A.shape
    q_mat = get_q_mat(A_tensor, prob, A_param, m, n)

    return P, A, cones, q_mat, theta_mat_jax, A_tensor


def generate_A_tensor(N, n_orig, r):
    """
    generates covariance matrices A_1, ..., A_N
        where each A_i has shape (n_orig, n_orig)
    A_i = F Sigma_i F^T
        where F has shape (n_orig, r)
    i.e. each Sigma_i is psd (Sigma_i = B_i B_i^T) and is different
        B_i has shape (r, r)
        F stays the same for each problem
    We let theta = upper_tri(Sigma_i)
    """
    # first generate a random A matrix
    A0 = np.random.rand(n_orig, n_orig)

    # take the SVD
    U, S, VT = np.linalg.svd(A0)

    # take F to be the first r columns of U
    F = U[:, :r]
    A_tensor = np.zeros((N, n_orig, n_orig))
    r_choose_2 = int(r * (r + 1) / 2)
    theta_mat = np.zeros((N, r_choose_2))
    for i in range(N):
        B = 2 * np.random.rand(r, r) - 1
        Sigma = .1 * B @ B.T
        col_idx, row_idx = np.triu_indices(r)
        theta_mat[i, :] = Sigma[(row_idx, col_idx)]
        A_tensor[i, :, :] = F @ Sigma @ F.T
    return A_tensor, theta_mat


def cvxpy_prob(n_orig, k):
    A_param = cp.Parameter((n_orig, n_orig), symmetric=True)
    X = cp.Variable((n_orig, n_orig), symmetric=True)
    constraints = [X >> 0, cp.sum(cp.abs(X)) <= k, cp.trace(X) == 1]
    prob = cp.Problem(cp.Minimize(-cp.trace(A_param @ X)), constraints)
    return prob, A_param


def get_q_mat(A_tensor, prob, A_param, m, n):
    N, n_orig, _ = A_tensor.shape
    q_mat = jnp.zeros((N, m + n))
    for i in range(N):
        # set the parameter
        A_param.value = A_tensor[i, :, :]

        # get the problem data
        data, _, __ = prob.get_problem_data(cp.SCS)

        c, b = data['c'], data['b']
        n = c.size
        q_mat = q_mat.at[i, :n].set(c)
        q_mat = q_mat.at[i, n:].set(b)
    return q_mat


def static_canon(n_orig, k):
    # create the cvxpy problem
    prob, A_param = cvxpy_prob(n_orig, k)

    # get the problem data
    data, _, __ = prob.get_problem_data(cp.SCS)

    A_sparse, c, b = data['A'], data['c'], data['b']
    m, n = A_sparse.shape
    P_sparse = csc_matrix(np.zeros((n, n)))
    cones_cp = data['dims']

    # factor for DR splitting
    m, n = A_sparse.shape
    P_jax, A_jax = jnp.array(P_sparse.todense()), jnp.array(A_sparse.todense())
    M = create_M(P_jax, A_jax)
    algo_factor = jsp.linalg.lu_factor(M + jnp.eye(n + m))

    # set the dict
    cones = {'z': cones_cp.zero, 'l': cones_cp.nonneg, 'q': cones_cp.soc, 's': cones_cp.psd}
    out_dict = dict(
        M=M,
        algo_factor=algo_factor,
        cones_dict=cones,
        A_sparse=A_sparse,
        P_sparse=P_sparse,
        b=b,
        c=c,
        prob=prob,
        A_param=A_param
    )
    return out_dict


def setup_probs(setup_cfg):
    cfg = setup_cfg
    N_train, N_test = cfg.N_train, cfg.N_test
    N = N_train + N_test
    n_orig = cfg.n_orig

    np.random.seed(cfg.seed)

    # save output to output_filename
    output_filename = f"{os.getcwd()}/data_setup"

    P, A, cones, q_mat, theta_mat_jax, A_tensor = multiple_random_sparse_pca(
        n_orig, cfg.k, cfg.r, N)
    P_sparse, A_sparse = csc_matrix(P), csc_matrix(A)
    m, n = A.shape

    # create scs solver object
    #    we can cache the factorization if we do it like this
    b_np, c_np = np.array(q_mat[0, n:]), np.array(q_mat[0, :n])
    data = dict(P=P_sparse, A=A_sparse, b=b_np, c=c_np)
    tol_abs = cfg.solve_acc_abs
    tol_rel = cfg.solve_acc_rel
    solver = scs.SCS(data, cones, eps_abs=tol_abs, eps_rel=tol_rel)

    setup_script(q_mat, theta_mat_jax, solver, data, cones, output_filename, solve=cfg.solve)
