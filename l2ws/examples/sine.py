import jax.numpy as jnp
import numpy as np
import cvxpy as cp
import yaml
from l2ws.launcher import Workspace
from l2ws.examples.solve_script import ista_setup_script
import os
from scipy.sparse import random
from jax import vmap
import matplotlib.pyplot as plt

def run(run_cfg):
    example = "sine"
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
    # m_orig, n_orig = setup_cfg['m_orig'], setup_cfg['n_orig']


    # # get D
    # D = np.random.normal(size=(m_orig, n_orig)) / np.sqrt(m_orig)
    # D = D / np.linalg.norm(D, axis=0)
    # D = np.array(D)

    

    # # get the ista values
    # evals, evecs = jnp.linalg.eigh(D.T @ D)
    # step = 1 / evals.max()
    # lambd = .1
    # eta = lambd * step



    # static_dict = dict(D=D, W=W, step=step, eta=eta)
    static_dict = dict()

    # we directly save q now
    static_flag = True
    # algo = 'alista'
    algo = run_cfg.get('algo', 'maml')
    workspace = Workspace(algo, run_cfg, static_flag, static_dict, example)

    # run the workspace
    workspace.run()


def setup_probs(setup_cfg):
    cfg = setup_cfg
    N_train, N_test = cfg.N_train, cfg.N_test
    N = N_train + N_test
    K = cfg.K
    K_total = cfg.K_total


    np.random.seed(cfg.seed)


    # sample amplitudes and phases
    amplitudes = jnp.array(0.1 + 4.9 * np.random.rand(N, 1))
    phases = jnp.array(np.random.rand(N, 1) * np.pi)

    # batching
    batch_sine = vmap(sine, in_axes=(0, 0, 0), out_axes=(0))

    # sample random theta = (x_i, y_i) i=1,...,K -- for K-shot learning
    x_k_shot = jnp.array(10 * (np.random.rand(N, K) - .5))
    y_k_shot = batch_sine(amplitudes, phases, x_k_shot)
    theta = jnp.hstack([x_k_shot, y_k_shot])

    # sample random z_star = (x_i, y_i) i=1,...,K_meta -- for meta-loss
    x_total = jnp.array(10 * (np.random.rand(N, K_total) - .5))
    y_total = batch_sine(amplitudes, phases, x_total)
    z_stars = jnp.hstack([x_total, y_total])

    # save output to output_filename
    output_filename = f"{os.getcwd()}/data_setup"

    # ista_setup_script(b_mat, A, lambd, output_filename)
    jnp.savez(
        output_filename,
        thetas=jnp.array(theta),
        z_stars=jnp.array(z_stars),
    )

def sine(amplitude, phase, x_points):
    y_points = amplitude * jnp.sin(x_points - phase)
    return y_points