from functools import partial

import cvxpy as cp
import jax.numpy as jnp
import jax.scipy as jsp
from cvxpylayers.jax import CvxpyLayer
from jax import grad, jit, lax, vmap
from jax.tree_util import tree_map

from l2ws.utils.generic_utils import python_fori_loop, unvec_symm, vec_symm

TAU_FACTOR = 1 #10


# def fixed_point_extragrad(z, Q, R, A, c, b, eg_step):
#     """
#     applies the extragradient fixed point operator
#     """
#     m, n = A.shape
#     x0 = z[:n]
#     y0 = z[n:]
#     x1 = x0 - eg_step * (2 * Q @ x0 + A.T @ y0 + c)
#     y1 = y0 + eg_step * (-2 * R @ y0 + A @ x0 - b)
#     x2 = x0 - eg_step * (2 * Q @ x1 + A.T @ y1 + c)
#     y2 = y0 + eg_step * (-2 * R @ y1 + A @ x1 - b)
#     return jnp.concatenate([x2, y2])

def k_steps_train_maml(k, z0, q, neural_net_fwd, neural_net_grad, gamma, supervised, z_star, jit):
    iter_losses = jnp.zeros(k)

    fp_train_partial = partial(fp_train_maml,
                               supervised=supervised,
                               z_star=z_star,
                               neural_net_fwd=neural_net_fwd,
                               neural_net_grad=neural_net_grad,
                               theta=q,
                               gamma=gamma
                               )
    val = z0, iter_losses
    start_iter = 0
    if jit:
        out = lax.fori_loop(start_iter, k, fp_train_partial, val)
    else:
        out = python_fori_loop(start_iter, k, fp_train_partial, val)
    z_final, iter_losses = out
    return z_final, iter_losses


def k_steps_eval_maml(k, z0, q, neural_net_fwd, neural_net_grad, gamma, supervised, z_star, jit):
    iter_losses, obj_diffs = jnp.zeros(k), jnp.zeros(k)
    # z_all_plus_1 = jnp.zeros((k + 1, z0.size))
    # z_all_plus_1 = z_all_plus_1.at[0, :].set(z0)
    fp_eval_partial = partial(fp_eval_maml,
                              supervised=supervised,
                               z_star=z_star,
                               neural_net_fwd=neural_net_fwd,
                               neural_net_grad=neural_net_grad,
                               theta=q,
                               gamma=gamma
                              )
    z_all = jnp.zeros((k, int(z_star.size / 2)))
    val = z0, iter_losses, z_all #, z_all, obj_diffs
    start_iter = 0
    if jit:
        out = lax.fori_loop(start_iter, k, fp_eval_partial, val)
    else:
        out = python_fori_loop(start_iter, k, fp_eval_partial, val)
    z_final, iter_losses, z_all = out #, z_all, obj_diffs = out
    # z_all_plus_1 = z_all_plus_1.at[1:, :].set(z_all)
    return z_final, iter_losses, z_all, jnp.zeros(k)


def fixed_point_maml(z, neural_net_grad, theta, gamma):
    """
    theta is the problem data
    """
    # return soft_threshold(z - gamma * W.T.dot(D.dot(z) - b), theta)

    # gain
    # gain = gain_gate(z, theta, mu, nu)
    # z_gain = jnp.multiply(gain, z)
    # z_tilde = fixed_point_alista(z_gain, W, D, b, gamma, theta)

    # # overshoot
    # overshoot = overshoot_gate(z, z_tilde, a)
    # z_next = jnp.multiply(overshoot, z_tilde) + jnp.multiply((1 - overshoot), z)
    # return z_next
    # curr_loss = neural_net_fwd(z, theta)
    gradient, aux_data = neural_net_grad(z, theta)
    # for i in range(len(z)):

    # import pdb
    # pdb.set_trace()
    #     z_next = z - gamma * gradient
    # z_next = z
    # z_next = [tuple(z_elem - gamma * grad_elem for z_elem, grad_elem in zip(z_tuple, grad_tuple))
    #                 for z_tuple, grad_tuple in zip(z, gradient)
    #             ]
    inner_sgd_fn = lambda g, state: (state - gamma*g)
    z_next = tree_map(inner_sgd_fn, gradient, z)
    return z_next


def fp_train_maml(i, val, supervised, z_star, neural_net_fwd, neural_net_grad, theta, gamma):
    z, loss_vec = val
    # gamma = params[i, 0] #jnp.exp(params[i, 0])
    # theta = params[i, 1] #jnp.exp(params[i, 1])
    # mu = params[i, 2]
    # nu = params[i, 3]
    # a = params[i, 4]
    z_next = fixed_point_maml(z, neural_net_grad, theta, gamma)
    # diff = jnp.linalg.norm(z - z_star) ** 2

    # z_star is all of the points
    # loss = neural_net_fwd(z_next, z_star)
    loss, aux = neural_net_fwd(z, z_star)

    loss_vec = loss_vec.at[i].set(loss)
    return z_next, loss_vec


def fp_eval_maml(i, val, supervised, z_star, neural_net_fwd, neural_net_grad, theta, gamma):
    # z, loss_vec, z_all, obj_diffs = val
    z, loss_vec, z_all = val

    z_next = fixed_point_maml(z, neural_net_grad, theta, gamma)

    # z_star is all of the points
    loss, aux = neural_net_fwd(z, z_star)
    predicted_outputs = aux[0]

    loss_vec = loss_vec.at[i].set(loss)

    z_all = z_all.at[i, :].set(predicted_outputs[:,0])
    # obj_diffs = obj_diffs.at[i].set(0)
    return z_next, loss_vec, z_all #, obj_diffs

    # z_next = fixed_point_glista(z, W, D, b, gamma, theta, mu, nu, a)
    # diff = 10 * jnp.log10(jnp.linalg.norm(z - z_star) ** 2 / jnp.linalg.norm(z_star) ** 2)
    # loss_vec = loss_vec.at[i].set(diff)
    # obj = .5 * jnp.linalg.norm(D @ z_next - b) ** 2 # + lambd * jnp.linalg.norm(z_next, ord=1)
    # opt_obj = .5 * jnp.linalg.norm(D @ z_star - b) ** 2 # + lambd * jnp.linalg.norm(z_star, ord=1)
    # obj_diffs = obj_diffs.at[i].set(obj - opt_obj)
    # z_all = z_all.at[i, :].set(z_next)
    # return z_next, loss_vec, z_all, obj_diffs


def k_steps_train_glista(k, z0, q, params, W, D, supervised, z_star, jit):
    iter_losses = jnp.zeros(k)

    fp_train_partial = partial(fp_train_glista,
                               supervised=supervised,
                               z_star=z_star,
                               W=W,
                               D=D,
                               b=q,
                               params=params
                               )
    val = z0, iter_losses
    start_iter = 0
    if jit:
        out = lax.fori_loop(start_iter, k, fp_train_partial, val)
    else:
        out = python_fori_loop(start_iter, k, fp_train_partial, val)
    z_final, iter_losses = out
    return z_final, iter_losses


def k_steps_eval_glista(k, z0, q, params, W, D, supervised, z_star, jit):
    iter_losses, obj_diffs = jnp.zeros(k), jnp.zeros(k)
    z_all_plus_1 = jnp.zeros((k + 1, z0.size))
    z_all_plus_1 = z_all_plus_1.at[0, :].set(z0)
    fp_eval_partial = partial(fp_eval_glista,
                              supervised=supervised,
                               z_star=z_star,
                               W=W,
                               D=D,
                               b=q,
                               params=params
                              )
    z_all = jnp.zeros((k, z0.size))
    val = z0, iter_losses, z_all, obj_diffs
    start_iter = 0
    if jit:
        out = lax.fori_loop(start_iter, k, fp_eval_partial, val)
    else:
        out = python_fori_loop(start_iter, k, fp_eval_partial, val)
    z_final, iter_losses, z_all, obj_diffs = out
    z_all_plus_1 = z_all_plus_1.at[1:, :].set(z_all)
    return z_final, iter_losses, z_all_plus_1, obj_diffs


def fixed_point_glista(z, W, D, b, gamma, theta, mu, nu, a):
    """
    applies the ista fixed point operator
    """
    # return soft_threshold(z - gamma * W.T.dot(D.dot(z) - b), theta)

    # gain
    gain = gain_gate(z, theta, mu, nu)
    z_gain = jnp.multiply(gain, z)
    z_tilde = fixed_point_alista(z_gain, W, D, b, gamma, theta)

    # overshoot
    overshoot = overshoot_gate(z, z_tilde, a)
    z_next = jnp.multiply(overshoot, z_tilde) + jnp.multiply((1 - overshoot), z)
    return z_next


def overshoot_gate(z, z_tilde, a):
    epsilon = .1
    return 1 + a / (jnp.abs(z_tilde - z) + epsilon)


def gain_gate(z, theta, mu, nu):
    # exponential function
    f = jnp.exp(-nu * jnp.abs(z))
    return 1 + mu * theta * f


def fp_train_glista(i, val, supervised, z_star, params, W, D, b):
    z, loss_vec = val
    gamma = params[i, 0] #jnp.exp(params[i, 0])
    theta = params[i, 1] #jnp.exp(params[i, 1])
    mu = params[i, 2]
    nu = params[i, 3]
    a = params[i, 4]
    z_next = fixed_point_glista(z, W, D, b, gamma, theta, mu, nu, a)
    diff = jnp.linalg.norm(z - z_star) ** 2
    loss_vec = loss_vec.at[i].set(diff)
    return z_next, loss_vec


def fp_eval_glista(i, val, supervised, z_star, params, W, D, b):
    z, loss_vec, z_all, obj_diffs = val
    gamma = params[i, 0]
    theta = params[i, 1]
    mu = params[i, 2]
    nu = params[i, 3]
    a = params[i, 4]
    z_next = fixed_point_glista(z, W, D, b, gamma, theta, mu, nu, a)
    diff = 10 * jnp.log10(jnp.linalg.norm(z - z_star) ** 2 / jnp.linalg.norm(z_star) ** 2)
    loss_vec = loss_vec.at[i].set(diff)
    obj = .5 * jnp.linalg.norm(D @ z_next - b) ** 2 # + lambd * jnp.linalg.norm(z_next, ord=1)
    opt_obj = .5 * jnp.linalg.norm(D @ z_star - b) ** 2 # + lambd * jnp.linalg.norm(z_star, ord=1)
    obj_diffs = obj_diffs.at[i].set(obj - opt_obj)
    z_all = z_all.at[i, :].set(z_next)
    return z_next, loss_vec, z_all, obj_diffs

def true_fun(_):
    return 0

def false_fun(_):
    return 1


def kl_inv_fn(r, q, c):
    p = (1 - q) * r + q
    return q * jnp.log(q / p) + (1 - q) * jnp.log((1 - q) / (1 - p)) - c
# def kl_inv_fn(p, q, c):
#     pen = lax.cond(p >= q, None, true_fun, None, false_fun)
#     return pen + q * jnp.log(q / p) + (1 - q) * jnp.log((1 - q) / (1 - p)) - c


def create_kl_inv_layer():
    """
    create the cvxpylayer
    """
    p = cp.Variable(2)
    q_param = cp.Parameter(2)
    c_param = cp.Parameter(1)

    constraints = [c_param >= cp.sum(cp.kl_div(q_param, p)), 
                   0 <= p[0], p[0] <= 1, p[1] == 1.0 - p[0]]
    objective = cp.Maximize(p[0])
    problem = cp.Problem(objective, constraints)
    assert problem.is_dpp()

    layer = CvxpyLayer(
        problem, parameters=[q_param, c_param], variables=[p]
    )

    return layer


def k_steps_train_lista_cpss(k, z0, q, params, D, supervised, z_star, jit):
    iter_losses = jnp.zeros(k)

    fp_train_partial = partial(fp_train_lista_cpss,
                               supervised=supervised,
                               z_star=z_star,
                               b=q,
                               D=D,
                               params=params
                               )
    val = z0, iter_losses
    start_iter = 0
    if jit:
        out = lax.fori_loop(start_iter, k, fp_train_partial, val)
    else:
        out = python_fori_loop(start_iter, k, fp_train_partial, val)
    z_final, iter_losses = out
    return z_final, iter_losses


def k_steps_eval_lista_cpss(k, z0, q, params, D, supervised, z_star, jit):
    iter_losses, obj_diffs = jnp.zeros(k), jnp.zeros(k)
    z_all_plus_1 = jnp.zeros((k + 1, z0.size))
    z_all_plus_1 = z_all_plus_1.at[0, :].set(z0)
    fp_eval_partial = partial(fp_eval_lista_cpss,
                              supervised=supervised,
                               z_star=z_star,
                               b=q,
                               D=D,
                               params=params
                              )
    z_all = jnp.zeros((k, z0.size))
    val = z0, iter_losses, z_all, obj_diffs
    start_iter = 0
    if jit:
        out = lax.fori_loop(start_iter, k, fp_eval_partial, val)
    else:
        out = python_fori_loop(start_iter, k, fp_eval_partial, val)
    z_final, iter_losses, z_all, obj_diffs = out
    z_all_plus_1 = z_all_plus_1.at[1:, :].set(z_all)
    return z_final, iter_losses, z_all_plus_1, obj_diffs


def fixed_point_lista_cpss(z, W_1, D, b, theta):
    """
    applies the ista fixed point operator
    """
    # return soft_threshold(W_1 @ b + W_2 @ z, theta)
    return soft_threshold(z - W_1.T.dot(D.dot(z) - b), theta)


def fp_train_lista_cpss(i, val, supervised, z_star, params, D, b):
    z, loss_vec = val
    # gamma = params[0][i, 0] #jnp.exp(params[i, 0])
    # theta = params[0][i, 1] #jnp.exp(params[i, 1])
    # W = params[1]
    theta = params[0][i]
    W_1 = params[1][:,:,i]
    # W_2 = params[2][:,:,i]
    # import pdb
    # pdb.set_trace()
    z_next = fixed_point_lista_cpss(z, W_1, D, b, theta)
    diff = jnp.linalg.norm(z - z_star) ** 2
    loss_vec = loss_vec.at[i].set(diff)
    return z_next, loss_vec


def fp_eval_lista_cpss(i, val, supervised, z_star, params, D, b):
    z, loss_vec, z_all, obj_diffs = val
    # gamma = params[0][i, 0] #jnp.exp(params[i, 0])
    # theta = params[0][i, 1] #jnp.exp(params[i, 1])
    # W = params[1]
    theta = params[0][i]
    W_1 = params[1][:,:,i]
    # W_2 = params[2][:,:,i]
    z_next = fixed_point_lista_cpss(z, W_1, D, b, theta)
    diff = 10 * jnp.log10(jnp.linalg.norm(z - z_star) ** 2 / jnp.linalg.norm(z_star) ** 2)
    loss_vec = loss_vec.at[i].set(diff)
    # obj = .5 * jnp.linalg.norm(D @ z_next - b) ** 2 # + lambd * jnp.linalg.norm(z_next, ord=1)
    # opt_obj = .5 * jnp.linalg.norm(D @ z_star - b) ** 2 # + lambd * jnp.linalg.norm(z_star, ord=1)
    # obj_diffs = obj_diffs.at[i].set(obj - opt_obj)
    z_all = z_all.at[i, :].set(z_next)
    return z_next, loss_vec, z_all, obj_diffs


def k_steps_train_lista(k, z0, q, params, supervised, z_star, jit):
    iter_losses = jnp.zeros(k)

    fp_train_partial = partial(fp_train_lista,
                               supervised=supervised,
                               z_star=z_star,
                               b=q,
                               params=params
                               )
    val = z0, iter_losses
    start_iter = 0
    if jit:
        out = lax.fori_loop(start_iter, k, fp_train_partial, val)
    else:
        out = python_fori_loop(start_iter, k, fp_train_partial, val)
    z_final, iter_losses = out
    return z_final, iter_losses


def k_steps_eval_lista(k, z0, q, params, supervised, z_star, jit):
    iter_losses, obj_diffs = jnp.zeros(k), jnp.zeros(k)
    z_all_plus_1 = jnp.zeros((k + 1, z0.size))
    z_all_plus_1 = z_all_plus_1.at[0, :].set(z0)
    fp_eval_partial = partial(fp_eval_lista,
                              supervised=supervised,
                               z_star=z_star,
                               b=q,
                               params=params
                              )
    z_all = jnp.zeros((k, z0.size))
    val = z0, iter_losses, z_all, obj_diffs
    start_iter = 0
    if jit:
        out = lax.fori_loop(start_iter, k, fp_eval_partial, val)
    else:
        out = python_fori_loop(start_iter, k, fp_eval_partial, val)
    z_final, iter_losses, z_all, obj_diffs = out
    z_all_plus_1 = z_all_plus_1.at[1:, :].set(z_all)
    return z_final, iter_losses, z_all_plus_1, obj_diffs


def fixed_point_lista(z, W_1, W_2, b, theta):
    """
    applies the ista fixed point operator
    """
    return soft_threshold(W_1 @ b + W_2 @ z, theta)
    # return soft_threshold(z - gamma * W.T.dot(D.dot(z) - b), theta)


def fp_train_lista(i, val, supervised, z_star, params, b):
    z, loss_vec = val
    # gamma = params[0][i, 0] #jnp.exp(params[i, 0])
    # theta = params[0][i, 1] #jnp.exp(params[i, 1])
    # W = params[1]
    theta = params[0][i]
    W_1 = params[1][:,:,i]
    W_2 = params[2][:,:,i]
    # import pdb
    # pdb.set_trace()
    z_next = fixed_point_lista(z, W_1, W_2, b, theta)
    diff = jnp.linalg.norm(z - z_star) ** 2
    loss_vec = loss_vec.at[i].set(diff)
    return z_next, loss_vec


def fp_eval_lista(i, val, supervised, z_star, params, b):
    z, loss_vec, z_all, obj_diffs = val
    # gamma = params[0][i, 0] #jnp.exp(params[i, 0])
    # theta = params[0][i, 1] #jnp.exp(params[i, 1])
    # W = params[1]
    theta = params[0][i]
    W_1 = params[1][:,:,i]
    W_2 = params[2][:,:,i]
    z_next = fixed_point_lista(z, W_1, W_2, b, theta)
    diff = 10 * jnp.log10(jnp.linalg.norm(z - z_star) ** 2 / jnp.linalg.norm(z_star) ** 2)
    loss_vec = loss_vec.at[i].set(diff)
    # obj = .5 * jnp.linalg.norm(D @ z_next - b) ** 2 # + lambd * jnp.linalg.norm(z_next, ord=1)
    # opt_obj = .5 * jnp.linalg.norm(D @ z_star - b) ** 2 # + lambd * jnp.linalg.norm(z_star, ord=1)
    # obj_diffs = obj_diffs.at[i].set(obj - opt_obj)
    z_all = z_all.at[i, :].set(z_next)
    return z_next, loss_vec, z_all, obj_diffs


def k_steps_train_tilista(k, z0, q, params, D, supervised, z_star, jit):
    iter_losses = jnp.zeros(k)

    fp_train_partial = partial(fp_train_tilista,
                               supervised=supervised,
                               z_star=z_star,
                               D=D,
                               b=q,
                               params=params
                               )
    val = z0, iter_losses
    start_iter = 0
    if jit:
        out = lax.fori_loop(start_iter, k, fp_train_partial, val)
    else:
        out = python_fori_loop(start_iter, k, fp_train_partial, val)
    z_final, iter_losses = out
    return z_final, iter_losses


def k_steps_eval_tilista(k, z0, q, params, D, supervised, z_star, jit):
    iter_losses, obj_diffs = jnp.zeros(k), jnp.zeros(k)
    z_all_plus_1 = jnp.zeros((k + 1, z0.size))
    z_all_plus_1 = z_all_plus_1.at[0, :].set(z0)
    fp_eval_partial = partial(fp_eval_tilista,
                              supervised=supervised,
                               z_star=z_star,
                               D=D,
                               b=q,
                               params=params
                              )
    z_all = jnp.zeros((k, z0.size))
    val = z0, iter_losses, z_all, obj_diffs
    start_iter = 0
    if jit:
        out = lax.fori_loop(start_iter, k, fp_eval_partial, val)
    else:
        out = python_fori_loop(start_iter, k, fp_eval_partial, val)
    z_final, iter_losses, z_all, obj_diffs = out
    z_all_plus_1 = z_all_plus_1.at[1:, :].set(z_all)
    return z_final, iter_losses, z_all_plus_1, obj_diffs


def fixed_point_tilista(z, W, D, b, gamma, theta):
    """
    applies the ista fixed point operator
    """
    return soft_threshold(z - gamma * W.T.dot(D.dot(z) - b), theta)


def fp_train_tilista(i, val, supervised, z_star, params, D, b):
    z, loss_vec = val
    gamma = params[0][i, 0] #jnp.exp(params[i, 0])
    theta = params[0][i, 1] #jnp.exp(params[i, 1])
    W = params[1]
    z_next = fixed_point_tilista(z, W, D, b, gamma, theta)
    diff = jnp.linalg.norm(z - z_star) ** 2
    loss_vec = loss_vec.at[i].set(diff)
    return z_next, loss_vec


def fp_eval_tilista(i, val, supervised, z_star, params, D, b):
    z, loss_vec, z_all, obj_diffs = val
    gamma = params[0][i, 0] #jnp.exp(params[i, 0])
    theta = params[0][i, 1] #jnp.exp(params[i, 1])
    W = params[1]
    z_next = fixed_point_tilista(z, W, D, b, gamma, theta)
    diff = 10 * jnp.log10(jnp.linalg.norm(z - z_star) ** 2 / jnp.linalg.norm(z_star) ** 2)
    loss_vec = loss_vec.at[i].set(diff)
    obj = .5 * jnp.linalg.norm(D @ z_next - b) ** 2 # + lambd * jnp.linalg.norm(z_next, ord=1)
    opt_obj = .5 * jnp.linalg.norm(D @ z_star - b) ** 2 # + lambd * jnp.linalg.norm(z_star, ord=1)
    obj_diffs = obj_diffs.at[i].set(obj - opt_obj)
    z_all = z_all.at[i, :].set(z_next)
    return z_next, loss_vec, z_all, obj_diffs


def k_steps_train_alista(k, z0, q, params, W, D, supervised, z_star, jit):
    iter_losses = jnp.zeros(k)

    fp_train_partial = partial(fp_train_alista,
                               supervised=supervised,
                               z_star=z_star,
                               W=W,
                               D=D,
                               b=q,
                               params=params
                               )
    val = z0, iter_losses
    start_iter = 0
    if jit:
        out = lax.fori_loop(start_iter, k, fp_train_partial, val)
    else:
        out = python_fori_loop(start_iter, k, fp_train_partial, val)
    z_final, iter_losses = out
    return z_final, iter_losses


def k_steps_eval_alista(k, z0, q, params, W, D, supervised, z_star, jit):
    iter_losses, obj_diffs = jnp.zeros(k), jnp.zeros(k)
    z_all_plus_1 = jnp.zeros((k + 1, z0.size))
    z_all_plus_1 = z_all_plus_1.at[0, :].set(z0)
    fp_eval_partial = partial(fp_eval_alista,
                              supervised=supervised,
                               z_star=z_star,
                               W=W,
                               D=D,
                               b=q,
                               params=params
                              )
    z_all = jnp.zeros((k, z0.size))
    val = z0, iter_losses, z_all, obj_diffs
    start_iter = 0
    if jit:
        out = lax.fori_loop(start_iter, k, fp_eval_partial, val)
    else:
        out = python_fori_loop(start_iter, k, fp_eval_partial, val)
    z_final, iter_losses, z_all, obj_diffs = out
    z_all_plus_1 = z_all_plus_1.at[1:, :].set(z_all)
    return z_final, iter_losses, z_all_plus_1, obj_diffs


def fixed_point_alista(z, W, D, b, gamma, theta):
    """
    applies the ista fixed point operator
    """
    return soft_threshold(z - gamma * W.T.dot(D.dot(z) - b), theta)


def fp_train_alista(i, val, supervised, z_star, params, W, D, b):
    z, loss_vec = val
    gamma = params[i, 0] #jnp.exp(params[i, 0])
    theta = params[i, 1] #jnp.exp(params[i, 1])
    z_next = fixed_point_alista(z, W, D, b, gamma, theta)
    diff = jnp.linalg.norm(z - z_star) ** 2
    loss_vec = loss_vec.at[i].set(diff)
    return z_next, loss_vec


def fp_eval_alista(i, val, supervised, z_star, params, W, D, b):
    z, loss_vec, z_all, obj_diffs = val
    gamma = params[i, 0]
    theta = params[i, 1]
    z_next = fixed_point_alista(z, W, D, b, gamma, theta)
    diff = 10 * jnp.log10(jnp.linalg.norm(z - z_star) ** 2 / jnp.linalg.norm(z_star) ** 2)
    loss_vec = loss_vec.at[i].set(diff)
    obj = .5 * jnp.linalg.norm(D @ z_next - b) ** 2 # + lambd * jnp.linalg.norm(z_next, ord=1)
    opt_obj = .5 * jnp.linalg.norm(D @ z_star - b) ** 2 # + lambd * jnp.linalg.norm(z_star, ord=1)
    obj_diffs = obj_diffs.at[i].set(obj - opt_obj)
    z_all = z_all.at[i, :].set(z_next)
    return z_next, loss_vec, z_all, obj_diffs


def cold_start_solve(fixed_point_fn, n, b_mat):
    N, m = b_mat.shape
    train_fn = create_train_fn(fixed_point_fn)
    batch_train_fn_proj_gd = vmap(train_fn, in_axes=(None, 0, 0, None, None, None), out_axes=(0, 0))
    z0_init_mat = jnp.zeros((N, n))
    z_finals, iter_losses = batch_train_fn_proj_gd(1000, z0_init_mat, b_mat, False, None, False)
    return z_finals, iter_losses


def create_train_fn(fixed_point_fn):
    def fp_train_generic(i, val, supervised, z_star, theta):
        z, loss_vec = val
        z_next = fixed_point_fn(z, theta)
        if supervised:
            diff = jnp.linalg.norm(z - z_star)
        else:
            diff = jnp.linalg.norm(z_next - z)
        loss_vec = loss_vec.at[i].set(diff)
        return z_next, loss_vec

    def k_steps_train(k, z0, q, supervised, z_star, jit):
        iter_losses = jnp.zeros(k)
        fp_train_partial = partial(fp_train_generic, supervised=supervised, z_star=z_star, theta=q)
        val = z0, iter_losses
        start_iter = 0
        if jit:
            out = lax.fori_loop(start_iter, k, fp_train_partial, val)
        else:
            out = python_fori_loop(start_iter, k, fp_train_partial, val)
        z_final, iter_losses = out
        return z_final, iter_losses
    return k_steps_train


def create_eval_fn(fixed_point_fn):
    def fp_train_generic(i, val, supervised, z_star, theta):
        z, loss_vec, z_all = val
        z_next = fixed_point_fn(z, theta)
        if supervised:
            diff = jnp.linalg.norm(z - z_star)
        else:
            diff = jnp.linalg.norm(z_next - z)
        loss_vec = loss_vec.at[i].set(diff)
        z_all = z_all.at[i, :].set(z_next)
        return z_next, loss_vec, z_all

    def k_steps_train(k, z0, q, supervised, z_star, jit):
        iter_losses = jnp.zeros(k)
        z_all_plus_1 = jnp.zeros((k + 1, z0.size))
        z_all_plus_1 = z_all_plus_1.at[0, :].set(z0)
        fp_train_partial = partial(fp_train_generic, supervised=supervised, z_star=z_star, theta=q)
        z_all = jnp.zeros((k, z0.size))
        val = z0, iter_losses, z_all
        start_iter = 0
        if jit:
            out = lax.fori_loop(start_iter, k, fp_train_partial, val)
        else:
            out = python_fori_loop(start_iter, k, fp_train_partial, val)
        z_final, iter_losses, z_all = out
        z_all_plus_1 = z_all_plus_1.at[1:, :].set(z_all)
        return z_final, iter_losses, z_all_plus_1
    return k_steps_train


# def create_k_steps_eval(fixed_point_fn):
#     iter_losses, obj_diffs = jnp.zeros(k), jnp.zeros(k)
#     z_all_plus_1 = jnp.zeros((k + 1, z0.size))
#     z_all_plus_1 = z_all_plus_1.at[0, :].set(z0)

#     f_theta = partial(f, theta=q)

#     fp_eval_partial = partial(fp_eval_extragrad,
#                               supervised=supervised,
#                               z_star=z_star,
#                               f=f_theta,
#                               proj_X=proj_X,
#                               proj_Y=proj_Y,
#                               eg_step=eg_step,
#                               n=n
#                               )
#     z_all = jnp.zeros((k, z0.size))
#     val = z0, iter_losses, z_all, obj_diffs
#     start_iter = 0
#     if jit:
#         out = lax.fori_loop(start_iter, k, fp_eval_partial, val)
#     else:
#         out = python_fori_loop(start_iter, k, fp_eval_partial, val)
#     z_final, iter_losses, z_all, obj_diffs = out
#     z_all_plus_1 = z_all_plus_1.at[1:, :].set(z_all)
#     return z_final, iter_losses, z_all_plus_1


def k_steps_eval_extragrad(k, z0, q, f, proj_X, proj_Y, n, eg_step, supervised, z_star, jit):
    iter_losses, obj_diffs = jnp.zeros(k), jnp.zeros(k)
    z_all_plus_1 = jnp.zeros((k + 1, z0.size))
    z_all_plus_1 = z_all_plus_1.at[0, :].set(z0)

    f_theta = partial(f, theta=q)

    fp_eval_partial = partial(fp_eval_extragrad,
                              supervised=supervised,
                              z_star=z_star,
                              f=f_theta,
                              proj_X=proj_X,
                              proj_Y=proj_Y,
                              eg_step=eg_step,
                              n=n
                              )
    z_all = jnp.zeros((k, z0.size))
    val = z0, iter_losses, z_all, obj_diffs
    start_iter = 0
    if jit:
        out = lax.fori_loop(start_iter, k, fp_eval_partial, val)
    else:
        out = python_fori_loop(start_iter, k, fp_eval_partial, val)
    z_final, iter_losses, z_all, obj_diffs = out
    z_all_plus_1 = z_all_plus_1.at[1:, :].set(z_all)
    return z_final, iter_losses, z_all_plus_1, obj_diffs


def k_steps_train_extragrad(k, z0, q, f, proj_X, proj_Y, n, eg_step, supervised, z_star, jit):
    """
    f is a function that takes in theta in addition to x and y, i.e., f(theta, x, y)
    """
    iter_losses = jnp.zeros(k)
    f_theta = partial(f, theta=q)

    fp_train_partial = partial(fp_train_extragrad,
                               supervised=supervised,
                               z_star=z_star,
                               f=f_theta,
                               proj_X=proj_X,
                               proj_Y=proj_Y,
                               eg_step=eg_step,
                               n=n
                               )
    val = z0, iter_losses
    start_iter = 0
    if jit:
        out = lax.fori_loop(start_iter, k, fp_train_partial, val)
    else:
        out = python_fori_loop(start_iter, k, fp_train_partial, val)
    z_final, iter_losses = out
    return z_final, iter_losses


def fp_train_extragrad(i, val, supervised, z_star, f, proj_X, proj_Y, eg_step, n):
    z, loss_vec = val
    z_next = fixed_point_extragrad(z, f, proj_X, proj_Y, eg_step, n)
    if supervised:
        diff = jnp.linalg.norm(z - z_star)
    else:
        diff = jnp.linalg.norm(z_next - z)
    loss_vec = loss_vec.at[i].set(diff)
    return z_next, loss_vec


# fp_train_extragrad(i, val, supervised, z_star, Q, R, A, c, b, eg_step)
def fp_eval_extragrad(i, val, supervised, z_star, f, proj_X, proj_Y, eg_step, n):
    z, loss_vec, z_all, obj_diffs = val
    z_next = fixed_point_extragrad(z, f, proj_X, proj_Y, eg_step, n)
    if supervised:
        diff = jnp.linalg.norm(z - z_star)
    else:
        diff = jnp.linalg.norm(z_next - z)
    loss_vec = loss_vec.at[i].set(diff)
    obj = 0
    opt_obj = 0
    obj_diffs = obj_diffs.at[i].set(obj - opt_obj)
    z_all = z_all.at[i, :].set(z_next)
    return z_next, loss_vec, z_all, obj_diffs


def fixed_point_extragrad(z, f, proj_X, proj_Y, eg_step, n):
    """
    applies the extragradient fixed point operator for 
    min_x max_y f(x, y)
        s.t. x in X, y in Y

    z = (x, y) in (R^n, R^m)

    proj_X and proj_Y are functions that do the projection
    f is a function t
    """
    x0 = z[:n]
    y0 = z[n:]

    # derivative of f wrt x
    fx = grad(f, argnums=0)
    fy = grad(f, argnums=1)
    x1 = proj_X(x0 - eg_step * fx(x0, y0))
    y1 = proj_Y(y0 + eg_step * fy(x0, y0))
    x2 = proj_X(x0 - eg_step * fx(x1, y1))
    y2 = proj_Y(y0 + eg_step * fx(x1, y1))
    return jnp.concatenate([x2, y2])


def form_osqp_matrix(P, A, rho_vec, sigma):
    m, n = A.shape
    return P + sigma * jnp.eye(n) + A.T @ jnp.diag(rho_vec) @ A


def eval_ista_obj(z, A, b, lambd):
    return .5 * jnp.linalg.norm(A @ z - b) ** 2 + lambd * jnp.linalg.norm(z, ord=1)


def fp_train(i, val, q_r, factor, supervised, z_star, proj, hsde, homogeneous, scale_vec, alpha):
    """
    q_r = r if hsde else q_r = q
    homogeneous tells us if we set tau = 1.0 or use the root_plus method
    """
    z, loss_vec = val
    if hsde:
        r = q_r
        z_next, u, u_tilde, v = fixed_point_hsde(z, homogeneous, r, factor, proj, scale_vec, alpha)
    else:
        q = q_r
        z_next, u, u_tilde, v = fixed_point(z, q, factor, proj, scale_vec, alpha)
    if supervised:
        # diff = jnp.linalg.norm(z[:-1] - z_star)
        diff = jnp.linalg.norm(z[:-1] / z[-1] - z_star)
    else:
        diff = jnp.linalg.norm(z_next / z_next[-1] - z / z[-1])
    loss_vec = loss_vec.at[i].set(diff)
    return z_next, loss_vec


def k_steps_train_osqp(k, z0, q, factor, A, rho, sigma, supervised, z_star, jit):
    iter_losses = jnp.zeros(k)
    m, n = A.shape

    # initialize z_init
    z_init = jnp.zeros((n + 2 * m))
    z_init = z_init.at[:m + n].set(z0)
    w = A @ z0[:n]
    z_init = z_init.at[m + n:].set(w)

    fp_train_partial = partial(fp_train_osqp,
                               supervised=supervised,
                               z_star=z_star,
                               factor=factor,
                               A=A,
                               q=q,
                               rho=rho,
                               sigma=sigma
                               )
    val = z_init, iter_losses
    start_iter = 0
    if jit:
        out = lax.fori_loop(start_iter, k, fp_train_partial, val)
    else:
        out = python_fori_loop(start_iter, k, fp_train_partial, val)
    z_final, iter_losses = out
    return z_final, iter_losses


def k_steps_eval_osqp(k, z0, q, factor, P, A, rho, sigma, supervised, z_star, jit, custom_loss=None):
    iter_losses = jnp.zeros(k)
    m, n = A.shape

    # initialize z_init
    z_init = jnp.zeros((n + 2 * m))
    z_init = z_init.at[:m + n].set(z0)
    w = A @ z0[:n]
    z_init = z_init.at[m + n:].set(w)

    z_all_plus_1 = jnp.zeros((k + 1, z_init.size))
    z_all_plus_1 = z_all_plus_1.at[0, :].set(z_init)
    fp_eval_partial = partial(fp_eval_osqp,
                              supervised=supervised,
                              z_star=z_star,
                              factor=factor,
                              P=P,
                              A=A,
                              q=q,
                              rho=rho,
                              sigma=sigma,
                              custom_loss=custom_loss
                              )
    z_all = jnp.zeros((k, z_init.size))
    primal_resids, dual_resids = jnp.zeros(k), jnp.zeros(k)
    val = z_init, iter_losses, z_all, primal_resids, dual_resids
    start_iter = 0
    if jit:
        out = lax.fori_loop(start_iter, k, fp_eval_partial, val)
    else:
        out = python_fori_loop(start_iter, k, fp_eval_partial, val)
    z_final, iter_losses, z_all, primal_resids, dual_resids = out
    z_all_plus_1 = z_all_plus_1.at[1:, :].set(z_all)
    return z_final, iter_losses, z_all_plus_1, primal_resids, dual_resids


def fp_train_osqp(i, val, supervised, z_star, factor, A, q, rho, sigma):
    z, loss_vec = val
    z_next = fixed_point_osqp(z, factor, A, q, rho, sigma)
    if supervised:
        diff = jnp.linalg.norm(z - z_star)
    else:
        diff = jnp.linalg.norm(z_next - z)
    loss_vec = loss_vec.at[i].set(diff)
    return z_next, loss_vec


def fp_eval_osqp(i, val, supervised, z_star, factor, P, A, q, rho, sigma, custom_loss=None, lightweight=False):
    m, n = A.shape
    z, loss_vec, z_all, primal_residuals, dual_residuals = val
    z_next = fixed_point_osqp(z, factor, A, q, rho, sigma)
    if custom_loss is None:
        if supervised:
            diff = jnp.linalg.norm(z - z_star)
        else:
            diff = jnp.linalg.norm(z_next - z)
    else:
        diff = custom_loss(z, z_star)
    loss_vec = loss_vec.at[i].set(diff)
    z_all = z_all.at[i, :].set(z_next)

    # primal and dual residuals
    if not lightweight:
        pr = jnp.linalg.norm(A @ z_next[:n] - z_next[n + m:])
        dr = jnp.linalg.norm(P @ z_next[:n] + A.T @ z_next[n:n + m] + q[:n])
        primal_residuals = primal_residuals.at[i].set(pr)
        dual_residuals = dual_residuals.at[i].set(dr)
    return z_next, loss_vec, z_all, primal_residuals, dual_residuals


def fixed_point_osqp(z, factor, A, q, rho, sigma):
    # z = (x, y, w) w is the z variable in osqp terminology
    m, n = A.shape
    x, y, w = z[:n], z[n:n + m], z[n + m:]
    c, l_bound, u_bound = q[:n], q[n:n + m], q[n + m:]

    # update (x, nu)
    rhs = sigma * x - c + A.T @ (rho * w - y)
    x_next = lin_sys_solve(factor, rhs)
    nu = rho * (A @ x_next - w) + y

    # update w_tilde
    w_tilde = w + (nu - y) / rho

    # update w
    w_next = jnp.clip(w_tilde + y / rho, a_min=l_bound, a_max=u_bound)

    # update y
    y_next = y + rho * (w_tilde - w_next)

    # concatenate into the fixed point vector
    z_next = jnp.concatenate([x_next, y_next, w_next])

    return z_next


def fp_train_ista(i, val, supervised, z_star, A, b, lambd, ista_step):
    z, loss_vec = val
    z_next = fixed_point_ista(z, A, b, lambd, ista_step)
    if supervised:
        diff = jnp.linalg.norm(z - z_star)
    else:
        diff = jnp.linalg.norm(z_next - z)
    loss_vec = loss_vec.at[i].set(diff)
    return z_next, loss_vec


def fp_train_gd(i, val, supervised, z_star, P, c, gd_step):
    z, loss_vec = val
    z_next = fixed_point_gd(z, P, c, gd_step)
    if supervised:
        diff = jnp.linalg.norm(z - z_star)
    else:
        diff = jnp.linalg.norm(z_next - z)
    loss_vec = loss_vec.at[i].set(diff)
    return z_next, loss_vec


def fp_train_fista(i, val, supervised, z_star, A, b, lambd, ista_step):
    z, y, t, loss_vec = val
    z_next, y_next, t_next = fixed_point_fista(z, y, t, A, b, lambd, ista_step)
    if supervised:
        diff = jnp.linalg.norm(z - z_star)
    else:
        diff = jnp.linalg.norm(z_next - z)
    # diff = eval_ista_obj(z_next, A, b, lambd)
    loss_vec = loss_vec.at[i].set(diff)
    return z_next, y_next, t_next, loss_vec


def fp_eval_ista(i, val, supervised, z_star, A, b, lambd, ista_step):
    z, loss_vec, z_all, obj_diffs = val
    z_next = fixed_point_ista(z, A, b, lambd, ista_step)
    if supervised:
        diff = jnp.linalg.norm(z - z_star)
    else:
        diff = jnp.linalg.norm(z_next - z)
    loss_vec = loss_vec.at[i].set(diff)
    obj = .5 * jnp.linalg.norm(A @ z_next - b) ** 2 + lambd * jnp.linalg.norm(z_next, ord=1)
    opt_obj = .5 * jnp.linalg.norm(A @ z_star - b) ** 2 + lambd * jnp.linalg.norm(z_star, ord=1)
    obj_diffs = obj_diffs.at[i].set(obj - opt_obj)
    z_all = z_all.at[i, :].set(z_next)
    return z_next, loss_vec, z_all, obj_diffs


def fp_eval_gd(i, val, supervised, z_star, P, c, gd_step):
    z, loss_vec, z_all, obj_diffs = val
    z_next = fixed_point_gd(z, P, c, gd_step)
    if supervised:
        diff = jnp.linalg.norm(z - z_star)
    else:
        diff = jnp.linalg.norm(z_next - z)
    loss_vec = loss_vec.at[i].set(diff)
    obj = .5 * z_next @ P @ z_next + c @ z_next
    opt_obj = .5 * z_star @ P @ z_star + c @ z_star
    obj_diffs = obj_diffs.at[i].set(obj - opt_obj)
    z_all = z_all.at[i, :].set(z_next)
    return z_next, loss_vec, z_all, obj_diffs


def fp_eval_fista(i, val, supervised, z_star, A, b, lambd, ista_step):
    z, y, t, loss_vec, z_all = val
    z_next, y_next, t_next = fixed_point_fista(z, y, t, A, b, lambd, ista_step)
    if supervised:
        diff = 10 * jnp.log10(jnp.linalg.norm(z - z_star) ** 2 / jnp.linalg.norm(z_star) ** 2)
        # diff = jnp.linalg.norm(z - z_star)
    else:
        diff = jnp.linalg.norm(z_next - z)
    loss_vec = loss_vec.at[i].set(diff)
    z_all = z_all.at[i, :].set(z_next)
    return z_next, y_next, t_next, loss_vec, z_all


def fp_eval(i, val, q_r, z_star, factor, proj, P, A, c, b, hsde, homogeneous, scale_vec, alpha,
            lightweight=False, custom_loss=None, verbose=False):
    """
    q_r = r if hsde else q_r = q
    homogeneous tells us if we set tau = 1.0 or use the root_plus method
    """
    m, n = A.shape
    z, z_prev, loss_vec, all_z, all_u, all_v, primal_residuals, dual_residuals = val

    if hsde:
        r = q_r
        z_next, u, u_tilde, v = fixed_point_hsde(
            z, homogeneous, r, factor, proj, scale_vec, alpha, verbose=verbose)
    else:
        q = q_r
        z_next, u, u_tilde, v = fixed_point(z, q, factor, proj, scale_vec, alpha, verbose=verbose)

    # diff = jnp.linalg.norm(z_next / z_next[-1] - z / z[-1])
    if custom_loss is None:
        diff = jnp.linalg.norm(z_next / z_next[-1] - z / z[-1])
    else:
        diff = rkf_loss(z_next / z_next[-1], z_star)
    loss_vec = loss_vec.at[i].set(diff)

    # primal and dual residuals
    if not lightweight:
        x, y, s = extract_sol(u, v, n, hsde)
        pr = jnp.linalg.norm(A @ x + s - b)
        dr = jnp.linalg.norm(A.T @ y + P @ x + c)
        primal_residuals = primal_residuals.at[i].set(pr)
        dual_residuals = dual_residuals.at[i].set(dr)
    all_z = all_z.at[i, :].set(z_next)
    all_u = all_u.at[i, :].set(u)
    all_v = all_v.at[i, :].set(v)
    return z_next, z_prev, loss_vec, all_z, all_u, all_v, primal_residuals, dual_residuals


def rkf_loss(z_next, z_star):
    x_mat = jnp.reshape(z_next[:100], (50, 2))
    x_star_mat = jnp.reshape(z_star[:100], (50, 2))
    norms = jnp.linalg.norm(x_mat - x_star_mat, axis=1)
    max_norm = jnp.max(norms)
    return max_norm


def k_steps_train_scs(k, z0, q, factor, supervised, z_star, proj, jit, hsde, m, n, zero_cone_size,
                      rho_x=1, scale=1, alpha=1.0):
    iter_losses = jnp.zeros(k)
    scale_vec = get_scale_vec(rho_x, scale, m, n, zero_cone_size, hsde=hsde)

    fp_train_partial = partial(fp_train, q_r=q, factor=factor,
                               supervised=supervised, z_star=z_star, proj=proj, hsde=hsde,
                               homogeneous=True, scale_vec=scale_vec, alpha=alpha)

    if hsde:
        # first step: iteration 0
        # we set homogeneous = False for the first iteration
        #   to match the SCS code which has the global variable FEASIBLE_ITERS
        #   which is set to 1
        homogeneous = False
        z_next, u, u_tilde, v = fixed_point_hsde(
            z0, homogeneous, q, factor, proj, scale_vec, alpha)
        iter_losses = iter_losses.at[0].set(jnp.linalg.norm(z_next - z0))
        z0 = z_next
    val = z0, iter_losses
    start_iter = 1 if hsde else 0
    if jit:
        out = lax.fori_loop(start_iter, k, fp_train_partial, val)
    else:
        out = python_fori_loop(start_iter, k, fp_train_partial, val)
    z_final, iter_losses = out
    return z_final, iter_losses


def k_steps_train_fista(k, z0, q, lambd, A, ista_step, supervised, z_star, jit):
    iter_losses = jnp.zeros(k)

    fp_train_partial = partial(fp_train_fista,
                               supervised=supervised,
                               z_star=z_star,
                               A=A,
                               b=q,
                               lambd=lambd,
                               ista_step=ista_step
                               )
    val = z0, z0, 1, iter_losses
    start_iter = 0
    if jit:
        out = lax.fori_loop(start_iter, k, fp_train_partial, val)
    else:
        out = python_fori_loop(start_iter, k, fp_train_partial, val)
    z_final, y_final, t_final, iter_losses = out
    return z_final, iter_losses


def k_steps_train_ista(k, z0, q, lambd, A, ista_step, supervised, z_star, jit):
    iter_losses = jnp.zeros(k)

    fp_train_partial = partial(fp_train_ista,
                               supervised=supervised,
                               z_star=z_star,
                               A=A,
                               b=q,
                               lambd=lambd,
                               ista_step=ista_step
                               )
    val = z0, iter_losses
    start_iter = 0
    if jit:
        out = lax.fori_loop(start_iter, k, fp_train_partial, val)
    else:
        out = python_fori_loop(start_iter, k, fp_train_partial, val)
    z_final, iter_losses = out
    return z_final, iter_losses


def k_steps_train_gd(k, z0, q, P, gd_step, supervised, z_star, jit):
    iter_losses = jnp.zeros(k)

    fp_train_partial = partial(fp_train_gd,
                               supervised=supervised,
                               z_star=z_star,
                               P=P,
                               c=q,
                               gd_step=gd_step
                               )
    val = z0, iter_losses
    start_iter = 0
    if jit:
        out = lax.fori_loop(start_iter, k, fp_train_partial, val)
    else:
        out = python_fori_loop(start_iter, k, fp_train_partial, val)
    z_final, iter_losses = out
    return z_final, iter_losses


def k_steps_eval_fista(k, z0, q, lambd, A, ista_step, supervised, z_star, jit):
    iter_losses = jnp.zeros(k)
    z_all_plus_1 = jnp.zeros((k + 1, z0.size))
    z_all_plus_1 = z_all_plus_1.at[0, :].set(z0)
    fp_eval_partial = partial(fp_eval_fista,
                              supervised=supervised,
                              z_star=z_star,
                              A=A,
                              b=q,
                              lambd=lambd,
                              ista_step=ista_step
                              )
    z_all = jnp.zeros((k, z0.size))
    val = z0, z0, 1, iter_losses, z_all
    start_iter = 0
    if jit:
        out = lax.fori_loop(start_iter, k, fp_eval_partial, val)
    else:
        out = python_fori_loop(start_iter, k, fp_eval_partial, val)
    z_final, y_final, t_final, iter_losses, z_all = out
    z_all_plus_1 = z_all_plus_1.at[1:, :].set(z_all)
    return z_final, iter_losses, z_all_plus_1, iter_losses


def k_steps_eval_ista(k, z0, q, lambd, A, ista_step, supervised, z_star, jit):
    iter_losses, obj_diffs = jnp.zeros(k), jnp.zeros(k)
    z_all_plus_1 = jnp.zeros((k + 1, z0.size))
    z_all_plus_1 = z_all_plus_1.at[0, :].set(z0)
    fp_eval_partial = partial(fp_eval_ista,
                              supervised=supervised,
                              z_star=z_star,
                              A=A,
                              b=q,
                              lambd=lambd,
                              ista_step=ista_step
                              )
    z_all = jnp.zeros((k, z0.size))
    val = z0, iter_losses, z_all, obj_diffs
    start_iter = 0
    if jit:
        out = lax.fori_loop(start_iter, k, fp_eval_partial, val)
    else:
        out = python_fori_loop(start_iter, k, fp_eval_partial, val)
    z_final, iter_losses, z_all, obj_diffs = out
    z_all_plus_1 = z_all_plus_1.at[1:, :].set(z_all)
    return z_final, iter_losses, z_all_plus_1, obj_diffs


def k_steps_eval_gd(k, z0, q, P, gd_step, supervised, z_star, jit):
    iter_losses = jnp.zeros(k)
    z_all_plus_1 = jnp.zeros((k + 1, z0.size))
    z_all_plus_1 = z_all_plus_1.at[0, :].set(z0)
    fp_eval_partial = partial(fp_eval_gd,
                              supervised=supervised,
                              z_star=z_star,
                              P=P,
                              c=q,
                              gd_step=gd_step
                              )
    z_all = jnp.zeros((k, z0.size))
    obj_diffs = jnp.zeros(k)
    val = z0, iter_losses, z_all, obj_diffs
    start_iter = 0
    if jit:
        out = lax.fori_loop(start_iter, k, fp_eval_partial, val)
    else:
        out = python_fori_loop(start_iter, k, fp_eval_partial, val)
    z_final, iter_losses, z_all, obj_diffs = out
    z_all_plus_1 = z_all_plus_1.at[1:, :].set(z_all)
    return z_final, iter_losses, z_all_plus_1, obj_diffs


def k_steps_eval_scs(k, z0, q, factor, proj, P, A, supervised, z_star, jit, hsde, zero_cone_size,
                     rho_x=1, scale=1, alpha=1.0, custom_loss=None, lightweight=False):
    """
    if k = 500 we store u_1, ..., u_500 and z_0, z_1, ..., z_500
        which is why we have all_z_plus_1
    """
    all_u, all_z = jnp.zeros((k, z0.size)), jnp.zeros((k, z0.size))
    all_z_plus_1 = jnp.zeros((k + 1, z0.size))
    all_z_plus_1 = all_z_plus_1.at[0, :].set(z0)
    all_v = jnp.zeros((k, z0.size))
    iter_losses = jnp.zeros(k)
    primal_residuals, dual_residuals = jnp.zeros(k), jnp.zeros(k)
    m, n = A.shape
    scale_vec = get_scale_vec(rho_x, scale, m, n, zero_cone_size, hsde=hsde)

    if jit:
        verbose = False
    else:
        verbose = True

    if hsde:
        # first step: iteration 0
        # we set homogeneous = False for the first iteration
        #   to match the SCS code which has the global variable FEASIBLE_ITERS
        #   which is set to 1
        homogeneous = False

        z_next, u, u_tilde, v = fixed_point_hsde(
            z0, homogeneous, q, factor, proj, scale_vec, alpha, verbose=verbose)
        all_z = all_z.at[0, :].set(z_next)
        all_u = all_u.at[0, :].set(u)
        all_v = all_v.at[0, :].set(v)
        iter_losses = iter_losses.at[0].set(jnp.linalg.norm(z_next - z0))
        z0 = z_next
    # c, b = q[:n], q[n:]
    M = create_M(P, A)
    rhs = (M + jnp.diag(scale_vec)) @ q
    # get_scaled_factor(M, factor)
    c, b = rhs[:n], rhs[n:]
    # print('b', b)

    fp_eval_partial = partial(fp_eval, q_r=q, z_star=z_star, factor=factor,
                              proj=proj, P=P, A=A, c=c, b=b, hsde=hsde,
                              homogeneous=True, scale_vec=scale_vec, alpha=alpha,
                              custom_loss=custom_loss,
                              verbose=verbose)
    val = z0, z0, iter_losses, all_z, all_u, all_v, primal_residuals, dual_residuals
    start_iter = 1 if hsde else 0
    if jit:
        out = lax.fori_loop(start_iter, k, fp_eval_partial, val)
    else:
        out = python_fori_loop(start_iter, k, fp_eval_partial, val)
    z_final, z_penult, iter_losses, all_z, all_u, all_v, primal_residuals, dual_residuals = out
    all_z_plus_1 = all_z_plus_1.at[1:, :].set(all_z)

    # return z_final, iter_losses, primal_residuals, dual_residuals, all_z_plus_1, all_u, all_v
    if lightweight:
        return z_final, iter_losses, all_z_plus_1[:10, :], primal_residuals, \
            dual_residuals, all_u[:10, :], all_v[:10, :]
    return z_final, iter_losses, all_z_plus_1, primal_residuals, dual_residuals, all_u, all_v


def get_scale_vec(rho_x, scale, m, n, zero_cone_size, hsde=True):
    """
    Returns the non-identity DR scaling vector
        which is used as a diagonal matrix

    scale_vec = (r_x, r_y)
    where r_x = rho_x * ones(n)
          r_y[:zero_cone_size] = 1 / (1000 * scale) * ones(zero_cone_size)
          r_y[zero_cone_size:] = 1 / scale * ones(m - zero_cone_size)
    scaling for y depends on if it's for the zero cone or not
    """
    scale_vec = jnp.ones(m + n)

    # x-component of scale_vec set to rho_x
    scale_vec = scale_vec.at[:n].set(rho_x)

    # zero cone of y-component of scale_vec set to 1 / (1000 * scale)
    if hsde:
        zero_scale_factor = 1 #1000
    else:
        zero_scale_factor = 1
    scale_vec = scale_vec.at[n:n + zero_cone_size].set(1 / (zero_scale_factor * scale))

    # other parts of y-component of scale_vec set to 1 / scale
    scale_vec = scale_vec.at[n + zero_cone_size:].set(1 / scale)

    return scale_vec


def get_scaled_factor(M, scale_vec):
    """
    given the non-identity DR scaling and M this method returns the factored matrix
    of M + diag(scale_vec)
    """
    scale_vec_diag = jnp.diag(scale_vec)
    factor = jsp.linalg.lu_factor(M + scale_vec_diag)

    # this is to replace the lu factor and use cg
    # M_plus_scale = M + scale_vec_diag
    # def lhs_mat(x):
    #     return M_plus_scale @ x
    # factor = jit(lhs_mat)
    return factor


def get_scaled_vec_and_factor(M, rho_x, scale, m, n, zero_cone_size, hsde=True):
    scale_vec = get_scale_vec(rho_x, scale, m, n, zero_cone_size, hsde=hsde)
    return get_scaled_factor(M, scale_vec), scale_vec


def extract_sol(u, v, n, hsde):
    if hsde:
        tau = u[-1]
        x, y, s = u[:n] / tau, u[n:-1] / tau, v[n:-1] / tau
    else:
        x, y, s = u[:n], u[n:], v[n:]
    return x, y, s


def create_projection_fn(cones, n):
    """
    cones is a dict with keys
    z: zero cone
    l: non-negative cone
    q: second-order cone
    s: positive semidefinite cone

    n is the size of the variable x in the problem
    min 1/2 x^T P x + c^T x
        s.t. Ax + s = b
             s in K
    This function returns a projection Pi
    which is defined by
    Pi(w) = argmin_v ||w - v||_2^2
                s.t. v in C
    where
    C = {0}^n x K^*
    i.e. the cartesian product of the zero cone of length n and the dual
        cone of K
    For all of the cones we consider, the cones are self-dual
    """
    zero_cone, nonneg_cone = cones['z'], cones['l']
    soc = 'q' in cones.keys() and len(cones['q']) > 0
    sdp = 's' in cones.keys() and len(cones['s']) > 0
    if soc:
        soc_cones_array = jnp.array(cones['q'])

        # soc_proj_sizes, soc_num_proj are lists
        # need to convert to list so that the item is not a traced object
        soc_proj_sizes, soc_num_proj = count_num_repeated_elements(soc_cones_array)
    else:
        soc_proj_sizes, soc_num_proj = [], []
    if sdp:
        sdp_cones_array = jnp.array(cones['s'])

        # soc_proj_sizes, soc_num_proj are lists
        # need to convert to list so that the item is not a traced object
        sdp_row_sizes, sdp_num_proj = count_num_repeated_elements(sdp_cones_array)
        sdp_vector_sizes = [int(row_size * (row_size + 1) / 2) for row_size in sdp_row_sizes]
    else:
        sdp_row_sizes, sdp_vector_sizes, sdp_num_proj = [], [], []

    projection = partial(proj,
                         n=n,
                         zero_cone_int=int(zero_cone),
                         nonneg_cone_int=int(nonneg_cone),
                         soc_proj_sizes=soc_proj_sizes,
                         soc_num_proj=soc_num_proj,
                         sdp_row_sizes=sdp_row_sizes,
                         sdp_vector_sizes=sdp_vector_sizes,
                         sdp_num_proj=sdp_num_proj,
                         )
    return jit(projection)


def get_psd_sizes(cones):
    """
    returns a list with the size of the psd projections
    """
    sdp = 's' in cones.keys() and len(cones['s']) > 0
    if sdp:
        sdp_cones_array = jnp.array(cones['s'])
        psd_sizes = sdp_cones_array.tolist()
    else:
        psd_sizes = [0]
    return psd_sizes


def root_plus(mu, eta, p, r, scale_vec):
    """
    mu, p, r are vectors each with size (m + n)
    eta is a scalar

    A step that solves the linear system
    (I + M)z + q tau = mu^k
    tau^2 - tau(eta^k + z^Tq) - z^T M z = 0
    where z in reals^d and tau > 0

    Since M is monotone, z^T M z >= 0
    Quadratic equation will have one non-negative root and one non-positive root

    solve by substituting z = p^k - r tau
        where r = (I + M)^{-1} q
        and p^k = (I + M)^{-1} mu^k

    the result is a closed-form solution involving the quadratic formula
        we take the positive root
    """
    r_scaled = jnp.multiply(r, scale_vec)
    a = TAU_FACTOR + r @ r_scaled
    b = mu @ r_scaled - 2 * r_scaled @ p - eta * TAU_FACTOR
    c = jnp.multiply(p, scale_vec) @ (p - mu)
    return (-b + jnp.sqrt(b ** 2 - 4 * a * c)) / (2 * a)


def fixed_point_ista(z, A, b, lambd, ista_step):
    """
    applies the ista fixed point operator
    """
    return soft_threshold(z + ista_step * A.T.dot(b - A.dot(z)), ista_step * lambd)


def fixed_point_gd(z, P, c, gd_step):
    """
    applies the ista fixed point operator
    """
    grad = P @ z + c
    return z - gd_step * grad


def fixed_point_fista(z, y, t, A, b, lambd, ista_step):
    """
    applies the fista fixed point operator
    """
    z_next = fixed_point_ista(y, A, b, lambd, ista_step)
    t_next = .5 * (1 + jnp.sqrt(1 + 4 * t ** 2))
    y_next = z_next + (t - 1) / t_next * (z_next - z)
    return z_next, y_next, t_next


def soft_threshold(z, alpha):
    """
    soft-thresholding function for ista
    """
    return jnp.clip(jnp.abs(z) - alpha, a_min=0) * jnp.sign(z)


def fixed_point(z_init, q, factor, proj, scale_vec, alpha, verbose=False):
    """
    implements 1 iteration of algorithm 1 in https://arxiv.org/pdf/2212.08260.pdf
    """
    rhs = jnp.multiply(z_init - q, scale_vec)
    u_tilde = lin_sys_solve(factor, rhs)
    u_temp = 2 * u_tilde - z_init
    u = proj(u_temp)
    v = jnp.multiply(u + z_init - 2 * u_tilde, scale_vec)
    z = z_init + alpha * (u - u_tilde)
    if verbose:
        print('pre-solve u_tilde', rhs)
        print('u_tilde', u_tilde)
        print('u', u)
        print('z', z)
    return z, u, u_tilde, v


def fixed_point_hsde(z_init, homogeneous, r, factor, proj, scale_vec, alpha, verbose=False):
    """
    implements 1 iteration of algorithm 5.1 in https://arxiv.org/pdf/2004.02177.pdf

    the names of the variables are a bit different compared with that paper

    we have
    u_tilde = (w_tilde, tau_tilde)
    u = (w, tau)
    z = (mu, eta)

    they have
    u_tilde = (z_tilde, tau_tilde)
    u = (z, tau)
    w = (mu, eta)

    tau_tilde, tau, eta are all scalars
    w_tilde, w, mu all have size (m + n)

    r = (I + M)^{-1} q
    requires the inital eta > 0

    if homogeneous, we normalize z s.t. ||z|| = sqrt(m + n + 1)
        and we do the root_plus calculation for tau_tilde
    else
        no normalization
        tau_tilde = 1 (bias towards feasibility)
    """

    if homogeneous:
        z_init = z_init / jnp.linalg.norm(z_init) * jnp.sqrt(z_init.size)

    # z = (mu, eta)
    mu, eta = z_init[:-1], z_init[-1]

    # u_tilde, tau_tilde update

    # non identity DR scaling
    rhs = jnp.multiply(scale_vec, mu)
    p = lin_sys_solve(factor, rhs)

    # non identity DR scaling
    # p = jnp.multiply(scale_vec, p)
    if homogeneous:
        tau_tilde = root_plus(mu, eta, p, r, scale_vec)
    else:
        tau_tilde = 1.0
    w_tilde = p - r * tau_tilde

    # u, tau update
    w_temp = 2 * w_tilde - mu
    w = proj(w_temp)
    tau = jnp.clip(2 * tau_tilde - eta, a_min=0)

    # mu, eta update
    mu = mu + alpha * (w - w_tilde)
    eta = eta + alpha * (tau - tau_tilde)

    # concatenate for z, u
    z = jnp.concatenate([mu, jnp.array([eta])])
    u = jnp.concatenate([w, jnp.array([tau])])
    u_tilde = jnp.concatenate([w_tilde, jnp.array([tau_tilde])])

    # for s extraction - not needed for algorithm
    full_scaled_vec = jnp.concatenate([scale_vec, jnp.array([TAU_FACTOR])])
    v = jnp.multiply(full_scaled_vec,  u + z_init - 2 * u_tilde)

    # z and u have size (m + n + 1)
    # v has shape (m + n)
    if verbose:
        print('pre-solve u_tilde', rhs)
        print('u_tilde', u_tilde)
        print('u', u)
        print('z', z)
    return z, u, u_tilde, v


def create_M(P, A):
    """
    create the matrix M in jax
    M = [ P   A.T
         -A   0  ]
    """
    m, n = A.shape
    M = jnp.zeros((n + m, n + m))
    M = M.at[:n, :n].set(P)
    M = M.at[:n, n:].set(A.T)
    M = M.at[n:, :n].set(-A)
    return M


def lin_sys_solve(factor, b):
    """
    solves the linear system
    Ax = b
    where factor is the lu factorization of A
    """
    return jsp.linalg.lu_solve(factor, b)
    # return jsp.sparse.linalg.cg(factor, b)


def proj(input, n, zero_cone_int, nonneg_cone_int, soc_proj_sizes, soc_num_proj, sdp_row_sizes,
         sdp_vector_sizes, sdp_num_proj):
    """
    projects the input onto a cone which is a cartesian product of the zero cone,
        non-negative orthant, many second order cones, and many positive semidefinite cones

    Assumes that the ordering is as follows
    zero, non-negative orthant, second order cone, psd cone
    ============================================================================
    SECOND ORDER CONE
    soc_proj_sizes: list of the sizes of the socp projections needed
    soc_num_proj: list of the number of socp projections needed for each size

    e.g. 50 socp projections of size 3 and 1 projection of size 100 would be
    soc_proj_sizes = [3, 100]
    soc_num_proj = [50, 1]
    ============================================================================
    PSD CONE
    sdp_proj_sizes: list of the sizes of the sdp projections needed
    sdp_vector_sizes: list of the sizes of the sdp projections needed
    soc_num_proj: list of the number of socp projections needed for each size

    e.g. 3 sdp projections of size 10, 10, and 100 would be
    sdp_proj_sizes = [10, 100]
    sdp_vector_sizes = [55, 5050]
    sdp_num_proj = [2, 1]
    """
    nonneg = jnp.clip(input[n + zero_cone_int: n + zero_cone_int + nonneg_cone_int], a_min=0)
    projection = jnp.concatenate([input[:n + zero_cone_int], nonneg])

    # soc setup
    num_soc_blocks = len(soc_proj_sizes)

    # avoiding doing inner product using jax so that we can jit
    soc_total = sum(i[0] * i[1] for i in zip(soc_proj_sizes, soc_num_proj))
    soc_bool = num_soc_blocks > 0

    # sdp setup
    num_sdp_blocks = len(sdp_row_sizes)
    sdp_total = sum(i[0] * i[1] for i in zip(sdp_vector_sizes, sdp_num_proj))
    sdp_bool = num_sdp_blocks > 0

    if soc_bool:
        socp = jnp.zeros(soc_total)
        soc_input = input[n+zero_cone_int+nonneg_cone_int:n +
                          zero_cone_int+nonneg_cone_int + soc_total]

        # iterate over the blocks
        start = 0
        for i in range(num_soc_blocks):
            # calculate the end point
            end = start + soc_proj_sizes[i] * soc_num_proj[i]

            # extract the right soc_input
            curr_soc_input = lax.dynamic_slice(
                soc_input, (start,), (soc_proj_sizes[i] * soc_num_proj[i],))

            # reshape so that we vmap all of the socp projections of the same size together
            curr_soc_input_reshaped = jnp.reshape(
                curr_soc_input, (soc_num_proj[i], soc_proj_sizes[i]))
            curr_soc_out_reshaped = soc_proj_single_batch(curr_soc_input_reshaped)
            curr_socp = jnp.ravel(curr_soc_out_reshaped)

            # place in the correct location in the socp vector
            socp = socp.at[start:end].set(curr_socp)

            # update the start point
            start = end

        projection = jnp.concatenate([projection, socp])
    if sdp_bool:
        sdp_proj = jnp.zeros(sdp_total)
        sdp_input = input[n + zero_cone_int + nonneg_cone_int + soc_total:]

        # iterate over the blocks
        start = 0
        for i in range(num_sdp_blocks):
            # calculate the end point
            end = start + sdp_vector_sizes[i] * sdp_num_proj[i]

            # extract the right sdp_input
            curr_sdp_input = lax.dynamic_slice(
                sdp_input, (start,), (sdp_vector_sizes[i] * sdp_num_proj[i],))

            # reshape so that we vmap all of the sdp projections of the same size together
            curr_sdp_input_reshaped = jnp.reshape(
                curr_sdp_input, (sdp_num_proj[i], sdp_vector_sizes[i]))
            curr_sdp_out_reshaped = sdp_proj_batch(curr_sdp_input_reshaped, sdp_row_sizes[i])
            curr_sdp = jnp.ravel(curr_sdp_out_reshaped)

            # place in the correct location in the sdp vector
            sdp_proj = sdp_proj.at[start:end].set(curr_sdp)

            # update the start point
            start = end

        projection = jnp.concatenate([projection, sdp_proj])
    return projection


def count_num_repeated_elements(vector):
    """
    given a vector, outputs the frequency in a row

    e.g. vector = [5, 5, 10, 10, 5]

    val_repeated = [5, 10, 5]
    num_repeated = [2, 2, 1]
    """
    m = jnp.r_[True, vector[:-1] != vector[1:], True]
    counts = jnp.diff(jnp.flatnonzero(m))
    unq = vector[m[:-1]]
    out = jnp.c_[unq, counts]

    val_repeated = out[:, 0].tolist()
    num_repeated = out[:, 1].tolist()
    return val_repeated, num_repeated


def soc_proj_single(input):
    """
    input is a single vector
        input = (s, y) where y is a vector and s is a scalar
    then we call soc_projection
    """
    # break into scalar and vector parts
    y, s = input[1:], input[0]

    # do the projection
    pi_y, pi_s = soc_projection(y, s)

    # stitch the pieces back together
    return jnp.append(pi_s, pi_y)


def sdp_proj_single(x, n):
    """
    x_proj = argmin_y ||y - x||_2^2
                s.t.   y is psd
    x is a vector with shape (n * (n + 1) / 2)

    we need to pass in n to jit this function
        we could extract dim from x.shape theoretically,
        but we cannot jit a function
        whose values depend on the size of inputs
    """
    # convert vector of size (n * (n + 1) / 2) to matrix of shape (n, n)
    X = unvec_symm(x, n)

    # do the eigendecomposition of X
    evals, evecs = jnp.linalg.eigh(X)

    # clip the negative eigenvalues
    evals_plus = jnp.clip(evals, 0, jnp.inf)

    # put the projection together with non-neg eigenvalues
    X_proj = evecs @ jnp.diag(evals_plus) @ evecs.T

    # vectorize the matrix
    x_proj = vec_symm(X_proj)
    return x_proj


def soc_projection(x, s):
    """
    returns the second order cone projection of (x, s)
    (y, t) = Pi_{K}(x, s)
    where K = {y, t | ||y||_2 <= t}

    the second order cone admits a closed form solution

    (y, t) = alpha (x, ||x||_2) if ||x|| >= |s|
             (x, s) if ||x|| <= |s|, s >= 0
             (0, 0) if ||x|| <= |s|, s <= 0

    where alpha = (s + ||x||_2) / (2 ||x||_2)

    case 1: ||x|| >= |s|
    case 2: ||x|| >= |s|
        case 2a: ||x|| >= |s|, s >= 0
        case 2b: ||x|| <= |s|, s <= 0

    """
    x_norm = jnp.linalg.norm(x)

    def case1_soc_proj(x, s):
        # case 1: y_norm >= |s|
        val = (s + x_norm) / (2 * x_norm)
        t = val * x_norm
        y = val * x
        return y, t

    def case2_soc_proj(x, s):
        # case 2: y_norm <= |s|
        # case 2a: s > 0
        def case2a(x, s):
            return x, s

        # case 2b: s < 0
        def case2b(x, s):
            return (0.0*jnp.zeros(x.size), 0.0)
        return lax.cond(s >= 0, case2a, case2b, x, s)
    return lax.cond(x_norm >= jnp.abs(s), case1_soc_proj, case2_soc_proj, x, s)


# provides vmapped versions of the projections for the soc and psd cones
soc_proj_single_batch = vmap(soc_proj_single, in_axes=(0), out_axes=(0))
sdp_proj_batch = vmap(sdp_proj_single, in_axes=(0, None), out_axes=(0))

"""
attempt to use jax.fori_loop for multiple soc projections of different sizes
not possible according to https://github.com/google/jax/issues/2962

def soc_body(i, val):
    socp, start = val

    # calculate the end point
    # end = start + soc_proj_sizes[i] * soc_num_proj[i]

    # extract the right soc_input
    # curr_soc_input = soc_input[start:end]
    curr_soc_input = lax.dynamic_slice(
        soc_input, (start,), (soc_proj_sizes[i] * soc_num_proj[i],))

    # reshape so that we vmap all of the socp projections of the same size together
    curr_soc_input_reshaped = jnp.reshape(
        curr_soc_input, (soc_num_proj[i], soc_proj_sizes[i]))
    curr_soc_out_reshaped = soc_proj_single_batch(curr_soc_input_reshaped)
    curr_socp = jnp.ravel(curr_soc_out_reshaped)

    # calculate the end point
    end = start + soc_proj_sizes[i] * soc_num_proj[i]

    # place in the correct location in the socp vector
    socp = socp.at[start:end].set(curr_socp)
    # socp = lax.dynamic_slice(
    #     soc_input, (start,), (soc_proj_sizes[i] * soc_num_proj[i],))

    # update the start point
    start = end

    new_val = socp, start
    return new_val

# val holds the vector and start point
start = 0
init_val = socp, start
val = lax.fori_loop(0, num_soc_blocks, soc_body, init_val)
socp, start = val
projection = jnp.concatenate([projection, socp])
"""
