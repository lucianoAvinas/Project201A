import numpy as np
import experiment
import os

from preprocess import to_01
from scipy.optimize import minimize


def rbf_funcs(D, s2_total):
    D2 = D**2
    n = D.shape[0]

    def kern(theta1, theta2):
        sig = np.exp(theta1)
        ell = np.exp(theta2)

        return sig*np.exp(-D2/(2 * ell**2)) + (s2_total - sig)*np.eye(n)

    def kern_jac(theta1, theta2):
        sig = np.exp(theta1)
        ell = np.exp(theta2)

        jac_material = [None, None]
        jac_material[0] = np.exp(-D2/(2 * ell**2)) - np.eye(n)
        jac_material[1] = sig*np.exp(-D2/(2 * ell**2)) * D2/ell**2

        return jac_material

    return kern, kern_jac


def matern3_2_funcs(D, s2_total):
    D2 = D**2
    n = D.shape[0]

    def kern(theta1, theta2):
        sig = np.exp(theta1)
        ell = np.exp(theta2)

        return sig*(1+np.sqrt(3)*D/ell)*np.exp(-np.sqrt(3)*D/ell) + (s2_total - sig)*np.eye(n)

    def kern_jac(theta1, theta2):
        sig = np.exp(theta1)
        ell = np.exp(theta2)

        jac_material = [None, None]
        jac_material[0] = (1+np.sqrt(3)*D/ell)*np.exp(-np.sqrt(3)*D/ell) - np.eye(n)
        jac_material[1] = sig*3*D2/ell**2 * np.exp(-np.sqrt(3)*D/ell)

        return jac_material

    return kern, kern_jac


def matern5_2_funcs(D, s2_total):
    D2 = D**2
    n = D.shape[0]

    def kern(theta1, theta2):
        sig = np.exp(theta1)
        ell = np.exp(theta2)

        return sig*(1+np.sqrt(5)*D/ell+5*D2/(3*ell**2))*np.exp(-np.sqrt(5)*D/ell) + \
               (s2_total - sig)*np.eye(n)

    def kern_jac(theta1, theta2):
        sig = np.exp(theta1)
        ell = np.exp(theta2)

        jac_material = [None, None]
        jac_material[0] = (1+np.sqrt(5)*D/ell+5*D2/(3*ell**2))*np.exp(-np.sqrt(5)*D/ell) - np.eye(n)
        jac_material[1] = sig*5*D2/(3*ell**2)*(1+D*np.sqrt(5)/ell)*np.exp(-np.sqrt(5)*D/ell)

        return jac_material

    return kern, kern_jac


def optim_lenscale(kern_type):
    # length scale l = exp(theta) for ease of optimization
    s2_total = 0.00239413

    design, data = experiment.setup_data('train')
    unif_arr = to_01(design)
    y = np.load(os.path.join('experiment_data', 'response.npy'))

    D = np.linalg.norm(unif_arr[None,:]-unif_arr[:,None], axis=2)
    n = D.shape[0]

    if kern_type == 'rbf':
        kern, kern_jac = rbf_funcs(D, s2_total)
    elif kern_type == 'matern3_2':
        kern, kern_jac = matern3_2_funcs(D, s2_total)
    elif kern_type == 'matern5_2':
        kern, kern_jac = matern5_2_funcs(D, s2_total)
    else:
        raise Exception(f'{kern_type} kernel not supported')

    def nll(theta):
        theta1, theta2 = theta
        K = kern(theta1, theta2)
        K_inv = np.linalg.solve(K, np.eye(n))

        return np.dot(y, K_inv.dot(y)) + np.linalg.slogdet(K)

    def d_nll(theta):
        theta1, theta2 = theta
        K_inv = np.linalg.solve(kern(theta1, theta2), np.eye(n))
        alpha = K_inv.dot(y)
        mat1 = np.outer(alpha, alpha) - K_inv

        jacs = np.zeros(2)
        jac_material = kern_jac(theta1, theta2)
        jacs[0] = -np.einsum('ij,ji->', mat1, jac_material[0]).sum()
        jacs[1] = -np.einsum('ij,ji->', mat1, jac_material[1]) 
        return jacs

    return minimize(nll, np.array([-7, 2.2]), jac=d_nll, 
                    bounds=[(None, np.log(s2_total)), (None, None)])
