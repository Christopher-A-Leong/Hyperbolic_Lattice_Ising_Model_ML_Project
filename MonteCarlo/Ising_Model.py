import numpy as np
from Fundamental.Number_Points import points

def random_spin_config(system_size):
    return(np.random.choice([-1,1], system_size))

def random_spin_config_dict(system_size, betalist):
    spin_config_dict = {}
    for beta in betalist:
        spin_config_dict['beta={}'.format(beta)] = random_spin_config(system_size)
    return(spin_config_dict)

def ising_energy_change(tight_bind_ham, trial_site, jval, spin_config):
    nearest_neighbors = np.where(tight_bind_ham[trial_site, :] != 0)[0]
    old_energy = -jval * np.sum(spin_config[trial_site] * spin_config[nearest_neighbors])
    new_energy = -jval * np.sum(-spin_config[trial_site] * spin_config[nearest_neighbors])
    return(new_energy - old_energy)

def magnetization_order_param(spin_config):
    magnetization = np.sum(spin_config)/len(spin_config)
    return(magnetization)

def magnetization_order_param_vectorized(spin_config_mat):
    magnetization = np.average(np.sum(spin_config_mat, axis=1)/np.size(spin_config_mat, 1))
    return(magnetization)

def magnetization_order_param_vectorized_noaverage(spin_config_mat):
    magnetization = np.sum(spin_config_mat, axis=1)/np.size(spin_config_mat, 1)
    return(magnetization)

def magnetization_order_param_hyperbolic_center(p, q, n, spin_config):
    points_per_level = points(p, q, n)[0]
    non_edge_spins = np.copy(spin_config[:int(np.sum(points_per_level[:-1]))])
    return(np.sum(non_edge_spins)/len(non_edge_spins))