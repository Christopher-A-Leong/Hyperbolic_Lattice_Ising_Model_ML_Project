import numpy as np
from MonteCarlo.Ising_Model import ising_energy_change
import multiprocessing as mp
import pickle
from MonteCarlo.Ising_Model import magnetization_order_param, magnetization_order_param_hyperbolic_center
from Fundamental.Number_Points import points

def metropolis_algorithm_save_magnetization_data_for_multiprocessing(data_package):
    tight_bind_ham, beta, spin_config_dict, num_iters, jval, num_bins, mag_data_dict, final_spin_dict = data_package
    # tight_bind_ham, beta, spin_config_dict, num_iters, jval, mag_data_dict, final_spin_dict = data_package

    spin_config = np.copy(spin_config_dict['beta={}'.format(beta)])
    num_sites = len(spin_config)
    num_iters_between_binning = int(num_iters/num_bins)
    magdata = np.zeros(num_bins)
    between_binning_values = np.zeros(num_iters_between_binning)
    num_between_steps = 0
    num_bins_filled = 0
    for ni in range(num_iters):
        trial_site = np.random.randint(num_sites)
        # print(trial_site)
        energy_change = ising_energy_change(tight_bind_ham, trial_site, jval, spin_config)
        if energy_change <= 0:
            spin_config[trial_site] = -1*spin_config[trial_site]
        else:
            accept_prob = np.exp(-beta * energy_change)
            accept_roll = np.random.uniform(0, 1)
            if accept_roll <= accept_prob:
                spin_config[trial_site] = -1*spin_config[trial_site]
            else:
                pass

        # num_between_steps = num_between_steps + 1
        # if num_between_steps == num_iters_between_binning:
        #     magdata[num_bins_filled] = magnetization_order_param(spin_config)
        #     num_bins_filled = num_bins_filled + 1
        #     num_between_steps = 0

        between_binning_values[num_between_steps] = magnetization_order_param(spin_config)
        num_between_steps = num_between_steps + 1

        if num_between_steps == num_iters_between_binning:
            magdata[num_bins_filled] = np.average(between_binning_values**2)
            num_bins_filled = num_bins_filled + 1
            num_between_steps = 0
            between_binning_values = np.zeros(num_iters_between_binning)

    mag_data_dict['beta={}'.format(beta)] = magdata
    final_spin_dict['beta={}'.format(beta)] = spin_config

def metropolis_mc_sweep(tight_bind_ham, betalist, spin_config_dict, num_iters, jval, num_bins, savedir, savenamepreface):
    manager = mp.Manager()
    mag_data_dict = manager.dict()
    final_spin_dict = manager.dict()

    data_package = ()
    for beta in betalist:
        data_package = data_package + ([tight_bind_ham, beta, spin_config_dict, num_iters, jval, num_bins,
                                        mag_data_dict, final_spin_dict],)

    # data_package = ()
    # for beta in betalist:
    #     data_package = data_package + ([tight_bind_ham, beta, spin_config_dict, num_iters, jval,
    #                                     mag_data_dict, final_spin_dict],)

    def mp_process_run(num_betas, data_package):
        p = mp.Pool(num_betas)
        p.map(metropolis_algorithm_save_magnetization_data_for_multiprocessing, data_package)

    mp_process_run(len(betalist), data_package)

    mag_savefile = open(savedir + '\\' + savenamepreface + '_magnetization_data_dict.pkl', 'wb')
    pickle.dump(mag_data_dict.copy(), mag_savefile)
    mag_savefile.close()

    fs_savefile = open(savedir + '\\' + savenamepreface + '_final_spin_config_dict.pkl', 'wb')
    pickle.dump(final_spin_dict.copy(), fs_savefile)
    fs_savefile.close()

############

# def non_uniform_trial_site_sample_edge_bulk(pbulk_c, pbulk_nc, pedge, bulkindices_c, bulkindices_nc, edgeindices):
#     tot_num_sites = len(bulkindices_c) + len(bulkindices_nc) + len(edgeindices)
#     probability_distribution = np.zeros(tot_num_sites)
#     probability_distribution[bulkindices_c] = pbulk_c
#     probability_distribution[bulkindices_nc] = pbulk_nc
#     probability_distribution[edgeindices] = pedge
#     return(np.random.choice(tot_num_sites, p=probability_distribution))

def non_uniform_trial_site_sample_edge_bulk(p, q, num_levels):
    points_per_level, totnumsites = points(p, q, num_levels)
    probab_dist = np.array([])
    for n in range(num_levels):
        weight = totnumsites / points_per_level[n]
        probab_dist = np.append(probab_dist, np.repeat(weight, points_per_level[n]))
    probab_dist = probab_dist / np.sum(probab_dist)
    return(np.random.choice(int(totnumsites), p=probab_dist))



def metropolis_algorithm_save_hyperbolic_bulk_magnetization_data_for_multiprocessing(data_package):
    p, q, num_levels, tight_bind_ham, beta, spin_config_dict, num_iters, jval, num_bins, mag_data_dict, final_spin_dict = data_package
    # tight_bind_ham, beta, spin_config_dict, num_iters, jval, mag_data_dict, final_spin_dict = data_package

    # points_per_level, totnumsites = points(p, q, num_levels)
    # bulkindices_center = np.arange(points_per_level[0])
    # bulkindices_notcenter = np.arange(points_per_level[0], int(np.sum(points_per_level[:-1])))
    # edgeindices = np.arange(int(np.sum(points_per_level[:-1])), int(totnumsites))
    # edgebulkratio = len(edgeindices)/(len(bulkindices_notcenter) + len(bulkindices_center))
    # pedge = 1/totnumsites/edgebulkratio
    # pbulk_nc = (1/totnumsites)*edgebulkratio
    # bulknccratio = len(bulkindices_notcenter)/len(bulkindices_center)
    # pbulk_c = pbulk_nc*bulknccratio
    # pbulk_nc = pbulk_nc/bulknccratio

    spin_config = np.copy(spin_config_dict['beta={}'.format(beta)])
    num_sites = len(spin_config)
    num_iters_between_binning = int(num_iters/num_bins)
    magdata = np.zeros(num_bins)
    # between_binning_values = np.zeros(num_iters_between_binning)
    num_between_steps = 0
    num_bins_filled = 0
    for ni in range(num_iters):
        trial_site = non_uniform_trial_site_sample_edge_bulk(p, q, num_levels)
        # trial_site = non_uniform_trial_site_sample_edge_bulk(pbulk_c=pbulk_c, pbulk_nc=pbulk_nc, pedge=pedge,
        #                                                      bulkindices_c=bulkindices_center,
        #                                                      bulkindices_nc=bulkindices_notcenter, edgeindices=edgeindices)
        energy_change = ising_energy_change(tight_bind_ham, trial_site, jval, spin_config)
        if energy_change <= 0:
            spin_config[trial_site] = -1*spin_config[trial_site]
        else:
            accept_prob = np.exp(-beta * energy_change)
            accept_roll = np.random.uniform(0, 1)
            if accept_roll <= accept_prob:
                spin_config[trial_site] = -1*spin_config[trial_site]
            else:
                pass

        num_between_steps = num_between_steps + 1

        if num_between_steps == num_iters_between_binning:
            magdata[num_bins_filled] = magnetization_order_param_hyperbolic_center(p, q, num_levels, spin_config)
            num_bins_filled = num_bins_filled + 1
            num_between_steps = 0

        # between_binning_values[num_between_steps] = magnetization_order_param(spin_config)
        # num_between_steps = num_between_steps + 1
        #
        # if num_between_steps == num_iters_between_binning:
        #     magdata[num_bins_filled] = np.average(between_binning_values)
        #     num_bins_filled = num_bins_filled + 1
        #     num_between_steps = 0
        #     between_binning_values = np.zeros(num_iters_between_binning)

    mag_data_dict['beta={}'.format(beta)] = magdata
    final_spin_dict['beta={}'.format(beta)] = spin_config

def metropolis_hyperbolic_bulk_mc_sweep(p, q, num_levels, tight_bind_ham, betalist, spin_config_dict, num_iters, jval, num_bins, savedir, savenamepreface):
    manager = mp.Manager()
    mag_data_dict = manager.dict()
    final_spin_dict = manager.dict()

    data_package = ()
    for beta in betalist:
        data_package = data_package + ([p, q, num_levels, tight_bind_ham, beta, spin_config_dict, num_iters, jval,
                                        num_bins, mag_data_dict, final_spin_dict],)

    # data_package = ()
    # for beta in betalist:
    #     data_package = data_package + ([tight_bind_ham, beta, spin_config_dict, num_iters, jval,
    #                                     mag_data_dict, final_spin_dict],)

    def mp_process_run(num_betas, data_package):
        p = mp.Pool(num_betas)
        p.map(metropolis_algorithm_save_hyperbolic_bulk_magnetization_data_for_multiprocessing, data_package)

    mp_process_run(len(betalist), data_package)

    mag_savefile = open(savedir + '\\' + savenamepreface + '_magnetization_data_dict.pkl', 'wb')
    pickle.dump(mag_data_dict.copy(), mag_savefile)
    mag_savefile.close()

    fs_savefile = open(savedir + '\\' + savenamepreface + '_final_spin_config_dict.pkl', 'wb')
    pickle.dump(final_spin_dict.copy(), fs_savefile)
    fs_savefile.close()

#######

def metropolis_algorithm_save_spin_data_for_multiprocessing(data_package):
    tight_bind_ham, beta, spin_config_dict, num_iters, jval, num_bins, mag_data_dict, final_spin_dict = data_package

    spin_config = np.copy(spin_config_dict['beta={}'.format(beta)])
    num_sites = len(spin_config)
    num_iters_between_binning = int(num_iters/num_bins)
    magdata = np.zeros((num_bins, num_sites))
    num_between_steps = 0
    num_bins_filled = 0
    for ni in range(num_iters):
        trial_site = np.random.randint(num_sites)
        # print(trial_site)
        energy_change = ising_energy_change(tight_bind_ham, trial_site, jval, spin_config)
        if energy_change <= 0:
            spin_config[trial_site] = -1*spin_config[trial_site]
        else:
            accept_prob = np.exp(-beta * energy_change)
            accept_roll = np.random.uniform(0, 1)
            if accept_roll <= accept_prob:
                spin_config[trial_site] = -1*spin_config[trial_site]
            else:
                pass

        num_between_steps = num_between_steps + 1
        if num_between_steps == num_iters_between_binning:
            magdata[num_bins_filled, :] = np.copy(spin_config)
            num_bins_filled = num_bins_filled + 1
            num_between_steps = 0

    mag_data_dict['beta={}'.format(beta)] = magdata
    final_spin_dict['beta={}'.format(beta)] = spin_config

if __name__=='__main__':
    pass
