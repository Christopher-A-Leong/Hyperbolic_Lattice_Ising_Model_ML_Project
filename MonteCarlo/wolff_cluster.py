import numpy as np
import multiprocessing as mp
import pickle

def get_frontier_expansion_routes(tight_bind_ham, spin_config, considered_cluster_site):
    nearest_neighbors = np.where(tight_bind_ham[considered_cluster_site, :] != 0)[0]
    same_spin = (spin_config[nearest_neighbors] == spin_config[considered_cluster_site])
    return(nearest_neighbors[same_spin])

def wolff_cluster_algorithm_for_multiprocessing(data_package):
    tight_bind_ham, spin_config_dict, jval, beta, num_iters, num_bins, mag_data_dict, final_spin_dict = data_package
    spin_config = np.copy(spin_config_dict['beta={}'.format(beta)])

    wolff_special_prob = 1 - np.exp(-beta*2*jval)
    bin_width = num_iters / num_bins
    mag_data = np.zeros((num_bins, len(spin_config)))
    inbetween_bins_counter = 0
    bin_filled_counter = 0
    for ni in range(num_iters):
        if ni%10000 == 0:
            print(ni)
        cluster_sites = np.array([], dtype=int)
        random_start = np.random.randint(0,len(spin_config))
        cluster_sites = np.append(cluster_sites, random_start)
        frontier_untested_neighbors = get_frontier_expansion_routes(tight_bind_ham, spin_config, random_start)
        while len(frontier_untested_neighbors) != 0:
            new_frontier = np.array([], dtype=int)
            already_rolled = np.array([], dtype=int)
            for fun in frontier_untested_neighbors:
                roll_dice = np.random.uniform(0,1)
                if (roll_dice <= wolff_special_prob) and (fun not in already_rolled):
                    cluster_sites = np.append(cluster_sites, fun)
                    cluster_sites = np.unique(cluster_sites)
                    new_frontier = np.append(new_frontier, get_frontier_expansion_routes(tight_bind_ham, spin_config, fun))
                    repeated_old_frontier = np.intersect1d(cluster_sites, new_frontier)
                    for rof in repeated_old_frontier:
                        new_frontier = np.delete(new_frontier, np.where(new_frontier == rof)[0])
                    already_rolled = np.append(already_rolled, fun)
                else:
                    pass
            frontier_untested_neighbors = np.copy(new_frontier)
        spin_config[cluster_sites] = -spin_config[cluster_sites]

        inbetween_bins_counter = inbetween_bins_counter + 1
        if inbetween_bins_counter == bin_width:
            mag_data[bin_filled_counter, :] = np.copy(spin_config)
            inbetween_bins_counter = 0
            bin_filled_counter = bin_filled_counter + 1

    mag_data_dict['beta={}'.format(beta)] = mag_data
    final_spin_dict['beta{}'.format(beta)] = spin_config

def wolff_cluster_mc_sweep(tight_bind_ham, spin_config_dict, jval, betalist, num_iters, num_bins, savedir, savenamepreface):
    manager = mp.Manager()
    mag_data_dict = manager.dict()
    final_spin_dict = manager.dict()

    data_package = ()
    for beta in betalist:
        data_package = data_package + ([tight_bind_ham, spin_config_dict, jval, beta, num_iters, num_bins,
                                        mag_data_dict, final_spin_dict],)

    def mp_process_run(num_betas, data_package):
        p = mp.Pool(num_betas)
        p.map(wolff_cluster_algorithm_for_multiprocessing, data_package)

    mp_process_run(len(betalist), data_package)

    mag_savefile = open(savedir + '\\' + savenamepreface + '_magnetization_data_dict.pkl', 'wb')
    pickle.dump(mag_data_dict.copy(), mag_savefile)
    mag_savefile.close()

    fs_savefile = open(savedir + '\\' + savenamepreface + '_final_spin_config_dict.pkl', 'wb')
    pickle.dump(final_spin_dict.copy(), fs_savefile)
    fs_savefile.close()

if __name__=='__main__':
    pass
