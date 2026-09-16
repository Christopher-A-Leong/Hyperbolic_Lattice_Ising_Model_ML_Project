# Import dependencies
from MonteCarlo.Metropolis import metropolis_algorithm_save_spin_data_for_multiprocessing
import numpy as np
from Fundamental.Hamiltonian import H0

# Give directory to save results inside of
savedir = ''

# Define list of inverse temperatures to simulate at
betalist = 1 / np.linspace(0.1, 2, 20)

# Define hyperbolic lattice tiling to run at (q=3 or q=4 only)
pval = 10
qval = 3

# Define list of number of generations to run simulations for
nllist = [3, 4, 5]

# Iterate over each system size listed above
for nl in nllist:
    print('Working on nl={}'.format(nl))

    # Generate nearest-neighbor hopping tight binding Hamiltonian on hyperbolic lattice
    tight_bind_ham = H0(pval, qval, nl)

    # Iterate over all inverse temperatures specified above
    for beta in betalist:
        print('Working on beta={}'.format(beta))

        # Generate noise to apply to initial configuration field
        noise = np.random.choice(np.size(tight_bind_ham, 0), size=int(np.size(tight_bind_ham, 0)/10), replace=False)

        # Initiate perfect, homogenous initial configuration field and apply noise to it
        initial_spin_config_dict = {
            'beta={}'.format(beta): np.repeat(1, np.size(tight_bind_ham, 0))
        }
        initial_spin_config_dict['beta={}'.format(beta)][noise] = -1

        # Define model and simulation parameters
        num_iters = 10000000
        jval = 1
        num_bins = 5000
        mag_data_dict = {}
        final_spin_dict = {}

        # package all inputs
        dp = tight_bind_ham, beta, initial_spin_config_dict, num_iters, jval, num_bins, mag_data_dict, final_spin_dict

        # Run Monte Carlo simulation
        metropolis_algorithm_save_spin_data_for_multiprocessing(dp)

        # Save results
        np.save(savedir + '/nl{}beta{}_sampled_spin_configs'.format(nl, beta), mag_data_dict['beta={}'.format(beta)])
