# Import dependencies
# Machine_Learning.GCNN is the GCNN_annotated code
# Points is custom code that gives: (number_sites_on_each_generation, total_number_of_sites)
from Fundamental.Number_Points import points
from Machine_Learning.GCNN import *

# savedir=location where model and other stuff is saved to
savedir = 'INSERT DIRECTORY WHERE MODEL SHOULD BE SAVED HERE'

# datadir_list gives list of directories where data is imported from
datadir_list = ['DATA DIRECTORY 1', 'DATA DIRECTORY 2', 'AND SO FORTH']

# Define Schlafli symbol: {p,q} and nl=number of generations
p = 10
q = 3
nl = 3

# points(p,q,nl)[1] gives number of sites in a {p,q} hyperbolic lattice with nl generations
num_features = points(p, q, nl)[1]

# selected_beta_vals = beta values to import
selected_beta_vals = ['''Insert float values representing considered beta values here''']

# Define Critical temperature
tc = 0.9

# Import data and split into training and test datasets using function spin_data_to_torchgeometric_data from GCNN code
all_data = spin_data_to_torchgeometric_data(p, q, nl, num_features, selected_beta_vals, tc, datadir_list)
dataset_train, dataset_test = all_data

# Batch data using minibatch_dataset function from GCNN code
batched_dataset_train, batched_dataset_test = minibatch_dataset(dataset_train, dataset_test)

# Define model
gcnn_model = GCNN_model()

# Train GCNN using train_GCNN model from GCNN code
train_GCNN(gcnn_model(), 5, batched_dataset_train, batched_dataset_test, savemodel=True, savedir=savedir)

