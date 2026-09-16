# Import dependencies
# Machine_Learning.FCNN is the FCNN_annotated code
from Machine_Learning.FCNN import *
from Fundamental.Number_Points import points
import pickle

# savedir=location where model and other stuff is saved to
savedir = 'INSERT DIRECTORY WHERE MODEL SHOULD BE SAVED HERE'

# datadir_list gives list of directories where data is imported from
datadir_list = ['DATA DIRECTORY 1', 'DATA DIRECTORY 2', 'AND SO FORTH']

# Define Schlafli symbol: {p,q} and nl=number of generations
p = 10
q = 3
nl = 3
# selected_beta_vals = beta values to import
selected_beta_vals = ['''Insert float values representing considered beta values here''']

# Define Critical temperature
tc = 0.9

# Import data using FCNN get_data function
alldata = get_data(nl, points(p, q, nl)[1], selected_beta_vals, tc, datadir_list)

# Train FCNN model using function from FCNN code
training_results = FCNN_Ising_model_singlelayer(alldata, savemodel=True, savedir=savedir)
history, test_loss, test_accuracy, confusion_matrix_result = training_results

# Define dictionary to hold non-model stuff to save, and save result using pickle
# Model is automatically saved to savedir by FCNN training code
analysis_save_stuff = {
    'history': history.history,
    'test_loss': test_loss,
    'test_accuracy': test_accuracy,
    'confusion_matrix': confusion_matrix_result
}

savefile = open(savedir + '\\end_analysis_data.pkl', 'wb')
pickle.dump(analysis_save_stuff, savefile)
savefile.close()
