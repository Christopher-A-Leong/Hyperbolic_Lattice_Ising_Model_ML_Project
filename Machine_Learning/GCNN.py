import torch
import torch_geometric
from torch_geometric.nn.conv import GCNConv
from torch_geometric.nn.pool import global_mean_pool, global_add_pool, global_max_pool
from Machine_Learning.FCNN import get_data
import numpy as np
from Fundamental.Hamiltonian import H0
from torcheval.metrics.functional import binary_accuracy
from Machine_Learning.FCNN import prepare_data_novalidation

def tight_bind_ham_to_connectivity(tbham):
    conns = np.where(tbham != 0)
    return(np.vstack((conns[0], conns[1])))

def spin_data_to_torchgeometric_data(p, q, nl, num_nodes, selected_beta_values, tc, datadir_list):
    data = get_data(nl, num_nodes, selected_beta_values, tc, datadir_list)
    training_features, training_labels, test_features, test_labels = prepare_data_novalidation(data)
    tbham = H0(p, q, nl)
    connectivity = tight_bind_ham_to_connectivity(tbham)
    training_features = training_features.astype(np.float32)
    test_features = test_features.astype(np.float32)
    training_labels = training_labels.astype(int)
    test_labels = test_labels.astype(int)
    dataset_train = []
    dataset_test = []
    for i in range(len(training_labels)):
        dataset_train.append(torch_geometric.data.Data(x=torch.from_numpy(np.transpose(training_features[i, :]).reshape(-1,1)),
                                                       edge_index=torch.from_numpy(connectivity),
                                                       y=torch.tensor([training_labels[i]])))
    for i in range(len(test_labels)):
        dataset_test.append(torch_geometric.data.Data(x=torch.from_numpy(np.transpose(test_features[i, :]).reshape(-1,1)),
                                                       edge_index=torch.from_numpy(connectivity),
                                                       y=torch.tensor([test_labels[i]])))
    return(dataset_train, dataset_test)

def minibatch_dataset(dataset_train, dataset_test):
    batched_dataset_train = torch_geometric.loader.DataLoader(dataset=dataset_train, shuffle=True, batch_size=256)
    batched_dataset_test = torch_geometric.loader.DataLoader(dataset=dataset_test, shuffle=True, batch_size=256)
    return(batched_dataset_train, batched_dataset_test)

def GCNN_model():
    class GCNN(torch.nn.Module):
        def __init__(self):
            super(GCNN, self).__init__()
            self.GCN1 = GCNConv(-1, 200)
            self.activ1 = torch.nn.ReLU()
            self.GCN2 = GCNConv(200, 200)
            self.activ2 = torch.nn.ReLU()
            self.GCN3 = GCNConv(200, 200)
            self.activ3 = torch.nn.ReLU()
            self.out = torch.nn.Linear(200, 2)
            self.activout = torch.nn.Softmax(dim=1)

        def forward(self, x, edge_index, batch=None):
            x = self.GCN1(x, edge_index)
            x = self.activ1(x)
            x = self.GCN2(x, edge_index)
            x = self.activ2(x)
            x = self.GCN3(x, edge_index)
            x = self.activ3(x)

            x = torch_geometric.nn.pool.global_mean_pool(x, batch)
            x = self.out(x)
            rawlogit_out = x
            x = self.activout(x)
            return(x, rawlogit_out)
    return(GCNN)

def train_GCNN(model, num_epochs, training_dataset, test_dataset, savemodel=False, savedir=''):
    model.train(mode=True)
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.NAdam(model.parameters(), lr=0.001)
    loss_values_over_epochs = np.array([])
    accuracy_values_over_epochs = np.array([])
    # val_loss_values_over_epochs = np.array([])
    # val_accuracy_values_over_epochs = np.array([])
    for i in range(num_epochs):
        batch_tested_counter = 0
        loss_values_over_batch = np.array([])
        accuracy_values_over_batch = np.array([])
        # val_loss_values_over_batch = np.array([])
        # val_accuracy_values_over_batch = np.array([])
        for data in training_dataset:
            print('Working on Epoch {}/{} | Progress={}%'.format(i+1, num_epochs,
                                                                 (batch_tested_counter / len(training_dataset))*100))
            optimizer.zero_grad()
            pred_out, raw_logit_out = model(data.x, data.edge_index, data.batch)
            pred_loss = loss_function(raw_logit_out, data.y)
            pred_accuracy = binary_accuracy(torch.argmax(pred_out, dim=1), data.y)
            # val_placeholder_loss = np.array([])
            # val_placeholder_accuracy = np.array([])
            # for val_data in validation_dataset:
            #     print('hi')
            #     val_placeholder_loss = np.append(val_placeholder_loss,
            #                                      loss_function(model(val_data.x, val_data.edge_index,
            #                                                          batch=val_data.batch)[1], val_data.y).item())
            #     val_placeholder_accuracy = np.append(val_placeholder_accuracy,
            #                                          binary_accuracy(torch.argmax(model(val_data.x,val_data.edge_index,
            #                                                            batch=val_data.batch)[0], dim=1), val_data.y).item())

            pred_loss.backward()
            optimizer.step()
            loss_values_over_batch = np.append(loss_values_over_batch, pred_loss.item())
            accuracy_values_over_batch = np.append(accuracy_values_over_batch, pred_accuracy.item())
            # val_loss_values_over_batch = np.append(val_loss_values_over_batch, np.average(val_placeholder_loss))
            # val_accuracy_values_over_batch = np.append(val_accuracy_values_over_batch, np.average(val_placeholder_accuracy))
            batch_tested_counter = batch_tested_counter + 1
        loss_values_over_epochs = np.append(loss_values_over_epochs, np.average(loss_values_over_batch))
        accuracy_values_over_epochs = np.append(accuracy_values_over_epochs, np.average(accuracy_values_over_batch))
        # val_loss_values_over_epochs = np.append(val_loss_values_over_epochs, np.average(loss_values_over_batch))
        # val_accuracy_values_over_epochs = np.append(val_accuracy_values_over_epochs, np.average(accuracy_values_over_batch))

    print('Evaluating over test data')
    test_loss = np.array([])
    test_accuracy = np.array([])
    for data in test_dataset:
        # print(loss_function(model(data.x, data.edge_index, data.batch)[1], data.y))
        # print(torch.argmax(binary_accuracy(model(data.x, data.edge_index, data.batch)[0], dim=1), data.y))
        test_loss = np.append(test_loss,
                              loss_function(model(data.x, data.edge_index, data.batch)[1], data.y).item())
        test_accuracy = np.append(test_accuracy,
                                  binary_accuracy(torch.argmax(model(data.x, data.edge_index, data.batch)[0], dim=1),
                                               data.y).item())

    print(np.average(test_loss))
    print(np.average(test_accuracy))
    if savemodel == True:
        np.save(savedir + '\\loss_values_over_epochs', loss_values_over_epochs)
        np.save(savedir + '\\accuracy_values_over_epochs', accuracy_values_over_epochs)
        np.save(savedir + '\\loss_values_test', test_loss)
        np.save(savedir + '\\accuracy_values_test', test_accuracy)
        # np.save(savedir + '\\val_loss_values_over_epochs', val_loss_values_over_epochs)
        # np.save(savedir + '\\val_accuracy_values_over_epochs', val_accuracy_values_over_epochs)
        torch.save(model.state_dict(), savedir + '\\trained_GCNN_model_state_dict.pt')

# def predict_GCNN(model):
#     model.eval()



