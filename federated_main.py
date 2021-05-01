import random
import numpy as np
import torch, torchvision
import torch.nn as nn
import pandas as pd
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.optim as optim
from torch.utils.data.dataset import Dataset   
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score,f1_score, confusion_matrix
from torch.utils.data import DataLoader, TensorDataset
import model as lin_model
from sklearn.utils import resample,shuffle

######################## Training Functions ########################

def train_user(model, optimizer,criterion, data_loader,device, epoch):
    l = []
    training_score = []

    model.train()
    for i, (x_cpu, y_cpu) in enumerate(data_loader):
            x, y = x_cpu.to(device), y_cpu.to(device)
            outputs = model(x)
            loss = criterion(outputs, y.long())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # Calculating training accuracy
            _, predicted = torch.max(outputs.data, 1)

            y_original = y.cpu().data.squeeze().numpy()
            y_pred = predicted.cpu().data.squeeze().numpy()
            training_accuracy = (accuracy_score(y_pred, y_original))*100
  
            l.append(loss.item())
            training_score.append(training_accuracy)
    
    l = np.mean(l)
    training_accuracy = np.mean(training_score)

    print("Training Summary: Loss: %.3f" % (l), " Training Accuracy: %3f" % (training_accuracy))

    return l, training_accuracy


def central_server_update(centralized_model, user_models):

    ### This will take simple mean of the weights of models ###
      layer_weights = centralized_model.state_dict()
      for k in layer_weights.keys():
          layer_weights[k] = torch.stack([user_models[i].state_dict()[k].float() for i in range(len(user_models))], 0).mean(0)
      centralized_model.load_state_dict(layer_weights)
      for model in user_models:
          model.load_state_dict(centralized_model.state_dict())

######################## Validation Function ########################

def validation(model, data_loader,criterion,  device, epoch, test):

    model.eval()
    val_loss = 0
    all_y = []
    all_y_pred = []
    with torch.no_grad():
        for X_cpu, y_cpu in data_loader:
                X, y = X_cpu.to(device, dtype=torch.float), y_cpu.to(device, dtype=torch.float)
                
                y = torch.squeeze(y)
                output = model(X)
                loss = criterion(output, y.long())
                val_loss += loss.item()                 # sum up batch loss 

                _, y_predicted = torch.max(output, 1)

            # collect all y and y_pred in all batches
                all_y.extend(y)
                all_y_pred.extend(y_predicted)
    # to compute accuracy
    val_loss /= len(data_loader.dataset)
    all_y = torch.stack(all_y, dim=0)
    all_y_pred = torch.stack(all_y_pred, dim=0)
    test_score = accuracy_score(all_y.cpu().data.squeeze().numpy(), all_y_pred.cpu().data.squeeze().numpy())

    if not test: 
        print("[INFO] Epoch (%s) Validation Summary: "%(epoch), " Validation Accuracy: ", '%.1f' % (test_score*100))  
 

    elif test: 
        conf_m = confusion_matrix(all_y.cpu().data.squeeze().numpy(), all_y_pred.cpu().data.squeeze().numpy())
        print("[INFO] Test Summary: Test Accuracy: ", '%.1f' % (test_score*100))
        print("Confussion Matrix: ", conf_m)

        micro_fi = f1_score(all_y.cpu().data.squeeze().numpy(), all_y_pred.cpu().data.squeeze().numpy())
        print('Micro F1_score: ', micro_fi)

        roc = roc_auc_score(all_y.cpu().data.squeeze().numpy(),all_y_pred.cpu().data.squeeze().numpy())
        print("Area Under The Curve: ", roc)

    return val_loss, test_score*100

############################################## MAIN FUNCTION ##############################################

def main():

    num_classes = 2
    num_users = 2
    epochs = 100
    batch_size = 64

    print("[INFO] Starting ...")

    # Detect devices for GPU calculations
    use_cuda = torch.cuda.is_available()                   # check if GPU exists
    device = torch.device("cuda" if use_cuda else "cpu")   # use CPU or GPU

######################## Reading the data ########################

    df = pd.read_csv('creditcard.csv')
    print('This data frame has {} rows and {} columns.'.format(df.shape[0], df.shape[1]))
    
    # Downsampling the data
    df_down = resample(df[df['Class']==0], replace=False, n_samples = len(df[df['Class']==1]), random_state = 42)
    balanced_df = pd.concat([df[df['Class']==1],df_down])
    df = shuffle(balanced_df)


    X = df[['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
        'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
        'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount']]
    y = df['Class']


    # Dividing the training data into num_users, with each client having equal number of samples

    X_train_prev, X_val, y_train_prev, y_val = train_test_split(X, y, test_size=0.20, random_state=42)
    X_train, X_test , y_train, y_test = train_test_split(X_train_prev, y_train_prev, test_size=0.15, random_state=42)

    # Convert everything to Tensors
    tr_values = np.float32(X_train)  # X_train
    train_labels = np.float32(y_train)   # Y_train

    vl_values = np.float32(X_val) # Y_val
    val_labels = np.float32(y_val) # X_val

    test_values = np.float32(X_test) # Y_test
    test_labels = np.float32(y_test) # X_test

    X_train = torch.from_numpy(tr_values).float()
    y_train = torch.squeeze(torch.from_numpy(train_labels).float())

    X_val = torch.from_numpy(vl_values).float()
    y_val = torch.squeeze(torch.from_numpy(val_labels).float())

    X_test = torch.from_numpy(test_values).float()
    y_test = torch.squeeze(torch.from_numpy(test_labels).float())

    print("Training Set size: ", list(X_train.shape))
    print("Validation Set size: ", list(X_val.shape))
    print("Test Set size: ", list(X_test.shape))

    train_dataset = TensorDataset( X_train, y_train )
    val_dataset = TensorDataset(X_val, y_val )
    test_dataset = TensorDataset(X_test, y_test )

    # Function to divide the dataset into the number of clients

    def generate_index(dataset_lenght, clients): 
        res = []
        num = dataset_lenght
        indx = dataset_lenght // clients
        for c in range(clients + 1): 
            if num != 0 :  
                if num >= indx : 
                    res.append(indx)
                    num -= indx
                else: 
                    res.append(num)
        return res

    def weights_balanced(labels, nclasses):                        
        count = [0] * nclasses                                 
        for item in labels: 
            index = int(item[1].item())
            count[index] += 1
        weight_per_class = [0.] * nclasses                                      
        N = float(sum(count))  
        for i in range(nclasses):                                                   
            weight_per_class[i] = N/float(count[i])
        weight = [0] * len(labels)  
        for idx, val in enumerate(labels):      
            test = int(val[1].item())                              
            weight[idx] = weight_per_class[test] 
        return weight 

    # Creating the dataset for each of the users 
           
    traindata_split = torch.utils.data.random_split(train_dataset, generate_index(len(train_dataset),num_users))
    trainloader = []
    for x in traindata_split: 

            weights = weights_balanced(x, num_classes)     
            weights = torch.DoubleTensor(weights)  
            samp = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights),replacement=True)    
            params_train = {'batch_size': batch_size, 'shuffle': False, 'num_workers': 4, 'pin_memory': True,'sampler' : samp, "drop_last":True} if use_cuda else {}
            trainloader.append(DataLoader(x, **params_train))

    params_val =  {'batch_size': batch_size, 'shuffle': False, 'num_workers': 4, 'pin_memory': True, "drop_last":True} if use_cuda else {}    
    validloader = DataLoader(val_dataset, **params_val)
    testloader = DataLoader(test_dataset, **params_val)

    print("[INFO] Data is loaded. ")

   # Defining loss function, model, and optimizer for the centralized model and clients
   
    centralized_model = lin_model.LinNet().cuda()
    client_models = [lin_model.LinNet().cuda() for _ in range(num_users)]

    # Before the training starts, we coordinate all the models

    for model in client_models:
        model.load_state_dict(centralized_model.state_dict()) 

    model = lin_model.LinNet().cuda()
    optimizer = [optim.Adam(model.parameters(), lr=0.0001) for model in client_models]
    criterion =  nn.CrossEntropyLoss()

    # record training process

    epoch_train_loss = []
    epoch_train_acc = []
    epoch_test_loss = []
    epoch_test_acc = []

    for epoch in range(epochs):

        # select random clients

        client_idx = np.random.permutation(num_users)

        # client update

        for i in (range(num_users)):

            train_l, train_acc = train_user(client_models[i], optimizer[i], criterion, trainloader[client_idx[i]], device, epoch)
        
        # Centralized model update

        central_server_update(centralized_model, client_models)
        val_l, val_acc = validation(centralized_model, validloader, criterion, device, epoch, test = False)

        # save results

        epoch_train_loss.append(train_l)
        epoch_train_acc.append(train_acc)
        epoch_test_loss.append(val_l)
        epoch_test_acc.append(val_acc)

        # save all train test results

        A = np.array(epoch_train_loss)
        B = np.array(epoch_train_acc)
        C = np.array(epoch_test_loss)
        D = np.array(epoch_test_acc)

    validation(centralized_model, testloader,criterion,device, epoch, test = True )

    fig = plt.figure(figsize=(10, 4))
    plt.subplot(121)
    plt.plot(np.arange(1, epochs + 1), A)  # train loss (on epoch end)
    plt.plot(np.arange(1, epochs + 1), C)         #  test loss (on epoch end)
    plt.title("model loss")
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend(['train', 'val'], loc="upper left")
    # 2nd figure
    plt.subplot(122)
    plt.plot(np.arange(1, epochs + 1), B)  # train accuracy (on epoch end)
    plt.plot(np.arange(1, epochs + 1), D)         #  test accuracy (on epoch end)
    # plt.plot(histories.losses_val)
    plt.title("training scores")
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.legend(['train', 'val'], loc="upper left")
    title ="Federated_Accuracy_Scores.png"
    plt.savefig(title, dpi=600)
    # plt.close(fig)
    plt.show()

if __name__=="__main__": 
    main() 
