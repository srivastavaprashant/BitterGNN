import pandas as pd
import numpy as np

import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder


def read_data(data_dir):
    """
    Read data from the specified directory.

    Args:
        data_dir (str): The directory path where the data files are located.

    Returns:
        pandas.DataFrame: The concatenated dataframe containing the data from all files.
    """
    train_pos = pd.read_csv(data_dir + "train-positive.txt", header=None)
    train_neg = pd.read_csv(data_dir + "train-negative.txt", header=None)
    test_pos = pd.read_csv(data_dir + "test-positive.txt", header=None)
    test_neg = pd.read_csv(data_dir + "test-negative.txt", header=None)

    train_pos = pd.DataFrame(
        [
            train_pos[::2].reset_index(drop=True)[0],
            train_pos[1::2].reset_index(drop=True)[0],
        ]
    ).T
    train_neg = pd.DataFrame(
        [
            train_neg[::2].reset_index(drop=True)[0],
            train_neg[1::2].reset_index(drop=True)[0],
        ]
    ).T
    test_pos = pd.DataFrame(
        [
            test_pos[::2].reset_index(drop=True)[0],
            test_pos[1::2].reset_index(drop=True)[0],
        ]
    ).T
    test_neg = pd.DataFrame(
        [
            test_neg[::2].reset_index(drop=True)[0],
            test_neg[1::2].reset_index(drop=True)[0],
        ]
    ).T

    df = pd.concat([train_pos, train_neg, test_pos, test_neg])
    df = df.sample(frac=1, random_state=911)
    return df


def create_graph_data(str, label, enc):
    """
    Create a graph data object for the given string, label, and encoder.

    Args:
        str (str): The input string.
        label (int): The label for the input string.
        enc (Encoder): The encoder object used to transform the string.

    Returns:
        Data: The graph data object containing node features, edge indices, and labels.
    """
    n = len(str)
    edge_index_up = [[i, i + 1] for i in range(n - 1)]
    edge_index_down = [[i + 1, i] for i in range(n - 1)]
    edge_index = torch.tensor(edge_index_down + edge_index_up, dtype=torch.long)

    node_feats = enc.transform([[i] for i in str]).toarray()
    node_features = torch.tensor(node_feats, dtype=torch.float)
    label = torch.tensor(label)
    d = Data(x=node_features, edge_index=edge_index.t().contiguous(), y=label)

    return d


def preprocess(df):
    """
    Preprocesses the input dataframe to create protein graphs.

    Args:
        df (pandas.DataFrame): The input dataframe containing 'bitter' and 'seq' columns.

    Returns:
        list: A list of protein graphs.

    """
    df.iloc[:, 0] = df.iloc[:, 0].apply(lambda x: x.split(" ")[0][1:])
    df.columns = ["bitter", "seq"]
    labels = df.bitter.apply(lambda x: 1 if x == "Positive" else 0).tolist()
    seqs = df.seq.tolist()
    n = len(labels)
    descriptors = list(set([j for i in seqs for j in i]))
    enc = OneHotEncoder()
    X = np.array(descriptors).reshape((-1, 1))
    enc_arrays = enc.fit(X)

    protein_graphs = []
    for i in range(n):
        d = create_graph_data(seqs[i], labels[i], enc)
        protein_graphs.append(d)

    return protein_graphs


def train(train_loader, model, optimizer, criterion):
    model.train()

    for data in train_loader:  # Iterate in batches over the training dataset.
        out = model(
            data.x, data.edge_index, data.batch
        )  # Perform a single forward pass.
        loss = criterion(out, data.y)  # Compute the loss.
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.

        optimizer.zero_grad()  # Clear gradients.
    model.eval()

    correct = 0
    for data in train_loader:  # Iterate in batches over the training/test dataset.
        out = model(data.x, data.edge_index, data.batch)
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        correct += int((pred == data.y).sum())  # Check against ground-truth labels.
        accuracy = correct / len(train_loader.dataset)
    return accuracy, model


def test(loader, model):
    """
    Evaluate the performance of the model on the given data loader.

    Args:
        loader (torch.utils.data.DataLoader): The data loader containing the test dataset.
        model (torch.nn.Module): The model to be evaluated.

    Returns:
        tuple: A tuple containing the accuracy and ROC AUC score of the model.
    """
    model.eval()

    correct = 0
    for data in loader:  # Iterate in batches over the training/test dataset.
        out = model(data.x, data.edge_index, data.batch)
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        correct += int((pred == data.y).sum())  # Check against ground-truth labels.
        accuracy = correct / len(loader.dataset)
        roc_auc = roc_auc_score(data.y.detach(), out.detach().numpy()[:, 1])
    return (
        np.round(accuracy, 2),
        np.round(roc_auc, 2),
    )  # Derive ratio of correct predictions.


def run_kfold_test(nsplits, graph_data, MODEL_INST, h=16, lr=0.005, b=32):
    """
    Run k-fold cross-validation on the given graph data using the specified model.

    Args:
        nsplits (int): The number of splits for cross-validation.
        graph_data (list): A list of graph data objects.
        MODEL_INST (torch.nn.Module): The model class to be instantiated.
        h (int, optional): The number of hidden channels in the model. Defaults to 16.
        lr (float, optional): The learning rate for the optimizer. Defaults to 0.005.
        b (int, optional): The batch size for training. Defaults to 32.

    Returns:
        tuple: A tuple containing the list of test accuracies and the list of ROC AUC scores for each fold.
    """

    n_data = len(graph_data)
    fold_test_roc = []
    fold_test_acc = []

    for i in range(nsplits):
        test_indexes = list(
            range(int(n_data * (i) / nsplits), int(n_data * (i + 1) / nsplits))
        )
        train_indexes = [i for i in range(n_data) if i not in test_indexes]

        test_data = [graph_data[i] for i in test_indexes]
        train_data = [graph_data[i] for i in train_indexes]

        train_loader = DataLoader(train_data, batch_size=b, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=32, shuffle=True)

        model = MODEL_INST(hidden_channels=h)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = torch.nn.CrossEntropyLoss()

        test_acc_list = []
        epochs_list = []

        for epoch in range(1, 30):
            train_acc, model = train(train_loader, model, optimizer, criterion)
            test_acc, test_roc = test(test_loader, model)
            epochs_list.append(epoch)
            # print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')

        print(f"Fold {i+1}:")
        print("Test Acc:", test_acc, "ROC:", test_roc)

        # fold_train_acc.append(train_acc)
        fold_test_acc.append(test_acc)
        fold_test_roc.append(test_roc)
    return fold_test_acc, fold_test_roc
