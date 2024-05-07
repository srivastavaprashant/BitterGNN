import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools
from seaborn import color_palette, scatterplot
from scikitplot.metrics import plot_roc
from matplotlib import gridspec


from torch import tensor, long, float, unique
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from torch.optim import Adam
from torch.nn import CrossEntropyLoss

from sklearn.decomposition import PCA
from scripts.settings import DATA_DIR, AMINO_ACIDS
import umap


def read_data(data_dir=DATA_DIR):
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
    df.iloc[:, 0] = df.iloc[:, 0].apply(lambda x: x.split(" ")[0][1:])
    df.columns = ["bitter", "seq"]
    df.bitter = df.bitter.apply(lambda x: 1 if x == "Positive" else 0).values
    return df


def preprocess(df):
    def create_graph_data(str, label, enc):
        """Create input graph data for Pytorch Geometric model."""
        n = len(str)  # length of peptide
        # each peptide graph has n nodes and 2*(n-1) edges (bidirectional- up and down)
        edge_index_up = [[i, i + 1] for i in range(n - 1)]
        edge_index_down = [[i + 1, i] for i in range(n - 1)]
        edge_index = tensor(edge_index_down + edge_index_up, dtype=long)

        node_feats = enc.transform([[i] for i in str]).toarray()
        node_features = tensor(node_feats, dtype=float)
        label = tensor(label)
        d = Data(x=node_features, edge_index=edge_index.t().contiguous(), y=label)

        return d

    seqs = df.seq.tolist()
    labels = df.bitter.tolist()
    n = len(df)
    descriptors = list(set([j for i in seqs for j in i]))
    encoder = OneHotEncoder()
    X = np.array(descriptors).reshape((-1, 1))
    enc_arrays = encoder.fit(X)

    protein_graphs = []
    for i in range(n):
        d = create_graph_data(seqs[i], labels[i], encoder)
        protein_graphs.append(d)

    return protein_graphs, encoder


def get_graph_embeddings(model, data_loader, encoder):
    activation = {}

    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()

        return hook

    graph_embeddings = []
    for data in data_loader:
        out = model(data.x, data.edge_index, data.batch)
        lab = data.y

        model.conv3.register_forward_hook(get_activation("conv3"))
        output = model(data.x, data.edge_index, data.batch)
        conv3_feats = activation["conv3"]

        for i in unique(data.batch):
            node_indexes = (data.batch == i).nonzero().ravel()

            original_enc = data.x[node_indexes]

            desc = "".join(list(encoder.inverse_transform(original_enc).ravel()))

            d = {
                "desc": desc,
                "n_nodes": len(node_indexes),
                "label": lab[i].tolist(),
                "pred": np.argmax(out[i].detach().numpy()),
                "conv3_feats": conv3_feats[node_indexes].numpy(),
            }
            graph_embeddings.append(d)

    for k in range(len(graph_embeddings)):
        graph_embeddings[k]["avg_desc_act"] = graph_embeddings[k]["conv3_feats"].mean(
            axis=1
        )
    return graph_embeddings


def pepvecs_2d_plot(all_graph_embeddings, method=any(["tsne", "umap", "pca"])):
    def _tsne(conv_feats):
        tsne = TSNE(n_components=2, random_state=911)
        tsne_embd = tsne.fit_transform(conv_feats)
        print("KL Div:", tsne.kl_divergence_)
        return tsne_embd

    def _umap(conv_feats):
        umap_embd = umap.UMAP(n_components=2, random_state=911).fit_transform(
            conv_feats
        )
        return umap_embd

    def _pca(conv_feats):
        pca = PCA(n_components=2, random_state=911)
        pca_embd = pca.fit_transform(conv_feats)
        return pca_embd

    descs = []
    conv_feats = np.zeros((len(all_graph_embeddings), 16))
    labels = []
    pred_vals = []
    for i in range(len(all_graph_embeddings)):
        descs.append(all_graph_embeddings[i]["desc"])
        conv_feats[i] = all_graph_embeddings[i]["conv3_feats"].mean(axis=0)
        labels.append(all_graph_embeddings[i]["label"])
        pred_vals.append(all_graph_embeddings[i]["pred"])

    if method == "tsne":
        embd = _tsne(conv_feats)
    elif method == "umap":
        embd = _umap(conv_feats)
    elif method == "pca":
        embd = _pca(conv_feats)

    fig = plt.figure(figsize=(8, 8))
    plt.scatter(
        embd[:, 0],
        embd[:, 1],
        c=[color_palette()[x + 2] for x in labels],
        s=3,
        label="Bitter",
    )
    plt.legend()
    plt.title(f"{method.upper()} plot of all peptide graph embeddings")
    plt.xticks([])
    plt.yticks([])
    plt.legend()
    plt.tight_layout()
    plt.close(fig)
    return embd, descs, labels, fig


def train(model, train_loader, criterion, optimizer):
    model.train()

    for data in train_loader:  # Iterate in batches over the training dataset.
        out = model(
            data.x, data.edge_index, data.batch
        )  # Perform a single forward pass.
        loss = criterion(out, data.y)  # Compute the loss.
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad()  # Clear gradients.


def test(model, loader):
    model.eval()

    correct = 0
    for data in loader:  # Iterate in batches over the training/test dataset.
        out = model(data.x, data.edge_index, data.batch)  # Predict the labels.
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        correct += int((pred == data.y).sum())  # Check against ground-truth labels.
    return correct / len(loader.dataset)  # Derive ratio of correct predictions.


def predict(dataloader, model):
    out = []
    lab = []
    for data in dataloader:
        o = model(data.x, data.edge_index, data.batch)
        l = data.y
        out = out + o.detach().numpy().tolist()
        lab = lab + l.detach().numpy().tolist()

    return lab, out


def get_clusters(
    embd, descs, all_graph_embeddings, method="kmeans", method_args={}, embd_type="2d"
):
    embd = np.array(embd)
    if method == "kmeans":
        clustering = KMeans(**method_args)
        # Run the fit
        clustering.fit(embd)

    else:
        raise ValueError("Invalid clustering method")

    # Get the results
    clusters = pd.DataFrame([descs, clustering.labels_]).T
    clusters.columns = ["desc", "Cluster"]
    emb = pd.DataFrame(all_graph_embeddings)
    clusters = emb.merge(clusters, on="desc")
    clusters["acc"] = 1 * (clusters.pred == clusters.label)
    clusters["kmeansbitter"] = 2 * (clusters.Cluster) + clusters.label + 1
    clusters["Cluster"] = clusters.Cluster.replace(
        clusters.groupby("Cluster").label.mean().rank().to_dict()
    )
    clusters["ClusterName"] = clusters.Cluster.apply(lambda x: f"Cluster-{x:1.0f}")

    if embd.shape[1] == 2:
        fig = plt.figure(constrained_layout=True, figsize=(12, 8))
        gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1], figure=fig)
        ax0 = plt.subplot(gs[0])
        ax1 = plt.subplot(gs[1])
        scatterplot(
            x=embd[:, 0], y=embd[:, 1], hue=clusters["ClusterName"], ax=ax0,
        )
        fig.suptitle(
            f"{method.lower()} clusters plot for {embd_type} peptide embeddings"
        )
    else:
        fig = plt.figure(figsize=(4, 8))
        gs = gridspec.GridSpec(1, 1, width_ratios=[1], figure=fig)
        ax1 = plt.subplot(gs[0])
        fig.suptitle(
            f"{method.lower()} clusters plot for {embd_type} peptide embeddings"
        )

    unique_clusters = clusters.Cluster.unique()
    unique_clusters.sort()
    aminoacid_pop_in_clusters = (
        pd.DataFrame(
            [
                clusters[clusters["Cluster"] == i]
                .desc.apply(lambda x: [j for j in x])
                .explode()
                .value_counts(normalize=True)
                .rename(f"Cluster-{i}")
                .round(2)
                for i in unique_clusters
            ]
        )
        .T.fillna(0)
        .sort_index()
    )
    aminoacid_pop_in_clusters.plot(
        kind="barh", width=0.8, ax=ax1,
    )
    ax1.set_xlabel("Percent population of AminoAcids")
    ax1.set_ylabel("Amino Acids")
    plt.close(fig)
    return clusters, clustering, aminoacid_pop_in_clusters, fig


def train_model(model, protien_graphs, optimizer, criterion, epochs=10):
    train_data, test_data = train_test_split(
        protien_graphs, test_size=1, random_state=6
    )
    data_loader = DataLoader(protien_graphs, batch_size=64)
    train_loader = DataLoader(train_data, batch_size=64)
    test_loader = DataLoader(test_data, batch_size=64)
    for epoch in range(epochs):
        train(model, train_loader, criterion, optimizer)
        acc = test(model, train_loader)
        print(f"Epoch: {epoch:03d}, Accuracy: {acc:.4f}")

    labels, output = predict(train_loader, model)
    fig, ax = plt.subplots(figsize=(6, 6))
    plot_roc(labels, output, ax=ax)
    plt.close(fig)
    return model, train_loader, test_loader, fig


def get_subseq_activations(all_graph_embeddings, l=1):
    """Get the activations of substructures within the peptides.
    For example: In a list of peptides [PFA, VA], the substructures are:
        l = 1: V, P, F, A
        l = 2: VA, PF, FA
        l = 3: PFA
        l = 4: None
    Args:
        all_graph_embeddings (_type_): List of the peptide embeddings.
        l (int, optional): The length of the substructure. Defaults to 1.
    Returns:
        _type_: _description_
    """
    subseqs = []
    seqdict = {}  # Dictionary to store the activations of the substructures
    avg_importance = []

    # Get all possible substructures of length l
    for i, k in enumerate(itertools.product(AMINO_ACIDS, repeat=l)):
        subseqs.append("".join(k))

    # Iterate over all peptides and get the list of activations of the substructures within the peptides.
    for i, d in enumerate(all_graph_embeddings):
        # Iterate over all substructures within the peptide sequence
        for j in range(len(d["desc"]) - 1):
            # Get the average activation of the substructure, sum the activations and divide by the length of the substructure
            # If the substructure is not in the dictionary, add it else append the activation to the list
            if d["desc"][j : j + l] not in seqdict.keys():
                seqdict[d["desc"][j : j + l]] = [sum(d["avg_desc_act"][j : j + l]) / l]
            else:
                seqdict[d["desc"][j : j + l]] = seqdict[d["desc"][j : j + l]] + [
                    sum(d["avg_desc_act"][j : j + l]) / l
                ]

    # Finally, get the average importance of the substructures.
    for i in subseqs:
        if i in seqdict.keys():
            avg_importance.append(np.mean(seqdict[i]))
        else:
            avg_importance.append(0)

    pepsdf = pd.DataFrame([subseqs, avg_importance]).T
    pepsdf.columns = ["pep", "Imp"]
    return pepsdf
