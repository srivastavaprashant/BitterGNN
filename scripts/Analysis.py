import pandas as pd
import numpy as np
import seaborn as sns
import warnings
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from scipy import stats
import matplotlib.pyplot as plt
import torch
import torch_geometric
from pathlib import Path
import sys

parent_path = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_path))

from settings import DATA_DIR, AMINO_ACIDS
from source.models import BitterGCN_MixedPool

warnings.filterwarnings("ignore")
print(
    f"Using torch version: {torch.__version__} and torch_geometric version: {torch_geometric.__version__}"
)

from source.analysis import (
    read_data,
    preprocess,
    get_graph_embeddings,
    pepvecs_2d_plot,
    get_clusters,
    train_model,
    get_subseq_activations,
)


def train_and_get_embeddings(df):
    """
    Trains a model and returns the graph embeddings.

    Args:
        df (pandas.DataFrame): The input dataframe containing peptide sequences.

    Returns:
        list: A list of graph embeddings.

    """
    # Initial model setup
    hidden_channels = 16
    epochs = 10

    # Read data and preprocess it
    print("Reading peptide sequences data...")
    print("Preprocessing protein sequences to create graphs...")
    protien_graphs, encoder = preprocess(df)

    # Train the model
    # Here we are using the GCN having three convolution layers and mixed pooling schema.
    print("Training the model...")
    model = BitterGCN_MixedPool(hidden_channels=hidden_channels)
    optimizer = Adam(model.parameters(), lr=0.05)
    criterion = CrossEntropyLoss()
    model, train_loader, test_loader, fig = train_model(
        model, protien_graphs, optimizer, criterion, epochs=10
    )
    fig.savefig("results/ROC Curve.pdf")

    # ## Analysis
    print("Extracting graph embeddings...")
    train_graph_embeddings = get_graph_embeddings(model, train_loader, encoder)
    test_graph_embeddings = get_graph_embeddings(model, test_loader, encoder)
    all_graph_embeddings = test_graph_embeddings + train_graph_embeddings

    return all_graph_embeddings


def log_clustering_results(
    embd,
    descs,
    graph_embeddings,
    method,
    method_args,
    embd_type,
    log_file="results/clusters.res",
):
    """
    Logs the clustering results to a file.

    Args:
        embd (numpy.ndarray): The embeddings of the peptides.
        descs (pandas.DataFrame): The descriptions of the peptides.
        graph_embeddings (numpy.ndarray): The graph embeddings of the peptides.
        method (str): The clustering method to use.
        method_args (dict): The arguments for the clustering method.
        embd_type (str): The type of embeddings used.
        log_file (str, optional): The path to the log file. Defaults to "results/clusters.res".
    """
    clusters, kmeans_tsne_embd, aminoacid_pop_in_clusters, fig = get_clusters(
        embd,
        descs,
        graph_embeddings,
        method=method,
        method_args=method_args,
        embd_type=embd_type,
    )
    fig.savefig(f"results/{method} Clustering {embd_type}.png")
    pop = clusters[["desc", "Cluster"]]

    pop.desc = pop.desc.apply(lambda x: [i for i in x])
    with open(log_file, "a") as f:
        f.write(
            f"{embd_type} with {method} for {len(clusters.Cluster.unique())} clusters\n"
        )
        f.write("Least Bitter Cluster ----> Most Bitter Cluster\n")
        f.write("----------------------------------------------\n")
        f.write(aminoacid_pop_in_clusters.round(2).to_string() + "\n\n")
        f.write("Accuracy and bitter population in the clusters\n")
        f.write(
            clusters.groupby("Cluster")
            .agg(
                n_bitterpeps=("label", sum),
                total_peps=("label", len),
                ModelAcc=("acc", "mean"),
            )
            .round(2)
            .to_string()
        )
        f.write("\n\n")


def main():
    print("BitterGCN-Analysis")
    df = read_data(DATA_DIR)
    labels = df.bitter
    seq = df.seq.values

    all_graph_embeddings = train_and_get_embeddings(df)

    conv_feats = [i["conv3_feats"].mean(axis=0) for i in all_graph_embeddings]
    print("Creating 2D plots...")
    print("Creating TSNE Embedding plot...")
    tsne_embd, descs, labels, fig = pepvecs_2d_plot(all_graph_embeddings, "tsne")
    fig.savefig("results/TSNE Embedding plot.pdf")
    print("Creating UMAP Embedding plot...")
    umap_embd, descs, labels, fig = pepvecs_2d_plot(all_graph_embeddings, "umap")
    fig.savefig("results/UMAP Embedding plot.pdf")
    print("Creating PCA Embedding plot...")
    pca_embd, descs, labels, fig = pepvecs_2d_plot(all_graph_embeddings, "pca")
    fig.savefig("results/PCA Embedding plot.pdf")

    print("Clustering the embeddings...")
    # Create a new results file
    open("results/clusters.res", "w").close()

    print("KMeans Clustering with TSNE embeddings...")
    log_clustering_results(
        embd=tsne_embd,
        descs=descs,
        graph_embeddings=all_graph_embeddings,
        method="kmeans",
        method_args={"n_clusters": 4},
        embd_type="TSNE",
        log_file="results/clusters.res",
    )

    print("KMeans Clustering with 16D embeddings...")
    log_clustering_results(
        embd=conv_feats,
        descs=descs,
        graph_embeddings=all_graph_embeddings,
        method="kmeans",
        method_args={"n_clusters": 4},
        embd_type="16D",
        log_file="results/clusters.res",
    )

    # ## Node level embeddings: All Data
    # ##### Amino-acid level (Mono-pep)
    # Mono-peptides
    # Avg
    monopeps = get_subseq_activations(all_graph_embeddings, l=1)

    # Pos
    pos_graphs = [i for i in all_graph_embeddings if i["label"] == 1]
    posmonopeps = get_subseq_activations(pos_graphs, l=1)

    # Neg
    neg_graphs = [i for i in all_graph_embeddings if i["label"] == 0]
    negmonopeps = get_subseq_activations(neg_graphs, l=1)

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.bar(
        x=np.arange(20) - 0.25,
        height=np.abs(negmonopeps.Imp) / sum(np.abs(negmonopeps.Imp)),
        width=-0.5,
        color=sns.color_palette()[2],
    )
    ax.bar(
        x=np.arange(20) + 0.25,
        height=np.abs(posmonopeps.Imp) / sum(np.abs(posmonopeps.Imp)),
        width=-0.5,
        color=sns.color_palette()[1],
    )
    ax.set_xticks(np.arange(20), AMINO_ACIDS)
    ax.legend(["Non-Bitter", "Bitter"])
    ax.set_title(
        "Average BitterGCN embedding outputs for bitter and non-bitter peptide descriptors"
    )
    fig.savefig("results/Amino-acid level (Mono-pep).png")

    # Di-peptides
    dipeps = get_subseq_activations(all_graph_embeddings, l=2)

    # Pos Di-peptides
    pos_graphs = [i for i in all_graph_embeddings if i["label"] == 1]
    posdipeps = get_subseq_activations(pos_graphs, l=2)

    # Neg Di-peptides
    neg_graphs = [i for i in all_graph_embeddings if i["label"] == 0]
    negdipeps = get_subseq_activations(neg_graphs, l=2)

    fig, axs = plt.subplots(
        1, 3, figsize=(16, 8), gridspec_kw={"width_ratios": [1, 1, 0.05]}
    )
    axs[0].imshow(
        np.rot90(np.reshape(np.abs(posdipeps["Imp"].to_list()), (20, 20))), cmap="YlGn"
    )
    axs[0].set_xticks(np.arange(20), AMINO_ACIDS, minor=False)
    axs[0].set_yticks(np.arange(20), AMINO_ACIDS[::-1])
    axs[0].set_title("Average activation of Dipeptide subsequences in bitter peptides")
    im2 = axs[1].imshow(
        np.rot90(np.reshape(np.abs(negdipeps["Imp"].to_list()), (20, 20))), cmap="YlGn"
    )
    axs[1].set_xticks(np.arange(20), AMINO_ACIDS, minor=False)
    axs[1].set_yticks(np.arange(20), AMINO_ACIDS[::-1])
    axs[1].set_title(
        "Average activation of Dipeptide subsequences in non-bitter peptides"
    )
    fig.colorbar(im2, cax=axs[2])
    fig.tight_layout()
    fig.savefig("results/Di-peptides.png")

    # Tri-peptides
    tripeps = get_subseq_activations(all_graph_embeddings, l=3)
    tripeps.sort_values(by="Imp", ascending=True)

    # Pos Tri-peptides
    pos_graphs = [i for i in all_graph_embeddings if i["label"] == 1]
    postripeps = get_subseq_activations(pos_graphs, l=3)

    # Neg Tri-peptides
    neg_graphs = [i for i in all_graph_embeddings if i["label"] == 0]
    negtripeps = get_subseq_activations(neg_graphs, l=3)

    # Tetra-peptides
    tetrapeps = get_subseq_activations(all_graph_embeddings, l=4)
    tetrapeps.sort_values(by="Imp", ascending=False)

    # Pos Tetra-peptides
    pos_graphs = [i for i in all_graph_embeddings if i["label"] == 1]
    postetrapeps = get_subseq_activations(pos_graphs, l=4)

    # Neg Tetra-peptides
    neg_graphs = [i for i in all_graph_embeddings if i["label"] == 0]
    negtetrapeps = get_subseq_activations(neg_graphs, l=4)

    posallsubpeps = (
        posdipeps.append(posmonopeps).append(postripeps).append(postetrapeps)
    )

    posallsubpeps["Imp"] = np.abs(posallsubpeps["Imp"]) / np.sum(
        np.abs(posallsubpeps["Imp"])
    )
    posallsubpeps["len"] = posallsubpeps["pep"].apply(len)
    posallsubpeps = posallsubpeps[posallsubpeps.Imp != 0]

    negallsubpeps = (
        negdipeps.append(negmonopeps).append(negtripeps).append(negtetrapeps)
    )

    negallsubpeps["Imp"] = np.abs(negallsubpeps["Imp"]) / np.sum(
        np.abs(negallsubpeps["Imp"])
    )
    negallsubpeps["len"] = negallsubpeps["pep"].apply(len)
    negallsubpeps = negallsubpeps[negallsubpeps.Imp != 0]

    allsubpeps = dipeps.append(monopeps).append(tripeps).append(tetrapeps)

    allsubpeps["Imp"] = np.abs(allsubpeps["Imp"]) / np.sum(np.abs(allsubpeps["Imp"]))
    allsubpeps["len"] = allsubpeps["pep"].apply(len)
    allsubpeps = allsubpeps[allsubpeps.Imp != 0]

    descs_qvals = pd.read_csv("data/Qvalues.csv", sep=", ")
    delf = descs_qvals.set_index("Desc").FreeEnergy.to_dict()

    def get_q(x):
        try:
            return np.int(sum([delf[i] for i in x]) / len(x))
        except:
            return None

    posallsubpeps["Qval"] = posallsubpeps.pep.apply(lambda x: get_q(x))
    posallsubpeps["Present"] = posallsubpeps.pep.isin(df.seq)
    negallsubpeps["Qval"] = negallsubpeps.pep.apply(lambda x: get_q(x))
    negallsubpeps["Present"] = negallsubpeps.pep.isin(df.seq)
    allsubpeps["Qval"] = allsubpeps.pep.apply(lambda x: get_q(x))
    allsubpeps["Present"] = allsubpeps.pep.isin(df.seq)

    posallsubpeps.shape

    negallsubpeps.shape

    posallsubpeps.sort_values("Imp", ascending=False)

    d1_peps = (
        posallsubpeps[posallsubpeps.len == 1]
        .sort_values(by="Imp", ascending=False)[["pep", "Imp"]]
        .apply(lambda x: f"{x['pep']} & {x['Imp']:.5f}", axis=1)
        .to_list()
    )
    d1_peps.append("C & -")
    d2_peps = (
        posallsubpeps[posallsubpeps.len == 2]
        .sort_values(by="Imp", ascending=False)[["pep", "Imp"]]
        .apply(lambda x: f"{x['pep']} & {x['Imp']:.5f}", axis=1)
        .to_list()
    )
    d3_peps = (
        posallsubpeps[posallsubpeps.len == 3]
        .sort_values(by="Imp", ascending=False)[["pep", "Imp"]]
        .apply(lambda x: f"{x['pep']} & {x['Imp']:.5f}", axis=1)
        .to_list()
    )
    d4_peps = (
        posallsubpeps[posallsubpeps.len == 4]
        .sort_values(by="Imp", ascending=False)[["pep", "Imp"]]
        .apply(lambda x: f"{x['pep']} & {x['Imp']:.5f}", axis=1)
        .to_list()
    )

    with open("results/Bitter peptides.res", "w") as f:
        f.write("Mono-peptides & Di-peptides & Tri-peptides & Tetra-peptides\n")
        for i, d1 in enumerate(d1_peps):
            f.write(f"{d1} & {d2_peps[i]} & {d3_peps[i]} & {d4_peps[i]} \n")

    # stats.ttest_ind(posallsubpeps.Imp, negallsubpeps.Imp)

    # sns.kdeplot(posallsubpeps.Imp.reset_index(drop=True))
    # sns.kdeplot(negallsubpeps.Imp.reset_index(drop=True))
    # sns.kdeplot(allsubpeps.Imp.reset_index(drop=True))

    # plt.legend(["Bitter", "Non-bitter", "All"])
    # plt.title("Distribution of BitterGCN embedding outputs for descriptors.")
    # plt.savefig("results/EmbeddingOutputDistribution.png")
    # df["Qval"] = df.seq.apply(lambda x: get_q(x))

    clusters, kmeans_tsne_embd, aminoacid_pop_in_clusters, fig = get_clusters(
        tsne_embd,
        descs,
        all_graph_embeddings,
        method="kmeans",
        method_args={"n_clusters": 4},
        embd_type="TSNE",
    )
    clusters["Imp"] = clusters.conv3_feats.apply(lambda x: x.mean())

    sns.kdeplot(clusters[clusters.label == 1].Imp)
    sns.kdeplot(clusters[clusters.label == 0].Imp)
    clusters.to_csv("results/all_embeddings.csv", index=False)


if __name__ == "__main__":
    main()
