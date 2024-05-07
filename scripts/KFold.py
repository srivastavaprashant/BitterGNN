import os

data_dir = "data/"
model_dir = "models/"
import pandas as pd
from pathlib import Path
import sys

parent_path = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_path))

from source.utils import read_data, preprocess, run_kfold_test
from source.models import (
    BitterGCN_Baseline,
    BitterGCN_MixedPool,
    BitterGAT_Baseline,
    BitterGAT_MixedPool,
    BitterGraphSAGE_Baseline,
    BitterGraphSAGE_MixedPool,
)

df = read_data(data_dir)


# k Fold
nsplits = 10
graph_data = preprocess(df)
n = len(graph_data)
graph_data[0], graph_data[7]


kfold_res_df = pd.DataFrame(index=list(range(10)))
model = BitterGCN_Baseline(hidden_channels=32)
fold_test_acc, fold_test_roc = run_kfold_test(nsplits, graph_data, BitterGCN_Baseline)
kfold_res_df.loc[:, "ACC_GCN_Baseline"] = fold_test_acc
kfold_res_df.loc[:, "ROC_GCN_Baseline"] = fold_test_roc

model = BitterGCN_MixedPool(hidden_channels=32)
fold_test_acc, fold_test_roc = run_kfold_test(nsplits, graph_data, BitterGCN_MixedPool)
kfold_res_df.loc[:, "ACC_GCN_MixedPool"] = fold_test_acc
kfold_res_df.loc[:, "ROC_GCN_MixedPool"] = fold_test_roc


model = BitterGAT_Baseline(hidden_channels=32)
fold_test_acc, fold_test_roc = run_kfold_test(nsplits, graph_data, BitterGAT_Baseline)
kfold_res_df.loc[:, "ACC_GAT_Baseline"] = fold_test_acc
kfold_res_df.loc[:, "ROC_GAT_Baseline"] = fold_test_roc


model = BitterGAT_MixedPool(hidden_channels=32)
fold_test_acc, fold_test_roc = run_kfold_test(nsplits, graph_data, BitterGAT_MixedPool)
kfold_res_df.loc[:, "ACC_GAT_MixedPool"] = fold_test_acc
kfold_res_df.loc[:, "ROC_GAT_MixedPool"] = fold_test_roc

model = BitterGraphSAGE_Baseline(hidden_channels=32)
fold_test_acc, fold_test_roc = run_kfold_test(
    nsplits, graph_data, BitterGraphSAGE_Baseline
)
kfold_res_df.loc[:, "ACC_GraphSAGE_Baseline"] = fold_test_acc
kfold_res_df.loc[:, "ROC_GraphSAGE_Baseline"] = fold_test_roc


model = BitterGraphSAGE_MixedPool(hidden_channels=32)
fold_test_acc, fold_test_roc = run_kfold_test(
    nsplits, graph_data, BitterGraphSAGE_MixedPool
)
kfold_res_df.loc[:, "ACC_GraphSAGE_MixedPool"] = fold_test_acc
kfold_res_df.loc[:, "ROC_GraphSAGE_MixedPool"] = fold_test_roc


kfold_res_df.to_csv("results/kfold_results.csv", index=False)
