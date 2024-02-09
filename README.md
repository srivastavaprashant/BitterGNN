Srivastava P.*, Steuer A.*, Nicoli A., Schulz K., Bej S., Di Pizio A. & Wolkenhauer O. Bitter Peptide Prediction Using Graph Neural, target journal: Journal of Cheminformatics, SUBMITTED

There are two parts of this project present in two jupyter notebooks:
1. Analysis.ipynb: BitterGCN model is trained and tested. The model is then used to get the graph embeddings and analyse them. 
    - Graph level analysis:
        - 2D representation (TSNE plot). 
        - Clustering using OPTICS clustering method. (found 4 clusters)
    - Node level details   
2. KFold.ipynb: Cross Validation comparison of the different model architectures.


## Installation
```bash
conda create -n bittepep python=3.9
conda activate bitterpep
pip install -r requirments.txt
```

