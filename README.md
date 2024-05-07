Srivastava P.*, Steuer A.*, Nicoli A., Schulz K., Bej S., Di Pizio A. & Wolkenhauer O. Bitter Peptide Prediction Using Graph Neural, target journal: Journal of Cheminformatics, SUBMITTED

## Installation
1. Create a python version 3.9.
```bash
conda create -n bittepep python=3.9
conda activate bitterpep
```

2. Install the python packages from requirements file.
```
pip install -r requirments.txt
```

## Running the benchmark and analysis
1. Run the analysis
```
python scripts/Analysis.py
```

2. Run the KFold benchmark
```
python scripts/KFold.py
```
