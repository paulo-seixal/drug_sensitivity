# drug_sensitivity

**Dataset Description**: Genomics in Drug Sensitivity in Cancer (GDSC) is a resource for therapeutic biomarker discovery in cancer cells. It contains wet lab IC50 for 100s of drugs in 1000 cancer cell lines. In this dataset, we use **RMD normalized gene expression** for cancer lines and SMILES for drugs. **Y** is the **log normalized IC50**. This is the version 2 of GDSC, which uses improved experimental procedures.

**Task Description**: Regression. Given the gene expression of cell lines and the SMILES of drug, predict the drug sensitivity level.

**Dataset Statistics**: 92,703 pairs, 805 cancer cells and 137 drugs


## Installation

### For pip users

```bash

pip install -r requirements.txt

```

### For conda users


 Navigate to the project directory:

   ```bash
    cd drug_sensitivity
   ```


Create a Conda environment:

```bash
conda env create -f environment.yml
```
