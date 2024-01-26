# drug_sensitivity

## Key features of the work

**<ins>Dataset Description<\ins>**: Genomics in Drug Sensitivity in Cancer (GDSC) is a resource for therapeutic biomarker discovery in cancer cells. It contains wet lab IC50 for 100s of drugs in 1000 cancer cell lines. In this dataset, we use **RMD normalized gene expression** for cancer lines and SMILES for drugs. **Y** is the **log normalized IC50**. This is the version 2 of GDSC, which uses improved experimental procedures.

**<ins>Task Description**: Regression. Given the gene expression of cell lines and the SMILES of drug, predict the drug sensitivity level.

**Dataset Statistics**: 92,703 pairs, 805 cancer cells and 137 drugs

**Feature Reduction**: We used the gene expression data from [PACCMANN](https://academic.oup.com/nar/article/48/W1/W502/5836770). The gene expression data is reduced to 2,089 represented by the top 20 genes included in the biomolecular interaction network for each drug.

**Fingerprint**: We used the Morgan fingerprint with radius 2 and 1024 bits for the majority of the models, expect for the GCN model and GAT model, which uses graph representation of the drug.

## <ins>Hugging Face Spaces Demo
We implemented a demo of the model using Hugging Face Spaces. The demo can be found [here](https://huggingface.co/spaces/paulo-seixal/drug_sensitivity). The demo allows users to input the drug in SMILES and get the predicted IC50, considering all the cell lines in the GDSC2 dataset.

![hug](data/hugging_face_spaces.png)

## Installation of dependencies

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
## Information about the files

**[data](data):** In this folder, you can find the data used in this project. The original data about the cell lines and drgus, as well as the metadata like the [gene expression](data/gene_expression.csv) and [cell lines information](data/cell_line_info.csv). In the sub-folder [splits](data/splits), you can find the splits used in this project used for training, validation and testing.

**<sis>[deep_learning](deep_learning):** Here you can find the implementation of the deep-learning models used in this project. The pipe line for the models implemented are: [DenseNet](deep_learning/pipeline_dense.ipynb), [GCN](deep_learning/pipeline_druggcn.ipynb) and [GAT](deep_learning/pipeline_druggat.ipynb).

**<sis>[trained_models](trained_models):** This folder contains the trained models for both machine learning and deep-learning models. The machine learning models are saved in the pickle format. The deep-learning models are saved in h5 format.

**<sis>[baseline_ml](baseline_ml.ipynb):** This notebook contains the implementation of the baseline machine learning models. The models implemented are: Linear Regression, HistGradientBoosting Regression and XGBoost.

**<sis>[data_splitting](data_splitting.ipynb):** This notebook contains the implementation of the data splitting. The data is split into train, validation and test sets. The files can be found in the [data/splits](data/splits) folder.

**<sis>[exploration](exploration.ipynb):** This notebook contains the exploration of the data. This notebook is used to understand the data and the features.

**<sis>[gradio](gradio.ipynb):** This notebook contains the implementation of the gradio demo. The demo allows users to input the drug in SMILES and get the predicted IC50, considering all the cell lines in the GDSC2 dataset.

**<sis>[unsupervised](unsupervised.ipynb):** This notebook contains the implementation of the unsupervised learning models. The models implemented are: Clustering and PCA. 


## Results

### Baseline Machine Learning Models
After hyperparameter tuning, we report the best performance of each model:

| Model | RMSE | MSE | MAE | R2 |
| --- | --- | --- | --- | --- |
| Linear Regression | 1.19 | 1.41 | 0.88 | 0.81 |
| HistGradientBoosting Regression | 1.14 | 1.30 | 0.86 | 0.83 |
| **XGBoost** | 1.09 | 1.20 | 0.82 | 0.84 |

The best model is XGBoost for IC50 prediction. Used in Hugging Face Spaces Demo - Drug Sensitivity Predictor - previously mentioned.

### Shap Analysis using XGBoost

![shap](data/Shap_analysis.png)

In SHAP analysis, it is possible to infer that the drugs Fingerprints are the most important features for the prediction of IC50. However, only 3 cell lines (TJP1; SDC4; SLC27A5) appear in the top 20 features that contribute the most to the IC50 prediction. 

### Deep-learning Models

For deep-learning models, we based our implementation on [Delora et al., 2022](https://github.com/BioSystemsUM/drug_response_pipeline). The model is trained for 100 epochs with early stopping patience of 10. The model is trained on a single GPU (NVIDIA GeForce RTX 3050TI).

| Model | RMSE | MSE | MAE | R2 |
| --- | --- | --- | --- | --- |
| **Dense** | 1.10 | 1.21 | 0.82 | 0.84 |
| GAT | 1.19 | 1.43 | 0.90 | 0.81 |
| GCN | 2.42 | 5.87 | 1.82 | 0.23 |

The best model is Dense for IC50 prediction.
