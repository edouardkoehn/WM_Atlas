# White Matter Atlas
### Defintion:
This contains all the source code for the WM-Atlas project. The goal of this repository
is to implement several pipelines to produce WM-Atlas using Voxel-wise white matter brain
graph from d-MRI data.
This repository is link to HCP100 dataset stored on the MIPLAB infrastructure.


The dependancy management as been done using [Poetry](https://python-poetry.org/). Please refer to the documentation for more informations.


To run a pipeline, follow the installation instructions and refer to the documentation contains on the README.


### Repository structure:
- ```/clustering``` contains all the source code of the project
- ```/scripts``` contains bash script for running the code on the MIPLAB infrastructure
- ```/matalb``` matlab utils fonction use in specific workflows
-  ```/tests``` unitary for the repository ( in progress)


Edouard koehn, MIPLAB, EPFL
edouard.koehn@epfl.ch

## Installation

1) Clone the repo

```bash
git clone https://github.com/edouardkoehn/WM_Atlas.git
```
2) Create your virtual env
```bash
conda create -n wm python=3
conda activate wm
```
3) Install poetry
```bash
pip install poetry
```
4) Install the dependancies
```bash
poetry install
```
5)  Modify the general path in ```/clustering/config.yml```
```bash
root_dir: "" #Path to the root of the directory
output_dir: "" #Path to the output dir
#General path to the graph data
general_path: ""
dir_path: ""
file_path: "" #name of the .mat file
src_nifi_path: "" #name of the src nifti file
```
6) Test the installation process
```bash
poetry run pytest
```


## Workflows
This repository contains 4 different workflow. The instructions to run those workflows are defined in the following sections.
### Extract Eigen modes
This workflow extract the eigenmodes of one specific subject. It has a command line interface. You can get information on the argument of this worklfow using :
```bash
/Code/MVetterli/WM_Atlas$ extract_eigen --help
```
It would display the following informations:

```bash
 Workflow to extract the eigenvalues of the laplacien of a subject's graph

  Args: subject_id(int): coresponding patient id in the database,
        method(str): method used for computing the laplacien either combinatorial (comb),randomwalk laplacian (rw) or symetric laplacian(sym)

        threshold(float): thresholding value for the binarisation of the matrix
        k_eigen(int):number of eigen value used

        nitfi_type(str): Which nifti you want to produce, if "mni" the workflow, would
        produce the extraction in the acpc and in the mni space.

        save(bool):saving the intermediate matrix

Options:
  -i, --subject_id INTEGER     Patient id  [required]
  -m, --method [comb|rw|sym]   Clustering method used
                               (comb:Combinatorial(default), rd:Random Walk
                               version, sym:Symetrical laplacian)
  -t, --threshold FLOAT        Thresholding values for the binarisation of A.
                               If not specified, no thresholding would be
                               applied
  -k, --k_eigen INTEGER        Number of computed eigen values  [required]
  -n, --nifti_type [acpc|mni]  Type of nifti saved
  -s, --save BOOLEAN           Saving the intermediate matrix (L,U,v)
  --help                       Show this message and exit.
```
Once the pipeline is finished the data would be save in the ouput folder specified in the ```/clustering/config.yml``` file.
### Clustering individual
This workflow produce a clustering on a unique subject.It has a command line interface. You can get information on the argument of this worklfow using :
```bash
/Code/MVetterli/WM_Atlas$  clustering_ind --help --help
```
It would display the following informations:

```bash

 Workflow to produce the spectral clustering for a specific individual

    Args: subject_id(int): coresponding patient id in the database,

        method(str):method used for computing the laplacien either combinatorial (comb),randomwalk laplacian (rw) or symetric laplacian(sym)

        threshold(float): thresholding value for the binarisation of the matrix

        k_eigen(int):number of eigen value used

        nifti_type(str): from which nifti space, you want to produce the clustering

        save(bool):saving the intermediate matrix

Options:
  -i, --subject_id INTEGER        Patient id  [required]
  -m, --method [comb|rw|sym]      Clustering method used
                                  (comb:Combinatorial(default), rd:Random Walk
                                  version, sym:Symetrical Laplacian)
  -t, --threshold FLOAT           Thresholding values for the binarisation of
                                  A. If not specified, no thresholding would
                                  be applied
  -k, --k_eigen INTEGER           Number of computed eigen values  [required]
  -n, --nifti_type [acpc|mni]     Nifti space used  [required]
  -v, --value_type [cluster|distance|z_score]
                                  Value to assign in the volume
  -s, --save BOOLEAN              Saving the intermediate matrix (L, Lrw)
  --help                          Show this message and exit.
```

Once the pipeline is finished the data would be save in the ouput folder specified in the ```/clustering/config.yml``` file.
### Clustering population
This workflow produce a the boostrap experiment. It has a command line interface. You can get information on the argument of this worklfow using :
```bash
/Code/MVetterli/WM_Atlas$  clustering_boostrap --help --help
```
It would display the following informations:

```bash
  Workflow to produce the spectral clustering at the population level

  Args: subject_id(int): coresponding patient id in the database,

        method(str): method used for computing the laplacien

        threshold(float): thresholding value for the binarisation of the matrix

        k_eigen(int):number of eigen value used

        nifti_type:from which nifti the clustering would be produced

        save(bool):saving the intermediate matrix

Options:
  -i, --subject_ids INTEGER       Patient id  [required]
  -m, --method [comb|rw|sym]      Clustering method used
                                  (comb:Combinatorial(default), rd:Random
                                  Walke version)
  -t, --threshold FLOAT           Thresholding values for the binarisation of
                                  A. If not specified, no thresholding would
                                  be applied
  -k, --k_eigen INTEGER           Number of computed eigen values  [required]
  -n, --nifti_type [reslice]      Nifti space used  [required]
  -v, --value_type [cluster|distance|z_score]
                                  Value to assign in the volume
  -s, --save BOOLEAN              Saving the intermediate matrix (L, Lrw)
  --help                          Show this message and exit.
```

Once the pipeline is finished the data would be save in the ouput folder specified in the ```/clustering/config.yml``` file.
### Clustering Boostrap
This workflow extract the eigenmodes of one specific subject. It has a command line interface. You can get information on the argument of this worklfow using :
```bash
/Code/MVetterli/WM_Atlas$ extract_eigen --help
```
It would display the following informations:

```bash
 Workflow to produce the spectral clustering at the population base
 Args:
  subject_id(int): coresponding patient id in the database,
  method(str): method used for computing the laplacien,
  threshold(float): thresholding value for the binarisation of the matrix
  k_eigen(int):number of eigen value used
  nifti_type:from which nifti the clustering would be produced
  save(bool):saving the intermediate matrix

Options:
  -i, --subject_ids INTEGER       Patient id  [required]
  -m, --method [comb|rw|sym]      Clustering method used
                                  (comb:Combinatorial(default), rd:Random
                                  Walke version)
  -t, --threshold FLOAT           Thresholding values for the binarisation of
                                  A. If not specified, no thresholding would
                                  be applied
  -k, --k_eigen INTEGER           Number of computed eigen values  [required]
  -c, --clusters INTEGER          Number of clusters
  -n, --nifti_type [reslice]      Nifti space used  [required]
  -v, --value_type [cluster|distance|z_score]
                                  Value to assign in the volume
  -b, --batch_number INTEGER      "Batch number
  -s, --save BOOLEAN              Saving the intermediate matrix (L, Lrw)
  --help                          Show this message and exit.
```
Once the pipeline is finished the data would be save in the ouput folder specified in the ```/clustering/config.yml``` file.
