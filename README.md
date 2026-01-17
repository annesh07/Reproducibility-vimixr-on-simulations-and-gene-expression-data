# Reproducibility:vimixr on simulations and gene-expression data
This repository contains R codes (***Code.R*** file) and files for reproducing results obtained by applying **vimixr** R package on simulations and gene expression data for [Leukemia sub-type clustering](https://schlieplab.org/Static/Supplements/CompCancer/Affymetrix/armstrong-2002-v2/). The package mainly performs probabilistic clustering of a given data.
Most of the computations are performed in Rstudio. Time intensive computations (~ 10-20 hours long) are carried out using [Curta Cluster](https://redmine.mcia.fr/projects/cluster-curta). For reference of the Curta codes, sample R codes are provided in ***SampleCodesCurta.R*** file. As the codes take 10~20 hours for computation even in the [Curta Cluster](https://redmine.mcia.fr/projects/cluster-curta), they are commented out in ***SampleCodesCurta.R*** file, but can be found under the commented line "#the code:" for each sample.

The layout is simple: 

  - ***Code.R*** contains the complete code for generating all the figures and one supplementary table
  
  - ***SampleCodesCurta.R*** for reference; contains the Curta codes (commented out)
  
  - *Results* folder contains the intermediate and final computation run results as .xlsx/.csv files (explained in ***Code.R***)
  
    1) Figures in *Results/Figures*  
    
    2) Tables in *Results/Table*

To access the repository from Github, the user can clone the git repository or download the zip file from the ![](https://img.shields.io/static/v1?label=&message=Code&color=green) tab. 