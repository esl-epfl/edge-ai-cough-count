# Edge Artificial Intelligence (edge-AI) Cough Counting

## Background 
Counting the number of times a patient coughs per day is an essential biomarker in determining treatment efficacy for novel antitussive therapies and personalizing patient care. There is a need for wearable devices that employ multimodal sensors to perform accurate, privacy-preserving, automatic cough counting algorithms directly on the device in an Edge-AI fashion. To advance this research field, our team from the Embedded Systems Laboratory (ESL) of EPFL contributed the first publicly accessible cough counting dataset of multimodal biosignals. The database contains nearly 4 hours of biosignal data, with both acoustic and kinematic modalities, covering 4,300 annotated cough events. Furthermore, a variety of non-cough sounds and motion scenarios mimicking daily life activities are also present, which the research community can use to accelerate ML algorithm development. 

This repository contains useful functions that researchers can use to use our public dataset, including:
* Code for generating biosignal segments from the public dataset to streamline ML algorithm development
* Testing your predicted cough locations against the ground truth in an event-based manner

## Data access
The edge-AI cough counting dataset can be found at the following Zenodo link: https://zenodo.org/record/7562332#.Y87MenbMKUm

## Getting started
To start using the code, please make sure you have all of the necessary Python dependencies. We recommend doing this in a new Conda or pip environment using one of the following commands:

Conda:
```
conda env create -f environment.yml

```
Pip:
```
pip install -r requirements.txt
```

## Usage

### Dataset generation
Several Jupyter notebooks are available to illustrate the functionality of the code.

In `Segmentation_Augmentation.ipynb`, we demonstrate how to turn the raw biosignals and annotations provided in the dataset into segmented biosignals ready to input into ML models.

In `Cough_Annotation.ipynb`, we explain how the fine-grained cough labeling was performed in a semi-automatic manner, in case other teams wish to merge their datasets and keep the labeling scheme consistent.

### Testing your predictions
Once you use our datset to develop a ML model for counting coughs, the next step is to validate your predicted cough locations against the ground-truth cough locations.

The `Compute_Success_Metrics.ipynb` notebook demonstrates how to use [the timescoring event-based success evaluation library](https://pypi.org/project/timescoring/) to extract clinically meaningful success metrics from your algorithms.

### Functions
The `helpers.py` file contains useful functions for quickly iterating through the database structure, performing some signal processing, and loading the biosignals and annotations.

The `dataset_gen.py` file segments the raw biosignals and contains useful functions for creating a cough detection database for training edge-AI Machine Learning Models.

# Citations

If you use the open-source dataset in your work, please cite our publication:

```
@inproceedings{orlandic_multimodal_2023,
	address = {Sydney, Australia},
	title = {A {Multimodal} {Dataset} for {Automatic} {Edge}-{AI} {Cough} {Detection}},
	copyright = {https://doi.org/10.15223/policy-029},
	url = {https://ieeexplore.ieee.org/document/10340413/},
	doi = {10.1109/EMBC40787.2023.10340413},
	language = {en},
	urldate = {2024-04-10},
	booktitle = {2023 45th {Annual} {International} {Conference} of the {IEEE} {Engineering} in {Medicine} \& {Biology} {Society} ({EMBC})},
	publisher = {IEEE},
	author = {Orlandic, Lara and Thevenot, Jérôme and Teijeiro, Tomas and Atienza, David},
	month = jul,
	year = {2023},
	pages = {1--7},
}```

# Contact

For questions or suggestions, please contact lara.orlandic@epfl.ch.

