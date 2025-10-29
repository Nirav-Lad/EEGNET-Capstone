# EEGNet Motor Imagery Classifier - Capstone Project 🧠

This repository contains the code for a Capstone project focused on implementing the EEGNet model for Brain-Computer Interface (BCI) motor imagery classification.

**Current Status:** This repository currently holds the **Data Loading and Preprocessing Pipeline** as required for **Milestone 2**. Model training and evaluation code will be added in subsequent milestones.

## 🎯 Project Goal

The overall objective of this project is to build, train, and evaluate an end-to-end deep learning pipeline using the **EEGNet architecture** to classify four distinct motor imagery tasks (left hand, right hand, feet, tongue) directly from raw electroencephalography (EEG) signals.

## 📊 Data Source

* **Dataset:** BCI Competition IV, Dataset 2a
* **Description:** This is a standard benchmark dataset containing EEG recordings from 9 subjects performing the four motor imagery tasks.
* **Download:** The dataset (`A01T.mat` through `A09T.mat` files) must be downloaded manually from the official source (requires free registration):
    * [http://bnci-horizon-2020.eu/database/data-sets](http://bnci-horizon-2020.eu/database/data-sets)

## 🛠️ Milestone 2: Preprocessing Pipeline

This repository currently contains the Python script (`preprocessing_pipeline.py`) responsible for:

1.  Loading the raw `.mat` files for each subject.
2.  Aggregating valid trials from subjects with labels.
3.  Epoching the data into 4-second trials aligned with the task cues.
4.  Applying an 8-30 Hz band-pass filter using the MNE library.
5.  Splitting the aggregated data into training and testing sets.
6.  Applying StandardScaler to the EEG channels.

