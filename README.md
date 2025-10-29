# EEGNet Motor Imagery Classifier - Capstone Project 🧠


This repository contains the code for a Capstone project focused on implementing the **EEGNet model** for Brain-Computer Interface (BCI) motor imagery classification using the BCI Competition IV Dataset 2a.

**Current Status:** This repository currently holds the **Data Loading and Preprocessing Pipeline** as required for **Milestone 2**. Model implementation, training, and evaluation code will be added in subsequent milestones.

---

## 🎯 Project Goal

The overall objective of this project is to build, train, and evaluate an end-to-end deep learning pipeline using the **EEGNet architecture** to classify four distinct motor imagery tasks (left hand, right hand, feet, tongue) directly from raw electroencephalography (EEG) signals. The project aims to achieve proficiency in handling EEG data, implementing a known deep learning architecture from research, and evaluating its performance on a benchmark dataset.

---

## 📊 Data Source & Setup Instructions <a name="data-setup"></a>

* **Dataset:** BCI Competition IV, Dataset 2a
* **Description:** This is a standard benchmark dataset containing EEG recordings from 9 subjects performing the four specified motor imagery tasks. This pipeline uses only the labeled training data (`*T.mat` files).
* **Download Location:** The dataset (`A01T.mat` through `A09T.mat` files) must be downloaded manually from the official source (requires free registration):
    * [BNCI Horizon 2020 - Dataset 2a](http://bnci-horizon-2020.eu/database/data-sets)
* **❗ Important Setup Step:**
    1.  After cloning this repository, create a folder named exactly `data` **inside** the `EEGNET-Capstone` directory.
    2.  Place **all** the downloaded `.mat` training files (`A01T.mat`, `A02T.mat`, ..., `A09T.mat`) directly into this `data` folder.
    3.  The `preprocessing_pipeline.py` script is already configured to look for the data in this `./data/` subfolder. **No code modifications are needed if the data is placed correctly.**

---

## 🛠️ Milestone 2: Preprocessing Pipeline

This repository currently contains the Python script (`preprocessing_pipeline.py`) which performs the following steps:

1.  Loads the raw `.mat` files for subjects A01T through A09T from the `./data/` folder.
2.  Validates subjects based on the presence of labeled trials.
3.  Aggregates valid trials from all included subjects.
4.  Epochs the continuous EEG data into 4-second trials aligned with the task cues (0s to 4s post-cue).
5.  Applies an 8-30 Hz band-pass filter using the MNE library.
    
6.  Splits the aggregated data into training (80%) and testing (20%) sets, stratified by class label.
7.  Applies StandardScaler (fit on training data only) to standardize the EEG channel data.
8.  Saves the fitted `StandardScaler` object (`eeg_scaler_multisubject.joblib`) for potential use in later stages.

