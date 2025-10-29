import numpy as np
import scipy.io
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import mne
import warnings
import joblib

warnings.filterwarnings('ignore', category=RuntimeWarning)

# --- 1. Data Loading and Aggregation ---
def load_aggregate_data(data_path):
    all_epochs = []
    all_labels = []
    sampling_freq = None

    print("\n--- Aggregating data from all subjects... ---")
    if not os.path.isdir(data_path):
        print(f"ERROR: Data directory '{data_path}' not found.")
        print("Please ensure the 'data' folder containing .mat files exists.")
        return None, None, None

    subject_files_found = 0
    for i in range(1, 10):
        file_path = os.path.join(data_path, f'A0{i}T.mat')
        if not os.path.exists(file_path):
            continue

        subject_files_found += 1
        print(f"Processing Subject File: {os.path.basename(file_path)}...")
        try:
            mat_data_squeezed = scipy.io.loadmat(file_path, squeeze_me=True)
            subject_data = mat_data_squeezed['data']

            if 'y' not in subject_data.dtype.names or subject_data['y'].item().size <= 1:
                print(f"  Subject {i} has no valid trials. Skipping.")
                continue

            X = subject_data['X'].item()
            trial_starts = subject_data['trial'].item()
            y_subject = subject_data['y'].item()
            fs = subject_data['fs'].item()

            if sampling_freq is None:
                sampling_freq = fs
            elif sampling_freq != fs:
                print(f"  Warning: Subject {i} has different sampling frequency ({fs}Hz). Skipping.")
                continue

            epoch_start_s = 0.0
            epoch_end_s = 4.0
            epoch_start_samples = int(epoch_start_s * sampling_freq)
            epoch_end_samples = int(epoch_end_s * sampling_freq)

            subject_epochs = []
            valid_labels = []
            for j, start_sample in enumerate(trial_starts):
                start_index = start_sample + epoch_start_samples
                end_index = start_sample + epoch_end_samples

                if start_index >= 0 and end_index <= X.shape[0]:
                    num_channels_to_use = 22
                    single_epoch = X[start_index:end_index, :num_channels_to_use]

                    if single_epoch.shape[0] == (epoch_end_samples - epoch_start_samples):
                        subject_epochs.append(single_epoch)
                        valid_labels.append(y_subject[j])

            if subject_epochs:
                all_epochs.extend(subject_epochs)
                all_labels.extend(valid_labels)
                print(f"  Processed Subject {i}: Found {len(subject_epochs)} valid trials.")
            else:
                print(f"  Subject {i}: No valid epochs extracted.")

        except Exception as e:
            print(f"  Error processing Subject {i}: {e}. Skipping.")

    if subject_files_found == 0:
        print(f"ERROR: No subject .mat files found in '{data_path}'.")
        return None, None, None

    if not all_epochs:
        print("ERROR: No valid epochs found.")
        return None, None, None

    epochs_array = np.array(all_epochs)
    y = np.array(all_labels)
    y = y - 1

    print(f"\n--- Aggregation Complete ---")
    print(f"Total trials from all valid subjects: {len(epochs_array)}")
    print(f"Shape of aggregated epoched data (trials, samples, channels): {epochs_array.shape}")
    print(f"Shape of aggregated labels: {y.shape}")
    print(f"Unique Labels found (0-indexed): {np.unique(y)}")

    return epochs_array, y, sampling_freq

# --- 2. Preprocessing Function ---
def preprocess_data(epochs_array, y, sampling_freq):
    print("\n--- Starting Preprocessing ---")
    X_train, X_test, y_train, y_test = train_test_split(
        epochs_array, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Split data: Train={X_train.shape}, Test={X_test.shape}")

    low_freq = 8.0
    high_freq = 30.0

    X_train_transposed = X_train.transpose(0, 2, 1)
    X_test_transposed = X_test.transpose(0, 2, 1)

    X_train_filtered = mne.filter.filter_data(X_train_transposed, sfreq=sampling_freq, l_freq=low_freq, h_freq=high_freq, verbose=False)
    X_test_filtered = mne.filter.filter_data(X_test_transposed, sfreq=sampling_freq, l_freq=low_freq, h_freq=high_freq, verbose=False)

    X_train_filtered_final = X_train_filtered.transpose(0, 2, 1)
    X_test_filtered_final = X_test_filtered.transpose(0, 2, 1)

    num_train_trials, num_samples, num_channels = X_train_filtered_final.shape
    X_train_reshaped = X_train_filtered_final.reshape(-1, num_channels)

    num_test_trials, _, _ = X_test_filtered_final.shape
    X_test_reshaped = X_test_filtered_final.reshape(-1, num_channels)

    scaler = StandardScaler()
    scaler.fit(X_train_reshaped)

    X_train_scaled_reshaped = scaler.transform(X_train_reshaped)
    X_test_scaled_reshaped = scaler.transform(X_test_reshaped)

    X_train_processed = X_train_scaled_reshaped.reshape(num_train_trials, num_samples, num_channels)
    X_test_processed = X_test_scaled_reshaped.reshape(num_test_trials, num_samples, num_channels)

    print("--- Preprocessing Complete ---")
    print(f"Shape of X_train_processed: {X_train_processed.shape}")
    print(f"Shape of y_train: {y_train.shape}")
    print(f"Shape of X_test_processed: {X_test_processed.shape}")
    print(f"Shape of y_test: {y_test.shape}")

    return X_train_processed, X_test_processed, y_train, y_test, scaler

# --- 3. Main Execution Block ---
if __name__ == '__main__':
    DATA_PATH = 'data/'

    epochs_array, y, sampling_freq = load_aggregate_data(DATA_PATH)

    if epochs_array is not None:
        X_train_processed, X_test_processed, y_train, y_test, scaler = preprocess_data(epochs_array, y, sampling_freq)

        print("\n--- Preprocessing Pipeline Finished ---")
        print("Preprocessed training and testing data (X_train_processed, X_test_processed) are ready.")
        print("Labels (y_train, y_test) are also ready.")

        try:
            # Uncomment if you want to save arrays
            # np.save('X_train_processed.npy', X_train_processed)
            # np.save('X_test_processed.npy', X_test_processed)
            # np.save('y_train.npy', y_train)
            # np.save('y_test.npy', y_test)
            scaler_filename = 'eeg_scaler_multisubject.joblib'
            joblib.dump(scaler, scaler_filename)
            print(f"Scaler saved successfully as {scaler_filename}")
        except Exception as e:
            print(f"Could not save processed data/scaler: {e}")
    else:
        print("\nPreprocessing pipeline could not complete due to data loading errors.")
