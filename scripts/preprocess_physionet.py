import os
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
from collections import defaultdict

VARIABLES = [
    'Albumin', 'ALP', 'ALT', 'AST', 'Bilirubin', 'BUN', 'Cholesterol',
    'Creatinine', 'DiasABP', 'FiO2', 'GCS', 'Glucose', 'HCO3', 'HCT',
    'HR', 'K', 'Lactate', 'Mg', 'MAP', 'MechVent', 'Na', 'NIDiasABP',
    'NIMAP', 'NISysABP', 'PaCO2', 'PaO2', 'pH', 'Platelets', 'RespRate',
    'SaO2', 'SysABP', 'Temp', 'TroponinI', 'TroponinT', 'Urine', 'WBC', 'Weight'
]

VAR_TO_IDX = {var: idx for idx, var in enumerate(VARIABLES)}


def parse_time(time_str):
    parts = time_str.split(':')
    return int(parts[0]) + int(parts[1]) / 60.0


def load_record(filepath):
    df = pd.read_csv(filepath)
    record_id = None
    data = defaultdict(list)

    for _, row in df.iterrows():
        time = parse_time(row['Time'])
        param = row['Parameter']
        value = row['Value']

        if param == 'RecordID':
            record_id = int(value)
        elif param in VAR_TO_IDX:
            if value != -1:
                data[param].append((time, value))

    return record_id, data


def create_time_series(data, max_hours=48, time_step=1.0):
    n_features = len(VARIABLES)
    n_steps = int(max_hours / time_step)

    input_arr = np.full((n_steps, n_features), np.nan)
    mask_arr = np.zeros((n_steps, n_features))
    timestamp_arr = np.arange(n_steps) * time_step

    for var, measurements in data.items():
        if var not in VAR_TO_IDX:
            continue
        var_idx = VAR_TO_IDX[var]

        for time, value in measurements:
            if time >= max_hours:
                continue
            step_idx = int(time / time_step)
            if step_idx < n_steps:
                input_arr[step_idx, var_idx] = value
                mask_arr[step_idx, var_idx] = 1.0

    return input_arr, mask_arr, timestamp_arr


def preprocess_physionet(data_dir, outcomes_file, output_file, max_hours=48, time_step=1.0):
    outcomes_df = pd.read_csv(outcomes_file)
    outcomes_dict = dict(zip(outcomes_df['RecordID'], outcomes_df['In-hospital_death']))

    record_files = sorted(glob(os.path.join(data_dir, '*.txt')))
    print(f"Found {len(record_files)} records")

    inputs = []
    masks = []
    timestamps = []
    labels = []
    record_ids = []

    for filepath in tqdm(record_files, desc="Processing records"):
        try:
            record_id, data = load_record(filepath)

            if record_id is None or record_id not in outcomes_dict:
                continue

            input_arr, mask_arr, timestamp_arr = create_time_series(
                data, max_hours=max_hours, time_step=time_step
            )

            if mask_arr.sum() == 0:
                continue

            inputs.append(input_arr)
            masks.append(mask_arr)
            timestamps.append(timestamp_arr)
            labels.append(outcomes_dict[record_id])
            record_ids.append(record_id)

        except Exception as e:
            print(f"Error processing {filepath}: {e}")
            continue

    inputs = np.array(inputs, dtype=object)
    masks = np.array(masks, dtype=object)
    timestamps = np.array(timestamps, dtype=object)
    labels = np.array(labels, dtype=np.int32)

    n_features = inputs[0].shape[1]
    feature_means = np.zeros(n_features)
    feature_stds = np.zeros(n_features)

    for f in range(n_features):
        f_vals = []
        for i, inp in enumerate(inputs):
            vals = inp[:, f][masks[i][:, f] == 1]
            f_vals.extend(vals)
        if len(f_vals) > 0:
            f_vals = np.array(f_vals)
            feature_means[f] = np.mean(f_vals)
            feature_stds[f] = np.std(f_vals)
            if feature_stds[f] < 1e-6:
                feature_stds[f] = 1.0

    print(f"Feature means: min={feature_means.min():.2f}, max={feature_means.max():.2f}")
    print(f"Feature stds: min={feature_stds.min():.2f}, max={feature_stds.max():.2f}")

    for i in range(len(inputs)):
        inputs[i] = (inputs[i] - feature_means) / feature_stds

    label_arr = np.zeros((len(labels), 2), dtype=np.int32)
    label_arr[:, 0] = 1 - labels
    label_arr[:, 1] = labels

    np.savez(
        output_file,
        input=inputs,
        masking=masks,
        timestamp=timestamps,
        label_mortality=label_arr,
        record_ids=record_ids,
        variables=VARIABLES,
        feature_means=feature_means,
        feature_stds=feature_stds
    )

    print(f"\nSaved {len(inputs)} records to {output_file}")
    print(f"Label distribution: {np.bincount(labels)}")
    print(f"Mortality rate: {labels.mean()*100:.1f}%")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--outcomes', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--max_hours', type=float, default=48)
    parser.add_argument('--time_step', type=float, default=1.0)
    args = parser.parse_args()

    preprocess_physionet(
        args.data_dir,
        args.outcomes,
        args.output,
        args.max_hours,
        args.time_step
    )
