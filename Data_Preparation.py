import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, f1_score
import torch
import torch.nn as nn



def demographic_information(data):

      data = data.sort_values(["RID", "EXAMDATE"]).drop_duplicates(subset=["RID"])
      summary = data.groupby("DXGrp").agg(
          Subject=("RID", "nunique"),  # Count unique subjects
          Gender=("PTGENDER", lambda x: f"{(x == 1).sum()}/{(x == 2).sum()}"),  # Count of gender 1 and gender 2
          Age_mean=("AGE", "mean"), 
          Age_sd=("AGE", "std"),
          Educ_mean=("PTEDUC", "mean"),
          Educ_sd=("PTEDUC", "std")
      )

      summary["Age(mean±sd)"] = summary["Age_mean"].round(6).astype(str) + " ± " + summary["Age_sd"].round(6).astype(str)
      summary["Educ(mean±sd)"] = summary["Educ_mean"].round(6).astype(str) + " ± " + summary["Educ_sd"].round(6).astype(str)

      summary = summary[["Subject", "Gender", "Age(mean±sd)", "Educ(mean±sd)"]].reset_index()

      group_map = {1: "CN", 2: "EMCI", 3: "LMCI", 4: "AD"}
      summary["DXGrp"] = summary["DXGrp"].map(group_map)

      summary = summary.set_index("DXGrp").T

      display(summary)




def data_preparation(df):

    non_continuous_cols = [col for col in df.columns if df[col].dtype == object]
    continuous_cols = [col for col in df.columns if col not in non_continuous_cols and col != "EXAMDATE"]
    continuous_df = df[continuous_cols]

    ctx_columns = [col for col in continuous_df.columns if col.startswith('CTX_')]

    for col in [ "RID", "DXGrp"]:
        ctx_columns.append(col)

    modelling_cols = df[ctx_columns+ ["AGE", "VISCODE2", "EXAMDATE"]]
    unique_subjects = modelling_cols['RID'].unique()

    train_groups, test_groups = train_test_split(unique_subjects, test_size=0.2, random_state=42)

    train_data = modelling_cols[modelling_cols['RID'].isin(train_groups)]
    test_data = modelling_cols[modelling_cols['RID'].isin(test_groups)]

    # Splitting the data

    X_train, X_test, y_train, y_test = train_data.drop(['RID', 'DXGrp', 'AGE', 'VISCODE2', 'EXAMDATE' ], axis=1), test_data.drop(['RID', 'DXGrp', 'AGE', 'VISCODE2', 'EXAMDATE'], axis=1), train_data["DXGrp"], test_data["DXGrp"]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Converting to tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)

    return continuous_df, continuous_cols, non_continuous_cols, modelling_cols, unique_subjects, train_data, test_data,\
            X_train, X_test, y_train, y_test, X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor


def prepare_data_EMCI_vs_AD(data, encoded_data, target_column='DXGrp', id_column='RID', date_column='EXAMDATE', target_classes=[1, 4]):
    encoded_data = pd.DataFrame(encoded_data)
    combined_data = pd.concat([encoded_data, data.reset_index(drop=True)[[id_column, target_column, date_column]]], axis=1)

    prepared_data = combined_data.sort_values([id_column, date_column]).drop_duplicates(subset=[id_column])

    # Filtering for target classes
    prepared_data = prepared_data[prepared_data[target_column].isin(target_classes)]

    # Mapping target classes to binary labels
    class_mapping = {target_classes[0]: 0, target_classes[1]: 1}
    prepared_data[target_column] = prepared_data[target_column].map(class_mapping)

    # Splitting features and labels
    X = prepared_data.drop([id_column, target_column, date_column], axis=1).values
    y = prepared_data[target_column].values

    return X, y, prepared_data



def prepare_data_EMCI_vs_AD_other_clf(data,  target_column='DXGrp', id_column='RID', date_column='EXAMDATE', target_classes=[1, 4]):

    prepared_data = data.sort_values([id_column, date_column]).drop_duplicates(subset=[id_column])


    # Mapping target classes to binary labels
    prepared_data = prepared_data[prepared_data[target_column].isin(target_classes)]

    # Mapping target classes to binary labels
    class_mapping = {target_classes[0]: 0, target_classes[1]: 1}
    prepared_data[target_column] = prepared_data[target_column].map(class_mapping)

    # Splitting features and labels
    X = prepared_data.drop([ 'RID', 'DXGrp', 'AGE', 'VISCODE2', 'EXAMDATE'], axis=1).values
    y = prepared_data[target_column].values

    return X, y, prepared_data




def compute_clf_metrics(y_true, y_pred):
    f1 = f1_score(y_true, y_pred)
    balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_pred)

    return {
        'F1-Score': f1,
        'Balanced Accuracy': balanced_accuracy,
        'ROC-AUC': roc_auc
    }



def compute_violation_ratio_violation_gap(data, threshold=0.0):
    """
    This function computes the violation ratio and violation gap based on Pseudotime_Normalized values.
    A violation is defined as the current pseudotime decreasing beyond a given threshold compared to the previous pseudotime.

    Args:
    - data (pd.DataFrame): DataFrame containing 'RID', 'AGE', and 'Pseudotime_Normalized' columns.
    - threshold (float): A small threshold value to identify violations (e.g., 0.01 or 0.1).

    Returns:
    - violation_ratio (float): The ratio of the number of violations to the total number of comparisons.
    - violation_gap (float): The average magnitude of the violations.
    """

    data = data.sort_values(by=['RID', 'AGE']).reset_index(drop=True)

    data['violation_flag'] = pd.NA  # 0 for no violation, positive for violations
    data['violation_magnitude'] = pd.NA  # Magnitude of violation

    for rid in data['RID'].unique():
        group = data[data['RID'] == rid].copy()
        for i in range(1, len(group)):
            current_val = group.iloc[i]['Pseudotime_Normalized']
            previous_val = group.iloc[i - 1]['Pseudotime_Normalized']
            if current_val >= previous_val - threshold:
                data.loc[group.index[i], 'violation_flag'] = 0
            else:
                violation_amount = abs(current_val - previous_val)
                data.loc[group.index[i], 'violation_flag'] = 1  # Indicating a violation occurred
                data.loc[group.index[i], 'violation_magnitude'] = violation_amount

    violation_count = data['violation_flag'].sum(skipna=True)
    total_comparisons = data['violation_flag'].count()  # Counting non-null values

    violation_ratio = violation_count / total_comparisons if total_comparisons > 0 else 0

    # Calculating violation gap: average magnitude of violations
    violation_gap = data['violation_magnitude'].mean(skipna=True) if violation_count > 0 else 0

    return violation_ratio, violation_gap

def compute_violation_metrics(data_objects, thresholds):
    results = {name: {'vio_ratios': [], 'vio_gaps': []} for name in data_objects.keys()}

    for threshold in thresholds:
        for name, data in data_objects.items():
            vio_ratio, vio_gap = compute_violation_ratio_violation_gap(data, threshold)
            results[name]['vio_ratios'].append(vio_ratio)
            results[name]['vio_gaps'].append(vio_gap)

    return results


