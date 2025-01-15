import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def load_data(data_filepath: str, *, info: bool = False) -> pd.DataFrame:
    data = None

    if data_filepath.endswith(".csv"):
        data = pd.read_csv(data_filepath)
    elif data_filepath.endswith(".xlsx"):
        data = pd.read_excel(data_filepath)
    elif data_filepath.endswith(".json"):
        data = pd.read_json(data_filepath)
    else:
        raise ValueError("Unsupported file format.")

    if info:
        print("# [INFO] Data Columns")
        data.info()

    return data


def check_missing_values(data: pd.DataFrame, *, info: bool = False) -> bool:
    result_series = data.isna().sum()
    result = all(result_series == 0)

    if info:
        if result:
            print("# [INFO] There is no missing value in the dataset.")
        else:
            print(f"# [INFO] There are missing values in the dataset. {result_series}")

    return result


def check_key_column(data: pd.DataFrame, column: str, *, info: bool = False) -> bool:
    result = data[column].nunique() == len(data)

    if info:
        if result:
            print("# [INFO] The column is the key column.")
        else:
            print(
                f"# [INFO] The column is not the key column. {data[column].nunique()}/{len(data)}"
            )

    return result


def draw_distribution(
    data: pd.DataFrame,
    column: str,
    *,
    hue: str = None,
    shape: str = "hist",
    log_scale: bool = False,
    transpose: bool = False,
):
    plt.figure(figsize=(8, 2))
    if shape == "hist":
        if transpose:
            sns.histplot(data, y=column, hue=hue, bins=100, kde=False)
        else:
            sns.histplot(data, x=column, hue=hue, bins=100, kde=False)
    elif shape == "box":
        if transpose:
            sns.boxplot(data=data, y=hue, x=column)
        else:
            sns.boxplot(data=data, x=hue, y=column)
    if log_scale:
        if transpose:
            plt.xscale("log")
        else:
            plt.yscale("log")
    plt.title(f"Distribution of {column}")
    plt.show()


def extract_outliers(
    data: pd.DataFrame, column: str, threshold: float, *, info: bool = False
):
    lower_bound = data[column].quantile(threshold)
    upper_bound = data[column].quantile(1 - threshold)
    general_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    upper_data = data[data[column] > upper_bound]
    lower_data = data[data[column] < lower_bound]

    if info:
        print("# [INFO] Outliers")
        print(f"Lower Bound: {lower_bound}, Upper Bound: {upper_bound}")
        print(f"# of original data: {len(data)}")
        print(f"# of general data: {len(general_data)}")
        print(f"# of upper data: {len(upper_data)}")
        print(f"# of lower data: {len(lower_data)}")

    return general_data, upper_data, lower_data
