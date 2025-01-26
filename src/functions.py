import os
import zipfile
import subprocess
import ydata_profiling
import time
import glob
import pandas as pd

from src.formats import (
    RequestDownloadData, ResponseDownloadData,
    RequestExtractInfo, ResponseExtractInfo,
    RequestPreprocessData, ResponsePreprocessData,
)


def download_data(message: RequestDownloadData) -> ResponseDownloadData:
    subprocess.run(["kaggle", "datasets", "download", message.url, "-p", message.local_path])
    local_zip_files = glob.glob(f"{message.local_path}/*.zip")
    for file_path in local_zip_files:
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(message.local_path)
        subprocess.run(["rm", file_path])
    return ResponseDownloadData(
        status="success",
        local_path=message.local_path
    )
    
    
def extract_data_info(message: RequestExtractInfo) -> ResponseExtractInfo:
    local_csv_files = glob.glob(f"{message.local_path}/*.csv")
    for file_path in local_csv_files:
        df = pd.read_csv(file_path)
        profile = ydata_profiling.ProfileReport(df)
        profile.to_file(os.path.join(message.output_path, f"{os.path.basename(file_path).replace(".csv", '')}_profile.html"))
        json_data = profile.to_json()
        with open(os.path.join(message.output_path, f"{os.path.basename(file_path).replace(".csv", '')}_profile.json"), "w") as f:
            f.write(json_data)
    return ResponseExtractInfo(
        status="success",
        output_path=message.output_path
    )   