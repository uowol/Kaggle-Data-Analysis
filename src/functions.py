import os
import zipfile
import subprocess

from src.formats import (
    RequestDownloadData, ResponseDownloadData,
    RequestExtractInfo, ResponseExtractInfo,
    RequestPreprocessData, ResponsePreprocessData,
)


def download_data(message: RequestDownloadData) -> ResponseDownloadData:
    subprocess.run(["kaggle", "datasets", "download", message.url, "-p", message.local_path])
    for file_name in os.listdir(message.local_path):
        if file_name.endswith(".zip"):
            with zipfile.ZipFile(os.path.join(message.local_path, file_name), 'r') as zip_ref:
                zip_ref.extractall(message.local_path)
            subprocess.run(["rm", f"{message.local_path}/{file_name}"])
    return ResponseDownloadData(
        status="success",
        local_path=message.local_path
    )