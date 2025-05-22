import os
import time
import glob
import pandas as pd
from src.formats import (
    RequestModeling, ResponseModeling
)


def predict(message: RequestModeling) -> ResponseModeling:
    os.makedirs(message.output_filepath, exist_ok=True)
    assert os.path.exists(message.train_filepath), f"Train file {message.train_filepath} does not exist"
    assert os.path.exists(message.test_filepath), f"Test file {message.test_filepath} does not exist"
    
    # TBD
    
    return ResponseModeling(
        status="success",
        **message.model_dump()
    )