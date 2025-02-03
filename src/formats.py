from pydantic import BaseModel
from typing import List, Dict, Union, Optional


class RequestMessage(BaseModel):
    pass

class ResponseMessage(BaseModel):
    pass

class RequestDownloadData(RequestMessage):
    url: str
    local_path: str

class ResponseDownloadData(RequestDownloadData):
    status: str
    
class RequestExtractInfo(RequestMessage):
    local_path: str
    output_path: str
    
class ResponseExtractInfo(RequestExtractInfo):
    status: str
    
class RequestPreprocessData(RequestMessage):
    local_path: str
    output_path: str
    target_columns: Optional[List[str]] = None
    outlier: Optional[List[Dict[str, Union[str, int, list]]]] = None
    missing: Optional[List[Dict[str, Union[str, int, list]]]] = None

class ResponsePreprocessData(RequestPreprocessData):
    status: str