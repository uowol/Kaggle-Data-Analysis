from pydantic import BaseModel
from typing import List, Dict, Union, Optional


class RequestMessage(BaseModel):
    pass

class ResponseMessage(BaseModel):
    status: str


class RequestDownloadData(RequestMessage):
    url: str
    local_path: str
    is_competition: bool = False

class ResponseDownloadData(ResponseMessage, RequestDownloadData):
    pass 
    

class RequestExtractInfo(RequestMessage):
    local_path: str
    output_path: str
    
class ResponseExtractInfo(ResponseMessage, RequestExtractInfo):
    pass