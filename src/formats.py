from pydantic import BaseModel


class RequestMessage(BaseModel):
    pass

class ResponseMessage(BaseModel):
    pass

class RequestDownloadData(RequestMessage):
    url: str
    local_path: str

class ResponseDownloadData(ResponseMessage):
    status: str
    local_path: str
    
class RequestExtractInfo(RequestMessage):
    local_path: str
    output_path: str
    
class ResponseExtractInfo(ResponseMessage):
    status: str
    output_path: str
    
class RequestPreprocessData(RequestMessage):
    local_path: str
    output_path: str

class ResponsePreprocessData(ResponseMessage):
    status: str
    output_path: str