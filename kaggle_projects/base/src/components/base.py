from abc import ABC, abstractmethod
from pydantic import BaseModel

import sys
sys.path.append("kaggle_projects/base/")

from src.formats import RequestMessage, ResponseMessage


class ComponentType(BaseModel):
    pass


class Component(ABC):
    def __init__(self, **config):
        self.init(**config)
    
    def __call__(self, request: RequestMessage, *args, **kwargs) -> ResponseMessage:
        return self.call(request, *args, **kwargs)
    
    @abstractmethod
    def init(self, **config):
        pass
    
    @abstractmethod
    def call(self, message: RequestMessage, *args, **kwargs) -> ResponseMessage:
        pass