from typing import List

from src.components import base
from src.functions import download_data
from src.formats import RequestDownloadData, ResponseDownloadData, ResponseMessage

class ComponentType(base.ComponentType):
    pass    

class Component(base.Component):
    def init(self, **config):
        self.config = ComponentType(**config)

    def call(self, message: RequestDownloadData,
             *, upstream_events: List[ResponseMessage] = []) -> ResponseDownloadData:
        return download_data(message)