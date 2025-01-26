from typing import List

from src.components import base
from src.functions import extract_data_info
from src.formats import RequestExtractInfo, ResponseExtractInfo, ResponseMessage

class ComponentType(base.ComponentType):
    pass    

class Component(base.Component):
    def init(self, **config):
        self.config = ComponentType(**config)

    def call(self, message: RequestExtractInfo,
             *, upstream_events: List[ResponseMessage] = []) -> ResponseExtractInfo:
        if upstream_events:
            print("# [INFO] upstream_events: ", upstream_events)
            # TODO
        return extract_data_info(message)