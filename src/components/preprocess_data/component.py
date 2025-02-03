from typing import List

from src.components import base
from src.functions import preprocess_data
from src.formats import RequestPreprocessData, ResponsePreprocessData, ResponseMessage

class ComponentType(base.ComponentType):
    pass    

class Component(base.Component):
    def init(self, **config):
        self.config = ComponentType(**config)

    def call(self, message: RequestPreprocessData,
             *, upstream_events: List[ResponseMessage] = []) -> ResponsePreprocessData:
        if upstream_events:
            print("# [INFO] upstream_events: ", upstream_events)
            # TODO
        return preprocess_data(message)