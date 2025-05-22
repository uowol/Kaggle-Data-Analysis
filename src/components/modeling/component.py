import os
import yaml
from pathlib import Path
from typing import List

from src.components import base
from src.models.base import predict as BasePredict
from src.models.title_prediction import predict as TitlePredictionPredict
from src.formats import RequestModeling, ResponseModeling, ResponseMessage

class ComponentType(base.ComponentType):
    pass    

class Component(base.Component):
    def init(self, **config):
        if config:
            self.config = ComponentType(**config)
        else:
            with open(os.path.join(Path(__file__).parent, "component.yaml"), "r") as fp:
                self.config = yaml.safe_load(fp)
                self.config = self.config if self.config is not None else {}

    def call(self, message: RequestModeling,
             *, upstream_events: List[ResponseMessage] = []) -> ResponseModeling:
        if upstream_events:
            print("# [INFO] upstream_events: ", upstream_events)
            
        if message.model_type == "title_prediction":
            return TitlePredictionPredict(message, self.config[message.model_type]['params'])
        
        return BasePredict(message)