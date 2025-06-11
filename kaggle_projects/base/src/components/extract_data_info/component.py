import os
import yaml
import logging
from pathlib import Path
from typing import List

import sys
sys.path.append("kaggle_projects/base/")

from src.components import base
from src.functions import extract_data_info
from src.formats import RequestExtractInfo, ResponseExtractInfo, ResponseMessage

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

    def call(self, message: RequestExtractInfo,
             *, 
             upstream_events: List[ResponseMessage] = [], 
             logger: logging.Logger = logging.getLogger(name="default")) -> ResponseExtractInfo:
        if upstream_events:
            logger.info("upstream_events: %s", upstream_events)
        return extract_data_info(message)