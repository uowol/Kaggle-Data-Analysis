import os
import yaml
from pathlib import Path
from typing import List

if __name__ == "__main__":    
    # Test the component
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))
from src.components import base
from src.models import (
    base as base_model,
    title_prediction,
    logistic_regression,
    decision_tree,
)
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
            return title_prediction.predict(message, self.config[message.model_type]['params'])
        if message.model_type == 'logistic_regression':
            return logistic_regression.predict(message, self.config[message.model_type]['params'])
        if message.model_type == 'decision_tree':
            return decision_tree.predict(message, self.config[message.model_type]['params'])
        
        return base_model.predict(message)


if __name__ == "__main__":    
    # Test the component
    message = RequestModeling(
        train_filepath="data/titanic/raw/train.csv",
        test_filepath="data/titanic/raw/test.csv",
        output_filepath="data/titanic/raw/output.csv",
        model_type="decision_tree",
    )
    component = Component()
    response = component(message)
    print(response)