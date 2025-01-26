from typing import Optional

from src.pipelines import base
from src.formats import (
    RequestDownloadData, ResponseDownloadData,
    RequestExtractInfo, ResponseExtractInfo,
    RequestPreprocessData, ResponsePreprocessData,
)
from src.components.download_data.component import Component as DownloadDataComponent
from src.components.extract_data_info.component import Component as ExtractDataComponent
# from src.components.preprocess_data.component import Component as PreprocessDataComponent

class PipelineType(base.PipelineType):
    download_data: Optional[RequestDownloadData] = None
    extract_data: Optional[RequestExtractInfo] = None
    preprocess_data: Optional[RequestPreprocessData] = None    

class Pipeline(base.Pipeline):
    def init(self, **config):
        self.config = PipelineType(**config)
        
    def call(self):
        upstream_events = []
        if self.config.download_data is not None:
            print("\n# [INFO] =============== download_data start ===============")
            component = DownloadDataComponent()
            request_message = self.config.download_data
            print("# [INFO] request_message: ", request_message)
            response_message = component(request_message)
            print("# [INFO] response_message: ", response_message)
            upstream_events.append(response_message)
            print("# [INFO] =============== download_data end ===============")
        if self.config.extract_data is not None:
            print("\n# [INFO] =============== extract_data start ===============")
            component = ExtractDataComponent()
            request_message = self.config.extract_data
            print("# [INFO] request_message: ", request_message)
            response_message = component(request_message, upstream_events=upstream_events)
            print("# [INFO] response_message: ", response_message)
            upstream_events.append(response_message)
            print("# [INFO] =============== extract_data end ===============")
        # if self.config.preprocess_data is not None:
        #     component = preprocess_data.Component()
        #     request_message = self.config.preprocess_data
        #     response_message = component(request_message, upstream_events=upstream_events)
        #     upstream_events.append(response_message)

        return