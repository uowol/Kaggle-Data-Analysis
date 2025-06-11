class RequestModeling(RequestMessage):
    train_filepath: str
    test_filepath: str
    output_filepath: str
    model_type: str

class ResponseModeling(ResponseMessage, RequestModeling):
    pass