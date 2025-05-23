import os
import pandas as pd
from pathlib import Path
from src.formats import (
    RequestModeling, ResponseModeling
)
from sklearn.linear_model import LogisticRegression


def predict(message: RequestModeling, params: dict = None) -> ResponseModeling:
    os.makedirs(Path(message.output_filepath).parent, exist_ok=True)
    assert os.path.exists(message.train_filepath), f"Train file {message.train_filepath} does not exist"
    assert os.path.exists(message.test_filepath), f"Test file {message.test_filepath} does not exist"
    
    train = pd.read_csv(message.train_filepath)
    X_train, y_train = train.drop(columns=['Survived']), train['Survived']

    # PassengerId와 Name 모두 고유값이나, PassengerId는 단순한 인덱스이므로 제거
    X_train = X_train.drop(columns=['PassengerId'], axis=1)
    
    # Survived ~ Pclass + Sex
    X_train = X_train[['Pclass', 'Sex']]
    X_train = pd.get_dummies(X_train, columns=['Pclass', 'Sex'], drop_first=True)
    
    model = LogisticRegression(
        penalty=params.get('penalty', 'l2'),
        C=params.get('C', 1.0),
        max_iter=params.get('max_iter', 100),
        solver=params.get('solver', 'lbfgs'),
        class_weight=params.get('class_weight', None),
        random_state=params.get('random_state', 42),
    )
    model.fit(X_train, y_train)

    X_test = pd.read_csv(message.test_filepath)
    X_test_PassengerId = X_test['PassengerId']
    X_test = X_test[['Pclass', 'Sex']]
    X_test = pd.get_dummies(X_test, columns=['Pclass', 'Sex'], drop_first=True)
    y_pred = model.predict(X_test)
    response = pd.DataFrame({
        'PassengerId': X_test_PassengerId,
        'Survived': y_pred
    })
    response.to_csv(os.path.join(message.output_filepath), index=False)

    return ResponseModeling(
        status="success",
        **message.model_dump()
    )