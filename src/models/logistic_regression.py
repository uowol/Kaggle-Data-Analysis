import os
import numpy as np
import pandas as pd
from pathlib import Path
from src.formats import (
    RequestModeling, ResponseModeling
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold


def predict(message: RequestModeling, params: dict = {}) -> ResponseModeling:
    os.makedirs(Path(message.output_filepath).parent, exist_ok=True)
    assert os.path.exists(message.train_filepath), f"Train file {message.train_filepath} does not exist"
    assert os.path.exists(message.test_filepath), f"Test file {message.test_filepath} does not exist"
    
    train = pd.read_csv(message.train_filepath)
    X_train, y_train = train.drop(columns=['Survived']), train['Survived']

    # PassengerId와 Name 모두 고유값이나, PassengerId는 단순한 인덱스이므로 제거
    X_train = X_train.drop(columns=['PassengerId'], axis=1)

    # Age는 연속형 변수이지만 범주별로 나타나는 경향이 다르기 때문에 범주형 변수로 변환
    # 결측값은 이름의 title을 이용하여 채우고, title이 Unseen인 경우가 있을 수 있으므로 이 경우 Pclass의 평균값으로 채운다.
    X_train['Title'] = X_train['Name'].str.split(',', expand=True)[1].str.split('.', expand=True)[0].str.strip()
    X_train['Age'].fillna(X_train.groupby('Title').Age.transform('median'), inplace=True)
    X_train['Age'].fillna(X_train.groupby('Pclass').Age.transform('median'), inplace=True)
    age_bins = [0, 10, 20, 30, 40, 50, 60, 100]
    age_labels = ['0-10', '10-20', '20-30', '30-40', '40-50', '50-60', '60+']
    X_train['AgeGroup'] = pd.cut(X_train['Age'], bins=age_bins, labels=age_labels)
    
    # Survived ~ Pclass + Sex + Age(categorical)
    X_columns = ['Pclass', 'Sex', 'AgeGroup']
    
    X_train = X_train[X_columns]
    X_train = pd.get_dummies(X_train, columns=X_columns, drop_first=True)
    
    model = LogisticRegression(
        penalty=params.get('penalty', 'l2'),
        C=params.get('C', 1.0),
        max_iter=params.get('max_iter', 100),
        solver=params.get('solver', 'lbfgs'),
        class_weight=params.get('class_weight', None),
        random_state=params.get('random_state', 42),
        verbose=True
    )
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=params.get('random_state', 42))
    cv_scores = cross_val_score(model, X_train, y_train, cv=skf, scoring='accuracy')
    print(f"Cross-validation Accuracy Scores: {cv_scores}")
    print(f"Mean Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    model.fit(X_train, y_train)
    importance = pd.Series(
        model.coef_[0],  # 이진 분류이므로 첫 번째 클래스의 계수
        index=X_train.columns
    ).sort_values(ascending=False)
    print(importance)

    X_test = pd.read_csv(message.test_filepath)
    X_test_PassengerId = X_test['PassengerId']
    X_test['Title'] = X_test['Name'].str.split(',', expand=True)[1].str.split('.', expand=True)[0].str.strip()
    X_test['Age'].fillna(X_test.groupby('Title').Age.transform('median'), inplace=True)
    X_test['Age'].fillna(X_test.groupby('Pclass').Age.transform('median'), inplace=True)
    X_test['AgeGroup'] = pd.cut(X_test['Age'], bins=age_bins, labels=age_labels)
    X_test = X_test[X_columns]
    X_test = pd.get_dummies(X_test, columns=X_columns, drop_first=True)
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