import os
import numpy as np
import pandas as pd
import logging
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

import sys
sys.path.append("kaggle_projects/")

from titanic.src.formats import (
    RequestModeling, ResponseModeling
)


def predict(message: RequestModeling, params: dict = {}) -> ResponseModeling:
    logger = get_logger()
    
    base_dir = Path(__file__).resolve().parent.parent.parent.parent
    output_filepath = base_dir / message.output_filepath
    train_filepath = base_dir / message.train_filepath
    test_filepath = base_dir / message.test_filepath

    os.makedirs(output_filepath.parent, exist_ok=True)
    assert os.path.exists(train_filepath), f"Train file {train_filepath} does not exist"
    assert os.path.exists(test_filepath), f"Test file {test_filepath} does not exist"
    
    train = pd.read_csv(train_filepath)
    X_train, y_train = train.drop(columns=['Survived']), train['Survived']

    # PassengerId와 Name 모두 고유값이나, PassengerId는 단순한 인덱스이므로 제거
    X_train = X_train.drop(columns=['PassengerId'], axis=1)

    # Age는 연속형 변수이지만 범주별로 나타나는 경향이 다르기 때문에 범주형 변수로 변환
    # 결측값은 이름의 title을 이용하여 채우고, title이 Unseen인 경우가 있을 수 있으므로 이 경우 Pclass의 평균값으로 채운다.
    X_train['Title'] = X_train['Name'].str.split(',', expand=True)[1].str.split('.', expand=True)[0].str.strip()
    X_train['Age'].fillna(X_train.groupby('Title').Age.transform('mean'), inplace=True)
    X_train['Age'].fillna(X_train.groupby('Pclass').Age.transform('mean'), inplace=True)
    # age_bins = [0, 5, 10, 15, 20, 30, 40, 50, 60, 100]
    # age_labels = ['0-5', '5-10', '10-15', '15-20', '20-30', '30-40', '40-50', '50-60', '60+']
    # X_train['AgeGroup'] = pd.cut(X_train['Age'], bins=age_bins, labels=age_labels)
    age_bins, X_train['AgeGroup'] = tree_bin_continuous_feature(X_train, y_train, 'Age', max_leaf_nodes=6)
    
    # SibSp와 Parch의 경우 데이터 수가 적은 케이스를 하나로 병합
    # X_train['SibSp'] = X_train['SibSp'].replace([x for x in range(3, 11)], 3)
    # X_train['Parch'] = X_train['Parch'].replace([x for x in range(3, 11)], 3)

    # FamilySize는 SibSp와 Parch를 합친 값으로, 가족의 크기를 나타내는 변수
    X_train['FamilySize'] = X_train['SibSp'] + X_train['Parch']
    X_train['FamilySize'] = X_train['FamilySize'].replace([x for x in range(4, 11)], 4)
    
    # Ticket은 객실, 동행자, 요금, 어디서 탑승했는지에 대한 정보를 포함, 샘플이 너무 적은 경우는 Unknown으로 대체
    X_train['TicketCleansed'] = X_train.Ticket.str.replace('.', '').str.upper()
    X_train['TicketHeader'] = X_train['TicketCleansed'].apply(
        lambda x: x.split(' ')[0] if len(x.split(' ')) > 1 else 'Unknown'
    )
    header_counts = X_train['TicketHeader'].value_counts()
    rare_headers = header_counts[header_counts < 5].index
    X_train['TicketHeader'] = X_train['TicketHeader'].replace(list(rare_headers), 'Unknown')
    known_headers = set(X_train['TicketHeader'].unique())

    # Age^2
    # X_train['Age^2'] = X_train['Age'] ** 2
    # Age^3
    # X_train['Age^3'] = X_train['Age'] ** 3
    
    # Survived ~ Pclass + Sex + Age(categorical) + FamilySize + TicketHeader
    num_cols = []
    cat_cols = ['Pclass', 'Sex', 'AgeGroup', 'FamilySize', 'TicketHeader']

    # Survived ~ Pclass + Sex + Age(continuous) + FamilySize + TicketHeader
    # num_cols = ['Age', 'Age^2', 'Age^3']
    # cat_cols = ['Pclass', 'Sex', 'FamilySize', 'TicketHeader']
        
    X_train_dummies = pd.get_dummies(X_train[cat_cols], columns=cat_cols, drop_first=True)
    dummy_cat_cols = X_train_dummies.columns.tolist()
    X_train = pd.concat([X_train[num_cols], X_train_dummies], axis=1)

    # 가중치 부여
    pclass_cols = [col for col in X_train.columns if col.startswith('Pclass_')]
    sex_cols = [col for col in X_train.columns if col.startswith('Sex_')]
    age_cols = [col for col in X_train.columns if col.startswith('AgeGroup_')]
    fam_cols = [col for col in X_train.columns if col.startswith('FamilySize_')]
    tick_cols = [col for col in X_train.columns if col.startswith('TicketHeader_')]
    # X_train[pclass_cols] = X_train[pclass_cols] * 0.2

    preprocess = ColumnTransformer(
        [('num', StandardScaler(), num_cols),
        ('cat', 'passthrough', dummy_cat_cols)]
    )

    model = Pipeline([
        ('prep', preprocess),
        ('lr', LogisticRegression(
            penalty=params.get('penalty', 'l2'),
            C=params.get('C', 1.0),
            max_iter=params.get('max_iter', 100),
            solver=params.get('solver', 'lbfgs'),
            class_weight=params.get('class_weight', None),
            random_state=params.get('random_state', 42),
            verbose=True
        ))
    ])
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=params.get('random_state', 42))
    y_prob = cross_val_predict(model, X_train, y_train, cv=skf, method='predict_proba')
    y_pred = y_prob.argmax(axis=1)
    residual = y_prob[:, 0] - y_train
    cv_result_df = pd.DataFrame({
        'Target': y_train,
        'Predict': y_prob[:, 0],
        'Residual': residual,
        'Diff(Abs)': np.abs(residual)
    })
    cv_result_df.sort_values(by='Diff(Abs)', ascending=False, inplace=True)
    logger.info("Cross-validation results:\n%s", cv_result_df.head(30))
    logger.info("CV Accuracy: %.4f", accuracy_score(y_train, y_pred))
    
    model.fit(X_train, y_train)
    importance = pd.Series(
        model.named_steps['lr'].coef_[0],  # 이진 분류이므로 첫 번째 클래스의 계수
        index=X_train.columns
    ).sort_values(ascending=False)
    logger.info("Feature Importances:\n%s", importance)

    X_test = pd.read_csv(test_filepath)
    X_test_PassengerId = X_test['PassengerId']
    X_test['Title'] = X_test['Name'].str.split(',', expand=True)[1].str.split('.', expand=True)[0].str.strip()
    X_test['Age'].fillna(X_test.groupby('Title').Age.transform('mean'), inplace=True)
    X_test['Age'].fillna(X_test.groupby('Pclass').Age.transform('mean'), inplace=True)
    X_test['AgeGroup'] = pd.cut(X_test['Age'], bins=age_bins)
    # X_test['SibSp'] = X_test['SibSp'].replace([x for x in range(3, 11)], 3)
    # X_test['Parch'] = X_test['Parch'].replace([x for x in range(3, 11)], 3)
    X_test['FamilySize'] = X_test['SibSp'] + X_test['Parch']
    X_test['FamilySize'] = X_test['FamilySize'].replace([x for x in range(4, 11)], 4)
    X_test['TicketCleansed'] = X_test.Ticket.str.replace('.', '').str.upper()
    X_test['TicketHeader'] = X_test['TicketCleansed'].apply(
        lambda x: x.split(' ')[0] if len(x.split(' ')) > 1 else 'Unknown'
    )
    X_test['TicketHeader'] = X_test['TicketHeader'].apply(
        lambda x: x if x in known_headers else 'Unknown'
    )
    X_test['Age^2'] = X_test['Age'] ** 2
    X_test['Age^3'] = X_test['Age'] ** 3
    
    X_test_dummies = pd.get_dummies(X_test[cat_cols], columns=cat_cols, drop_first=True)
    X_test = pd.concat([X_test[num_cols], X_test_dummies], axis=1)
    y_pred = model.predict(X_test)
    response = pd.DataFrame({
        'PassengerId': X_test_PassengerId,
        'Survived': y_pred
    })
    response.to_csv(output_filepath, index=False)

    return ResponseModeling(
        status="success",
        **message.model_dump()
    )


def tree_bin_continuous_feature(X, y, feature_name, max_leaf_nodes=6):
    tree = DecisionTreeClassifier(max_leaf_nodes=max_leaf_nodes)
    X_ = X[[feature_name]].copy()
    tree.fit(X_, y)
    thresholds = tree.tree_.threshold
    thresholds = thresholds[thresholds != -2]  # -2는 리프 노드
    bins = [-np.inf] + sorted(thresholds.tolist()) + [np.inf]
    return bins, pd.cut(X[feature_name], bins=bins, include_lowest=True)


def get_logger():

    formatter = logging.Formatter("%(message)s")

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    os.makedirs("/tmp", exist_ok=True)

    file_handler = logging.FileHandler("/tmp/predict.log", mode="w")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger
