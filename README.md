# kaggle_titanic
Source code for Kaggle Titanic Competition

# 1. 데이터 분석
- Survied 포함 총 12개의 Variable (제외하고는 11개의 variable)
- Survived 변수는 정답 label로써
  - train set에는 포함
  - test set에는 미포함

> train_df.columns
```
    Index(['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp',
       'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'],
      dtype='object')
```

- 그냥 쳐다봤을 때 관련이 없어 보이는 변수들이 존재함
- 이 변수들을 삭제하고 NumPy 배열로 변환
```
    train = train_df.drop(["Ticket", "Fare", "Cabin", "PassengerId", "Name"], axis=1, )
    test = test_df.drop(["Ticket", "Fare", "Cabin", "PassengerId", "Name"], axis=1, )
```