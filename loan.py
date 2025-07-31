import pandas as pd 
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib


data=pd.read_csv("C:/Users/Subaina/OneDrive - Higher Education Commission/Documents/Internship/Elevvo/Loan-approval-Prediction/loan_approval_dataset.csv")

# print(data)
x = data.drop(columns=[' loan_status','loan_id']) 
y=data[' loan_status']

# label encode to convert string to number
label=LabelEncoder()
y = label.fit_transform(y)
x[' education'] = label.fit_transform(x[' education'])
x[' self_employed'] = label.fit_transform(x[' self_employed'])

# scaling

numeric_cols = [
    ' no_of_dependents', ' income_annum', ' loan_amount', ' loan_term',
    ' cibil_score', ' residential_assets_value', ' commercial_assets_value', ' luxury_assets_value', ' bank_asset_value'
]
xTrain, xTest, yTrain, yTest = train_test_split(
    x, y, test_size=0.2, random_state=42
)

# Define preprocessing
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numeric_cols)
], remainder='passthrough')

pipeline = Pipeline([
    ('preprocessing', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

pipeline.fit(xTrain,yTrain)

accuracy=pipeline.score(xTest,yTest)
print(accuracy)

y_pred = pipeline.predict(xTest)

# Print classification report
print(classification_report(yTest, y_pred, target_names=label.classes_))

joblib.dump(pipeline, 'loan_model.pkl')