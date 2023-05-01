import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

def read_data(file_path):
    data = pd.read_csv(file_path)
    return data

def train_model(X_train, y_train):
    clf = Pipeline([
        ('clf', RandomForestClassifier())
    ])

    clf.fit(X_train, y_train)
    return clf

def main():
    parser = argparse.ArgumentParser(description="Train a classifier on sample data")
    parser.add_argument("--data", type=str, help="Path to the input data file", required=True)
    parser.add_argument("--test_size", type=float, default=0.2, help="Size of the test set as a fraction (default: 0.2)")

    args = parser.parse_args()

    data = read_data(args.data)
    X = data.drop("target", axis=1)
    y = data["target"]

    le = LabelEncoder()
    y = le.fit_transform(y.astype(str))  # Chuyển đổi nhãn sang kiểu chuỗi

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, random_state=42)

    clf = train_model(X_train, y_train)
    y_pred = clf.predict(X_test)

    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

if __name__ == "__main__":
    main()
