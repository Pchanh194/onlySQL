import pandas as pd
import numpy as np
import argparse

def generate_data(num_samples, num_features, num_classes, random_seed):
    np.random.seed(random_seed)
    X = np.random.rand(num_samples, num_features)
    y = np.random.randint(0, num_classes, size=(num_samples, 1))

    data = np.hstack((X, y))
    return data

def save_data(data, file_path):
    columns = [f"feature_{i}" for i in range(data.shape[1] - 1)] + ["target"]
    df = pd.DataFrame(data, columns=columns)
    df.to_csv(file_path, index=False)

def main():
    parser = argparse.ArgumentParser(description="Generate sample data for classification")
    parser.add_argument("--num_samples", type=int, default=1000, help="Number of samples (default: 1000)")
    parser.add_argument("--num_features", type=int, default=4, help="Number of features (default: 4)")
    parser.add_argument("--num_classes", type=int, default=3, help="Number of classes (default: 3)")
    parser.add_argument("--output", type=str, help="Path to save the output CSV file", required=True)
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed (default: 42)")

    args = parser.parse_args()

    data = generate_data(args.num_samples, args.num_features, args.num_classes, args.random_seed)
    save_data(data, args.output)

if __name__ == "__main__":
    main()
