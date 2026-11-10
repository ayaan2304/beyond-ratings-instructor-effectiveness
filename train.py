"""Train the reproducible experiment and research model."""
import argparse
import json
import pandas as pd
from effectiveness import train_experiment

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/instructor_batches.csv')
    args = parser.parse_args()
    result, _, _, _ = train_experiment(pd.read_csv(args.data))
    print(json.dumps(result, indent=2))
