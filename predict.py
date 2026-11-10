"""Batch CSV -> one research prediction per instructor."""
import argparse
import pandas as pd
from effectiveness import predict_batches

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--model', default='models/instructor_model.pkl')
    parser.add_argument('--output', default='outputs/new_predictions.csv')
    args = parser.parse_args()
    result = predict_batches(pd.read_csv(args.input), args.model)
    result.to_csv(args.output)
    print(result.to_string())
