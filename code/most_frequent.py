import argparse
import json

def predict(data, diction, freq):
    predictions = []
    for d in data:
        pred = {
            'id': d['id'],
            'prediction': ''
        }
        candids = diction[d['tokens'][d['acronym']]]
        highest_score = 0
        best = ''
        for candid in candids:
            if freq[candid] > highest_score:
                highest_score = freq[candid]
                best = candid
        if best == '':
            best = candids[0]
        pred['prediction'] = best
        predictions.append(pred)
    return predictions

def compute_frequency(train, diction):
    freq = {}
    for acr, lfs in diction.items():
        for lf in lfs:
            freq[lf] = 0
    for d in train:
        freq[d['expansion']] += 1
    return freq

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-input', type=str,
                        help='Path to the input file (e.g., dev.json)')
    parser.add_argument('-train', type=str,
                        help='Path to the train file')
    parser.add_argument('-diction', type=str,
                        help='Path to the dictionary')
    parser.add_argument('-output', type=str,
                        help='Prediction file path')
    args = parser.parse_args()

    ## READ data
    with open(args.input) as file:
        data = json.load(file)
    with open(args.train) as file:
        train = json.load(file)
    with open(args.diction) as file:
        diction = json.load(file)

    ## Compute Frequency
    freq = compute_frequency(train,diction)

    ## Predict
    predictions = predict(data, diction, freq)

    ## Save
    with open(args.output, 'w') as file:
        json.dump(predictions, file)