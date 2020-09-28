import argparse
import json
from collections import defaultdict

def run_evaluation(args):
    verbose = args.v
    with open(args.g) as file:
        gold = dict([(d['id'], d['expansion']) for d in json.load(file)])
    with open(args.p) as file:
        pred = dict([(d['id'], d['prediction']) for d in json.load(file)])
        pred = [pred[k] for k,v in gold.items()]
        gold = [gold[k] for k,v in gold.items()]
    p, r, f1 = score_expansion(gold, pred, verbos=verbose)
    return p, r, f1

def score_expansion(key, prediction, verbos=False):
    correct = 0
    for i in range(len(key)):
        if key[i] == prediction[i]:
            correct += 1
    acc = correct / len(prediction)

    expansions = set()

    correct_per_expansion = defaultdict(int)
    total_per_expansion = defaultdict(int)
    pred_per_expansion = defaultdict(int)
    for i in range(len(key)):
        expansions.add(key[i])
        total_per_expansion[key[i]] += 1
        pred_per_expansion[prediction[i]] += 1
        if key[i] == prediction[i]:
            correct_per_expansion[key[i]] += 1

    precs = defaultdict(int)
    recalls = defaultdict(int)

    for exp in expansions:
        precs[exp] = correct_per_expansion[exp] / pred_per_expansion[exp] if exp in pred_per_expansion else 1
        recalls[exp] = correct_per_expansion[exp] / total_per_expansion[exp]

    micro_prec = sum(correct_per_expansion.values()) / sum(pred_per_expansion.values())
    micro_recall = sum(correct_per_expansion.values()) / sum(total_per_expansion.values())
    micro_f1 = 2*micro_prec*micro_recall/(micro_prec+micro_recall) if micro_prec+micro_recall != 0 else 0

    macro_prec = sum(precs.values()) / len(precs)
    macro_recall = sum(recalls.values()) / len(recalls)
    macro_f1 = 2*macro_prec*macro_recall / (macro_prec+macro_recall) if macro_prec+macro_recall != 0 else 0

    if verbos:
        print('Accuracy: {:.3%}'.format(acc))
        print('-'*10)
        print('Micro Precision: {:.3%}'.format(micro_prec))
        print('Micro Recall: {:.3%}'.format(micro_recall))
        print('Micro F1: {:.3%}'.format(micro_f1))
        print('-'*10)
        print('Macro Precision: {:.3%}'.format(macro_prec))
        print('Macro Recall: {:.3%}'.format(macro_recall))
        print('Macro F1: {:.3%}'.format(macro_f1))
        print('-'*10)

    return macro_prec, macro_recall, macro_f1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', type=str,
                        help='Gold file path')
    parser.add_argument('-p', type=str,
                        help='Predictions file path')
    parser.add_argument('-v', dest='v',
                        default=False, action='store_true',
                        help="Verbose Evaluation")


    args = parser.parse_args()
    p, r, f1 = run_evaluation(args)
    print('Official Scores:')
    print('P: {:.2%}, R: {:.2%}, F1: {:.2%}'.format(p,r,f1))
