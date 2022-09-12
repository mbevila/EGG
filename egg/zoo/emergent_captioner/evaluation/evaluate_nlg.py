"""!!!SETUP!!!

Run the following (in the same dir as this script):

git clone https://github.com/mtanti/coco-caption.git
cd coco-caption
bash get_stanford_models.sh
cd ..
"""


import json

def read_plaintext_file(file):
    """
    Plaintext, TSV file.

    This is a caption   This is another caption
    This is yet another caption
    """

    data = {}
    with open(file) as fin:
        for i, line in enumerate(fin):
            captions = line.strip().split('\t')
            data[i] = [{'caption': c} for c in captions]
    return data


def read_jsonfile(file):
    """


    Should be formatted like this
    {
        "1": [
            {"caption": "This is a caption"},
            {"caption": "This is another caption"}
        ],
        "2": [
            {"caption": "This is yet another caption"}
        ]
    }
    """
    with open(file) as fin:
        return json.load(fin)


if __name__ == '__main__':
    import sys
    from pathlib import Path
    evaluation_repo_path = Path(__file__).absolute().parent / 'coco-caption'
    evaluation_repo_path = str(evaluation_repo_path)
    sys.path.append(evaluation_repo_path)

    from argparse import ArgumentParser
    from pycocoevalcap.eval import PTBTokenizer, Bleu, Meteor, Rouge, Cider, Spice
    from pprint import pprint

    parser = ArgumentParser()
    parser.add_argument('prediction_file',
                        help="File containing predictions. Can be a plaintext or JSON file. Format details in the script.")
    parser.add_argument('gold_file', help="File containing gold captions. Can be a plaintext or JSON file.")
    parser.add_argument('--format', choices="plaintext json".split(), default="plaintext")
    parser.add_argument('--write_detailed_results', default="")
    args = parser.parse_args()

    tokenizer = PTBTokenizer()

    if args.format == 'json':
        predictions = tokenizer.tokenize(read_jsonfile(args.prediction_file))
        ground_truth = tokenizer.tokenize(read_jsonfile(args.gold_file))
    elif args.format == 'plaintext':
        predictions = tokenizer.tokenize(read_plaintext_file(args.prediction_file))
        ground_truth = tokenizer.tokenize(read_plaintext_file(args.gold_file))

    scorers = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        (Meteor(), "METEOR"),
        (Rouge(), "ROUGE_L"),
        (Cider(), "CIDEr"),
        (Spice(), "SPICE"),
    ]

    summary = {}
    instance_information = {}

    for scorer, method in scorers:
        print('computing %s score...' % (scorer.method()))
        score, scores = scorer.compute_score(ground_truth, predictions)
        if isinstance(method, list):
            for sc, scs, m in zip(score, scores, method):
                summary[m] = sc
                for k, v in predictions.items():
                    for cap in v:
                        instance_information.setdefault('scores', {})[m] = scs
        else:
            summary[method] = score
            for k, v in predictions.items():
                for cap in v:
                    instance_information.setdefault('scores', {})[method] = scores

    print()
    pprint(summary)

    if args.write_detailed_results:
        summary.update(instance_information)
        with open(args.write_detailed_results, 'w') as fout:
            json.dump(instance_information, fout)

