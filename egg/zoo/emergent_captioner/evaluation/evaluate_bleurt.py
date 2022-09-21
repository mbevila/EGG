"""!!!SETUP!!!

Run the following (in the same dir as this script):

git clone https://github.com/mtanti/coco-caption.git
cd coco-caption
bash get_stanford_models.sh
cd ..
"""

from collections import Counter
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import more_itertools
import json
from tqdm import tqdm

def read_plaintext_files(preds, gold):
    """
    Plaintext, TSV file.

    This is a caption   This is another caption
    This is yet another caption
    """
    with open(preds) as fin1, open(gold) as fin2:
        for i, (p, gg) in tqdm(enumerate(zip(fin1, fin2))):
            p = p.strip()
            gg = gg.strip().split('\t')
            for g in gg:
                yield i, p, g

@torch.no_grad()
def bleurt_scores(model, tokenizer, iterable, batch_size=10):
    device = next(model.parameters()).device
    for batch in more_itertools.chunked(iterable, batch_size):
        ii, pp, gg = zip(*batch)
        batch = tokenizer(pp, gg, padding=True, return_tensors="pt").to(device)
        out1 = model(**batch)[0].squeeze(-1).tolist()
        yield from zip(ii, out1, pp, gg)

if __name__ == '__main__':

    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('prediction_file',
                        help="File containing predictions.")
    parser.add_argument('gold_file', help="File containing gold captions.")
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--verbose', action="store_true")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained("Elron/bleurt-large-512")
    model = AutoModelForSequenceClassification.from_pretrained("Elron/bleurt-large-512").eval().to(args.device)

    it = read_plaintext_files(args.prediction_file, args.gold_file)
    it = bleurt_scores(model, tokenizer, it)

    bleurt1 = {}

    for x in zip(it):
        i, o1, p, g = x[0]
        bleurt1[i] = max(bleurt1.get(i, 0.0), o1)
        if args.verbose:
            print(i, o1, p, g)

    if args.verbose:
        print("\n\n==========")

    print('BLEURT:', sum(bleurt1.values()) / len(bleurt1))
