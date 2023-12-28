import json
import nltk
from tqdm import tqdm
from allennlp.predictors.predictor import Predictor
import re
import argparse

predictor_ner = Predictor.from_path(
    "https://storage.googleapis.com/allennlp-public-models/ner-elmo.2021-02-12.tar.gz",
    cuda_device=0,
)

def load_jsonl(filename):
    with open(filename, "r") as file:
        for line in file:
            yield json.loads(line.strip())


def found_key_words(claims):
    all_ent_res = predictor_ner.predict_batch_json(
        inputs=[{"sentence": text} for text in claims]
    )
    all_keywords = []
    for i in range(len(claims)):
        claim = claims[i]
        ent_res = all_ent_res[i]

        key_words = {"noun": [], "claim": claim, "subject": [], "entity": []}
        all_ents = extract_entity_allennlp(ent_res["words"], ent_res["tags"])
        key_words["entity"].extend(all_ents)
        key_words = {"keywords": key_words, "sentence": claim}

        all_keywords.append(key_words)
    return all_keywords


def analyze_document(doc):
    sens = nltk.sent_tokenize(doc)
    resplit_sens = []
    for sen in sens:
        resplit_sens += [s.strip() for s in sen.split("\n") if s.strip() != ""]
    sens = resplit_sens
    try:
        all_keywords = found_key_words(sens)
    except Exception as e:
        print(e)
        all_keywords = []

    return all_keywords


def extract_entity_allennlp(words, tags):
    assert len(words) == len(tags)
    e_list_cache = []
    e_list_final = []
    for i in range(len(tags)):
        if tags[i] != "O":
            if tags[i].startswith("B"):
                start_index = i
                continue

            if tags[i].startswith("I"):
                continue

            if tags[i].startswith("L"):
                end_index = i

                for j in range(start_index, end_index + 1):
                    cword = re.sub(r"[^a-zA-Z0-9,.\'-/!?]+", "", words[j])
                    e_list_cache.append(cword)
                entity = " ".join(e_list_cache)
                e_list_cache = []
                e_list_final.append(entity)

            if tags[i].startswith("U"):
                cword = re.sub(r"[^a-zA-Z0-9,.\'-/!?]+", "", words[i])
                e_list_final.append(cword)

    return e_list_final


parser = argparse.ArgumentParser()
parser.add_argument(
    "--raw_dir", 
    type=str, 
    required=True, 
    help="The path of raw dataset."
)
parser.add_argument(
    "--output_dir",
    type=str,
    default="",
    help="The path to output dataset with keywords.",
)
args = parser.parse_args()

if __name__ == "__main__":
    if args.output_dir == "":
        args.output_dir = args.raw_dir.replace(".jsonl", "_kws.jsonl")
    file = list(load_jsonl(args.raw_dir))
    with open(args.output_dir, "w") as out_file:
        for line in tqdm(file):
            doc = line["article"]
            line["information"] = {"keywords": analyze_document(doc)}
            out_file.write(json.dumps(line) + "\n")
