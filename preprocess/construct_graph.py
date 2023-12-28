import sys
import os
script_dir = os.path.dirname(os.path.abspath(__file__)) 
parent_dir = os.path.dirname(script_dir)  
sys.path.append(parent_dir)
from utils.common import *
import nltk
import re
from transformers import RobertaTokenizer
from tqdm import tqdm
import argparse

def build_graph(all_info):
    nodes = []
    edges = []
    entity_occur = {}
    last_sen_cnt = 0
    sens = [sen["sentence"].replace("\t", " ") for sen in all_info]
    all_kws = [sen["keywords"]["entity"] for sen in all_info]
    sen2node = []
    for sen_idx, sen_kws in enumerate(all_kws):
        sen_tmp_node = []
        kws_cnt = 0
        sen_kws = list([kw for kw in set(sen_kws) if kw.strip() not in cachedStopWords])
        if not keep_sen(sen_kws):
            sen2node.append([])
            continue
        for _, kw in enumerate(sen_kws):
            kw = re.sub(r"[^a-zA-Z0-9,.\'\`!?]+", " ", kw)
            words = [
                word
                for word in nltk.word_tokenize(kw)
                if (
                    word not in cachedStopWords
                    and word.capitalize() not in cachedStopWords
                )
            ]
            if keep_node(kw, words):
                sen_tmp_node.append(len(nodes))
                nodes.append({"text": kw, "words": words, "sentence_id": sen_idx})
                if kw not in entity_occur.keys():
                    entity_occur[kw] = 0
                entity_occur[kw] += 1
                kws_cnt += 1

        edges += [
            tuple([last_sen_cnt + i, last_sen_cnt + i + 1, "inner"])
            for i in list(range(kws_cnt - 1))
        ]

        last_sen_cnt += kws_cnt
        sen2node.append(sen_tmp_node)

    for i in range(len(nodes)):
        for j in range(len(nodes)):
            if j == i:
                continue
            ans = nodes[i]["text"].strip() == nodes[j]["text"].strip()
            if ans != 0:
                edges.append(tuple([min(i, j), max(i, j), "inter"]))

    if not nodes:
        return [], [], [], [], []
    return nodes, list(set(edges)), entity_occur, sens, sen2node


def clean_string(string):
    return re.sub(r"[^a-zA-Z0-9,.\'!?]+", "", string)


def generate_rep_mask_based_on_graph(ent_nodes, sens, tokenizer):
    sen_start_idx = [0]
    sen_idx_pair, sen_tokens, all_tokens, drop_nodes = [], [], [], []
    for sen in sens:
        sen_token = tokenizer.tokenize(sen)
        cleaned_sen_token = [clean_string(token) for token in sen_token]
        sen_tokens.append(cleaned_sen_token)
        sen_idx_pair.append(
            tuple([sen_start_idx[-1], sen_start_idx[-1] + len(sen_token)])
        )
        sen_start_idx.append(sen_start_idx[-1] + len(sen_token))
        all_tokens += sen_token

    for nidx, node in enumerate(ent_nodes):
        node_text = node["text"]
        start_pos, node_len = first_index_list(
            sen_tokens[node["sentence_id"]], clean_string(node_text)
        )
        if start_pos != -1:
            final_start_pos = sen_start_idx[node["sentence_id"]] + start_pos
            max_pos = final_start_pos + node_len
            ent_nodes[nidx]["spans"] = tuple([final_start_pos, max_pos])
        else:
            ent_nodes[nidx]["spans"] = tuple([-1, -1])
        if ent_nodes[nidx]["spans"][0] == -1:
            drop_nodes.append(nidx)
        else:
            ent_nodes[nidx]["spans_check"] = all_tokens[final_start_pos:max_pos]

    return ent_nodes, all_tokens, drop_nodes, sen_idx_pair


parser = argparse.ArgumentParser()
parser.add_argument(
    "--kw_file_dir",
    type=str,
    required=True,
    help="The path of the input dataset with keywords.",
)
parser.add_argument(
    "--output_dir",
    type=str,
    default="",
    help="The path to the output dataset with graph.",
)
args = parser.parse_args()

if __name__ == "__main__":
    if args.output_dir == "":
        args.output_dir = args.kw_file_dir.replace("_kws.jsonl", "_graph.jsonl")
    print("Loading Dataset ...")
    data = read_data(args.kw_file_dir)
    print("Loading Tokenizer ...")
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base", do_lower_case=False)
    max_seq_length = 512
    no_node = 0
    with open(args.output_dir, "w", encoding="utf8") as outf:
        for idx, line in tqdm(enumerate(data)):
            kws = line["information"]["keywords"]
            nodes, edges, entity_occur, sens, sen2node = build_graph(kws)
            if not nodes:
                no_node += 1
            (
                nodes,
                all_tokens,
                drop_nodes,
                sen_idx_pair,
            ) = generate_rep_mask_based_on_graph(nodes, sens, tokenizer, max_seq_length)

            line["information"]["graph"] = {
                "nodes": nodes,
                "edges": edges,
                "all_tokens": all_tokens,
                "drop_nodes": drop_nodes,
                "sentence_to_node_id": sen2node,
                "sentence_start_end_idx_pair": sen_idx_pair,
            }
            outf.write(json.dumps(line) + "\n")

    print("{} instances are too short that have no graph".format(no_node))
