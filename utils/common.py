import json
import re
import nltk
import torch

nltk.download("stopwords")
from nltk.corpus import stopwords

cachedStopWords = stopwords.words("english") + [
    "it",
    "she",
    "in",
    "the",
    "he",
    "i",
    "be",
    "s",
    "there",
    "here",
    "The",
    "at",
    "to",
    "n't",
    "'s",
    "That",
    "It",
    "I",
    "To",
    "But",
    "s",
    "re",
    " s",
    "Why",
    "However",
    "A",
    "It",
    "They",
    "She",
    "He",
    "But",
    "Although",
    "In",
    "Just",
    "When",
    "This",
    "Also",
    "As",
    "Have",
    "If",
    "So",
    "Other",
    "You",
    "We",
    "For",
    "Or",
    "Both",
    "Ever",
    "Because",
    "Now",
    "n t",
    "Once",
]


def read_data(fp):
    if fp.endswith(".jsonl"):
        file = open(fp, "r", encoding="utf8")
        data = [json.loads(line) for line in file.readlines()]
    elif fp.endswith(".cache"):
        file = torch.load(fp)
        data = file
    return data


def judge_upper(text):
    bigchar = re.findall(r"[A-Z]", text)
    return len(bigchar) > 0


def keep_sen(sen_kws):
    if len(sen_kws) == 0:
        return False
    if len(sen_kws) == 1:
        if not judge_upper(sen_kws[0]):
            return False

    return True


def keep_node(text, words):
    if text not in cachedStopWords and words:
        return True
    else:
        return False


def combine_tokens(tokens, c_tokens):
    new_tokens = []
    str = ""
    for idx, token in enumerate(tokens):
        ctoken = re.sub(r"[^a-zA-Z0-9,.\'-/!?]+", "", token)
        if token == ctoken:
            str += c_tokens[idx]
        else:
            new_tokens.append(str)
            str = ""
    return new_tokens


def find_index(a, b):
    length_b = len(b)
    text_b = " ".join(b)
    for i in range(len(a)):
        if i + len(b) > len(a):
            return -1, 0
        if " ".join(a[i : i + length_b]) == text_b:
            return i, length_b
    return -1, 0


def cal_pos(cleaned_tokens, text):
    pos = []
    max_start, max_length = 0, 0
    single = []
    for idx, token in enumerate(cleaned_tokens):
        p = text.find(token)
        if p != -1 and token != "":
            if len(pos) == 0:
                pos = [idx]
            else:
                if abs(idx - pos[-1]) == 1 or (
                    abs(idx - pos[-1]) == 2 and cleaned_tokens[idx - 1] == ""
                ):
                    pos.append(idx)
                else:
                    single.append(pos)
                    pos = [idx]
                if (pos[-1] - pos[0] + 1) > max_length:
                    max_start = pos[0]
                    max_length = pos[-1] - pos[0] + 1
        else:
            if len(pos) > 0:
                single.append(pos)
                if (pos[-1] - pos[0] + 1) > max_length:
                    max_start = pos[0]
                    max_length = pos[-1] - pos[0] + 1
                pos = []

    if len(single) > 1:
        min_dis, final_span = 999999, []
        for span in single:
            span_text = "".join([cleaned_tokens[s] for s in span])
            dis = abs(len(text) - len(span_text))
            if dis < min_dis:
                min_dis = dis
                final_span = span
        return final_span[0], final_span[-1] - final_span[0] + 1

    if max_length == 0 and not pos:
        return -1, -1

    return max_start, max_length if max_length > 0 else pos[-1] - pos[0] + 1


def first_index_list(cleaned_tokens, text):
    start, length = cal_pos(cleaned_tokens, text)
    return start, length


def number_h(num):
    for unit in ["", "K", "M", "G", "T", "P", "E", "Z"]:
        if abs(num) < 1000.0:
            return "%3.1f%s" % (num, unit)
        num /= 1000.0
    return "%.1f%s" % (num, "Yi")
