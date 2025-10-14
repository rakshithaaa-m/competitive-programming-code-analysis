#!/usr/bin/env python3
"""
preprocess.py
- Walks data/<contest>/*.json, removes comments, replaces identifiers with markers (!!VAR, !!STR, !!CHR),
  and writes processed JSON into processed/<contest>/<submission_id>.json with fields:
    { "submission_id":..., "tokens_seq": "tok1 tok2 tok3 ...", "country":..., "rating":..., "rank_label":... }
- If libclang is present it can be integrated (not implemented here); current method uses regex-based replacement.
"""
import os, re, json, argparse
from glob import glob

CPP_KEYWORDS = {
 "alignas","alignof","and","and_eq","asm","atomic_cancel","atomic_commit","atomic_noexcept","auto",
 "bitand","bitor","bool","break","case","catch","char","char16_t","char32_t","class","compl",
 "concept","const","constexpr","const_cast","continue","co_await","co_return","co_yield","decltype",
 "default","delete","do","double","dynamic_cast","else","enum","explicit","export","extern","false",
 "float","for","friend","goto","if","inline","int","long","mutable","namespace","new","noexcept",
 "not","not_eq","nullptr","operator","or","or_eq","private","protected","public","register",
 "reinterpret_cast","requires","return","short","signed","sizeof","static","static_assert",
 "static_cast","struct","switch","template","this","thread_local","throw","true","try","typedef",
 "typeid","typename","union","unsigned","using","virtual","void","volatile","wchar_t","while",
 "xor","xor_eq"
}

IDENT_RE = re.compile(r'\b([A-Za-z_][A-Za-z0-9_]*)\b')
STR_RE = re.compile(r'\"(\\.|[^"\\])*\"', re.S)
CHR_RE = re.compile(r'\'(\\.|[^\'\\])*\'', re.S)
C_COMMENT_RE = re.compile(r'/\*.*?\*/', re.S)
CPP_LINE_COMMENT_RE = re.compile(r'//.*?$' , re.M)

def preprocess_code(code):
    # remove C-style comments
    code = C_COMMENT_RE.sub(' ', code)
    code = CPP_LINE_COMMENT_RE.sub(' ', code)
    # replace strings and chars
    code = STR_RE.sub(' !!STR ', code)
    code = CHR_RE.sub(' !!CHR ', code)
    # replace identifiers (approx): leave keywords alone
    def repl_ident(m):
        token = m.group(1)
        if token in CPP_KEYWORDS:
            return token
        # numeric literal? leave
        if token.isdigit():
            return token
        return ' !!VAR '
    code = IDENT_RE.sub(repl_ident, code)
    # normalize braces and punctuation into separate tokens
    code = re.sub(r'([{}()\[\];,<>+\-*/=%:&|^~!])', r' \1 ', code)
    # collapse whitespace
    toks = [t for t in re.split(r'\s+', code) if t]
    return toks

def main(args):
    for contest_dir in os.listdir(args.data_dir):
        cpath = os.path.join(args.data_dir, contest_dir)
        if not os.path.isdir(cpath): continue
        outdir = os.path.join(args.out_dir, contest_dir)
        os.makedirs(outdir, exist_ok=True)
        for jfile in os.listdir(cpath):
            try:
                src = json.load(open(os.path.join(cpath, jfile), encoding='utf-8'))
            except Exception:
                continue
            code = src.get('code') or ''
            if not code or len(code) < 5: continue
            toks = preprocess_code(code)
            out = {
                'submission_id': src['submission_id'],
                'tokens': toks,
                'country': src.get('country'),
                'creationTimeSeconds': src.get('creationTimeSeconds')
            }
            open(os.path.join(outdir, f"{src['submission_id']}.json"), 'w', encoding='utf-8').write(json.dumps(out))

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', default='data', help='raw data dir (per-contest)')
    p.add_argument('--out_dir', default='processed', help='processed output dir')
    args = p.parse_args()
    main(args)
