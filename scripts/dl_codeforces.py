#!/usr/bin/env python3
"""
dl_codeforces.py
- Usage example:
    python scripts/dl_codeforces.py --contests 1100 1107 --outdir data --max-per-contest 5000
Notes:
 - Uses Codeforces API contest.status to collect submissions metadata, then fetches the submission page
   to extract the source code. Saves one JSON per submission in data/<contest>/
 - Includes simple caching and polite rate-limiting.
"""
import os, time, json, argparse, requests
from lxml import html
from tqdm import tqdm

API_BASE = "https://codeforces.com/api"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; project-bot/1.0; +https://example.com/)",
}

def get_contest_status(contest_id, count=100000):
    url = f"{API_BASE}/contest.status?contestId={contest_id}&from=1&count={count}"
    r = requests.get(url, headers=HEADERS)
    r.raise_for_status()
    data = r.json()
    if data['status'] != 'OK':
        raise RuntimeError("API error: " + str(data))
    return data['result']

def best_ok_submissions(submissions):
    # Keep latest OK submission per (handle, problem)
    by_key = {}
    for s in submissions:
        if s.get('verdict') != 'OK': continue
        author = s.get('author', {})
        members = author.get('members', [])
        if not members: continue
        handle = members[0].get('handle')
        prob = s['problem']['index']
        key = (handle, prob)
        cur = by_key.get(key)
        # prefer larger creationTimeSeconds (later)
        if cur is None or s['creationTimeSeconds'] > cur['creationTimeSeconds']:
            by_key[key] = s
    return list(by_key.values())

def fetch_source_from_submission_page(contest_id, submission_id):
    url = f"https://codeforces.com/contest/{contest_id}/submission/{submission_id}"
    r = requests.get(url, headers=HEADERS)
    if r.status_code != 200:
        return None
    tree = html.fromstring(r.text)
    # Code often appears inside <pre id="program-source-text"> or <pre class="prettyprint">
    pre = tree.xpath('//pre[@id="program-source-text"]')
    if not pre:
        pre = tree.xpath('//pre[contains(@class,"prettyprint")]')
    if not pre:
        # try other heuristics
        pre = tree.xpath('//div[contains(@class,"program")]/pre')
    if pre:
        return pre[0].text_content()
    return None

def batch_get_user_countries(handles):
    # Codeforces API: user.info?handles=a;b;c  (up to large number)
    out = {}
    chunk = 1000
    for i in range(0, len(handles), chunk):
        subset = handles[i:i+chunk]
        handles_param = ';'.join(subset)
        url = f"{API_BASE}/user.info?handles={handles_param}"
        r = requests.get(url, headers=HEADERS)
        r.raise_for_status()
        data = r.json()
        if data['status'] != 'OK':
            continue
        for u in data['result']:
            out[u['handle']] = u.get('country')  # may be None
        time.sleep(0.2)
    return out

def main(args):
    os.makedirs(args.outdir, exist_ok=True)
    for cid in args.contests:
        print("Processing contest", cid)
        outdir = os.path.join(args.outdir, str(cid))
        os.makedirs(outdir, exist_ok=True)
        submissions = get_contest_status(cid, count=args.count)
        ok = best_ok_submissions(submissions)
        ok = ok[:args.max_per_contest]
        handles = sorted({s['author']['members'][0]['handle'] for s in ok})
        countries = batch_get_user_countries(handles)
        for s in tqdm(ok):
            sid = s['id']
            handle = s['author']['members'][0]['handle']
            prob = s['problem']['index']
            lang = s.get('programmingLanguage','')
            outpath = os.path.join(outdir, f"{sid}.json")
            if os.path.exists(outpath) and not args.redownload:
                continue
            code = fetch_source_from_submission_page(cid, sid)
            if not code:
                # try alternative endpoint or skip
                time.sleep(0.5)
                code = fetch_source_from_submission_page(cid, sid)
            payload = {
                'submission_id': sid,
                'contest_id': cid,
                'handle': handle,
                'problem': prob,
                'language': lang,
                'country': countries.get(handle),
                'creationTimeSeconds': s.get('creationTimeSeconds'),
                'code': code
            }
            open(outpath, 'w', encoding='utf-8').write(json.dumps(payload))
            time.sleep(args.sleep)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--contests', nargs='+', required=True, help='contest ids', type=int)
    p.add_argument('--outdir', default='data', help='output directory')
    p.add_argument('--count', type=int, default=100000, help='how many submissions to request from API')
    p.add_argument('--max-per-contest', type=int, default=5000)
    p.add_argument('--sleep', type=float, default=0.5)
    p.add_argument('--redownload', action='store_true')
    args = p.parse_args()
    main(args)
