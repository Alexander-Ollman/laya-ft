"""Post-hoc matched Jev moderation evaluation; never a training source."""
import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import threading

from moderation_metrics import moderation_metrics
from public_workflow import ROOT, digest, read_jsonl, parse_answer, validate_case_file
from run_bench import load_env, make_target
from telemetry import Recorder

DATA = ROOT / 'train/moderation-study'
RESULTS = ROOT / 'bench/results/moderation-study'
MODEL = 'typesafe/jev-1.13'
LOCAL = threading.local()


def request(case, rec, split='heldout'):
    if not hasattr(LOCAL, 'target'):
        LOCAL.target = make_target('jev:' + MODEL)
    start = time.perf_counter()
    answers, usage, resolved, error, prediction, probs = {}, {}, None, None, None, None
    try:
        answers, usage, resolved = LOCAL.target.ask(case)
        prediction, probs = parse_answer(answers, ['safe', 'unsafe'])
    except Exception as exc:
        # HTTP errors contain an endpoint but no request body or credentials.
        error = type(exc).__name__ + ': ' + str(exc)[:250]
    latency = (time.perf_counter() - start) * 1000
    rec.record('jev', MODEL, case['state'], case['questions'], answers,
               model_resolved=resolved, case_id=case['id'], split=split,
               expected=case['expected'], label_source=case['label_source'],
               latency_ms=latency, usage=usage, error=error)
    return {'id': case['id'], 'text_group': case['text_group'],
            'expected': case['expected']['intent'], 'prediction': prediction,
            'probabilities': probs, 'error': error, 'latency_ms': latency,
            'usage': usage, 'model_resolved': resolved,
            'categories': case.get('categories', []),
            'policy_category': case.get('policy_category'),
            'input_truncated': None}


def validate_resume(rows, cases):
    if len(rows) > len(cases):
        raise ValueError('too many saved rows')
    for row, case in zip(rows, cases):
        for field, expected in [('id', case['id']), ('text_group', case['text_group']),
                                ('expected', case['expected']['intent']),
                                ('categories', case.get('categories', [])),
                                ('policy_category', case.get('policy_category'))]:
            if row.get(field) != expected:
                raise ValueError('saved row differs from frozen case: ' + field)
    identities = {r['model_resolved'] for r in rows if not r['error']}
    if None in identities or len(identities) > 1:
        raise ValueError('missing or changed resolved model identity')
    return identities


def evaluate(workers=4):
    load_env()
    manifest = json.loads((DATA / 'study_manifest.json').read_text())
    warm = read_jsonl(DATA / 'aegis_1000/development.jsonl')[0]
    global_identity = set()
    for slug in manifest['test_counts']:
        folder = DATA / slug
        path = folder / 'heldout.jsonl'
        sha = validate_case_file(path, folder)
        cases = read_jsonl(path)
        dest = RESULTS / (slug + '_jev.json')
        progress = RESULTS / (slug + '_jev.progress.jsonl')
        meta_path = RESULTS / (slug + '_jev.progress.meta.json')
        fixed = {'dataset': slug, 'target': 'jev', 'model_requested': MODEL,
                 'cases_sha256': sha, 'manifest_sha256': digest((folder/'manifest.json').read_bytes()),
                 'study_sha256': digest((DATA/'study_manifest.json').read_bytes()),
                 'protocol': 'post-hoc secondary comparison; unchanged frozen state/questions',
                 'concurrency': workers, 'input_handling': 'full request sent; server truncation unknown'}
        if meta_path.exists():
            if json.loads(meta_path.read_text()) != fixed:
                raise ValueError('resume metadata changed')
        else:
            if progress.exists() or dest.exists():
                raise ValueError('results without resume metadata')
            meta_path.write_text(json.dumps(fixed, indent=2)+'\n')
        rows = read_jsonl(progress) if progress.exists() else []
        global_identity.update(validate_resume(rows, cases))
        if len(global_identity) > 1:
            raise ValueError('resolved identity changed across datasets')
        if dest.exists():
            saved = json.loads(dest.read_text())
            if saved['rows'] != rows or len(rows) != len(cases):
                raise ValueError('completed result does not match progress')
            continue
        rec = Recorder(time.strftime('%Y%m%dT%H%M%S')+'_moderation_'+slug+'_jev',
                       'public/moderation-study/jev-posthoc', meta=fixed)
        rec.attach(__file__, meta_path, folder/'manifest.json', DATA/'study_manifest.json')
        start = time.perf_counter()
        try:
            warmed = request(warm, rec, 'development')
            if warmed['error']:
                raise RuntimeError('warmup failed: '+warmed['error'])
            global_identity.add(warmed['model_resolved'])
            if None in global_identity or len(global_identity) != 1:
                raise ValueError('warmup identity changed')
            with ThreadPoolExecutor(max_workers=workers) as pool:
                while len(rows) < len(cases):
                    batch = cases[len(rows):len(rows)+workers]
                    results = list(pool.map(lambda c: request(c, rec), batch))
                    with progress.open('a') as out:
                        for row in results:
                            out.write(json.dumps(row)+'\n')
                            out.flush()
                            rows.append(row)
                    global_identity.update(validate_resume(rows, cases))
                    if len(global_identity) != 1:
                        raise ValueError('resolved identity changed')
                    if len(rows) >= 5 and all(r['error'] for r in rows[-5:]):
                        raise RuntimeError('five consecutive failures')
                    if len(rows) % 100 < workers or len(rows) == len(cases):
                        cost = sum(float(r['usage'].get('cost_usd') or 0) for r in rows)
                        print(slug, len(rows), '/', len(cases), 'seconds', round(time.perf_counter()-start),
                              'USD', round(cost, 6), 'errors', sum(bool(r['error']) for r in rows), flush=True)
            result = dict(fixed, model_resolved=sorted(global_identity), run_id=rec.run_id,
                          summary=moderation_metrics(rows), rows=rows)
            dest.write_text(json.dumps(result, indent=1)+'\n')
            rec.attach(dest)
            print('COMPLETE', slug, result['summary']['harmful_f1'], flush=True)
        finally:
            rec.close(outcome=f'evaluated {len(rows)}/{len(cases)}; Jev outputs prohibited for training')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=4, choices=[1, 2, 4])
    evaluate(parser.parse_args().workers)
