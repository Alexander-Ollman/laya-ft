"""Validate and score the secondary matched Jev comparison without changing Laya results."""
import json
from collections import defaultdict
from pathlib import Path

from evaluate_moderation_jev import DATA, RESULTS, MODEL, validate_resume
from analyze_moderation import validated_run
from moderation_metrics import moderation_metrics, paired_bootstrap
from public_workflow import digest, read_jsonl, validate_case_file


def analyze(data=DATA, results=RESULTS):
    study = json.loads((data/'study_manifest.json').read_text())
    study_sha = digest((data/'study_manifest.json').read_bytes())
    output = {'schema': 'moderation-jev-posthoc-v1', 'study_sha256': study_sha,
              'scope': 'Secondary post-hoc comparisons; 95% descriptive intervals, no multiplicity correction',
              'complete': True, 'tasks': []}
    identities = set()
    checkpoint_cache = {}
    original_path = results/'analysis.json'
    original = json.loads(original_path.read_text()) if original_path.exists() else None
    if original is not None and (not original.get('complete') or original.get('study_sha256') != study_sha):
        raise ValueError('original Laya analysis incomplete or different study')
    originals = {t['slug']:t for t in original['tasks']} if original else {}
    preflight = json.loads((results/'preflight.json').read_text())
    for slug, count in study['test_counts'].items():
        path = results/(slug+'_jev.json')
        if not path.exists():
            output['complete'] = False
            output['tasks'].append({'slug':slug, 'complete':False})
            continue
        source = json.loads(path.read_text())
        folder = data/slug
        sha = validate_case_file(folder/'heldout.jsonl', folder)
        cases = read_jsonl(folder/'heldout.jsonl')
        manifest = json.loads((folder/'manifest.json').read_text())
        matches = [v for k,v in preflight.items() if k.endswith('/'+slug+'/heldout.jsonl')]
        if len(matches) != 1 or matches[0]['n'] != count:
            raise ValueError('invalid tokenizer preflight')
        truncated = set(matches[0]['truncated_ids'])
        for key, expected in [('cases_sha256', sha), ('study_sha256', study_sha),
                              ('manifest_sha256', digest((folder/'manifest.json').read_bytes())),
                              ('dataset', slug), ('target','jev'), ('model_requested', MODEL)]:
            if source.get(key) != expected:
                raise ValueError('invalid Jev metadata: '+key)
        rows = source['rows']
        if len(rows) != count or len(cases) != count:
            raise ValueError('incomplete Jev result')
        identities.update(validate_resume(rows, cases))
        if len(identities) != 1 or source['model_resolved'] != sorted(identities):
            raise ValueError('inconsistent Jev identities')
        summary = moderation_metrics(rows)
        if summary != source['summary']:
            raise ValueError('Jev saved metrics differ from recomputation')
        task = {'slug':slug, 'complete':True, 'n':count, 'input_sha256':digest(path.read_bytes()),
                'summary':summary, 'comparisons':{}}
        for name in ['base','fine1000','fine5000']:
            local_path = results/(slug+'_'+name+'.json')
            local_blob = local_path.read_bytes()
            local = json.loads(local_blob)
            if slug not in originals or name not in originals[slug]['runs']:
                raise ValueError('missing original validated Laya analysis')
            original_run = originals[slug]['runs'][name]
            if digest(local_blob) != original_run['input_sha256']:
                raise ValueError('local raw result differs from original validated analysis')
            local_rows = validated_run(local, name, manifest, source['manifest_sha256'], study_sha,
                                       cases, truncated, checkpoint_cache)
            recomputed = moderation_metrics(local_rows)
            if recomputed != local['summary'] or recomputed != original_run['summary']:
                raise ValueError('local summary differs from original validated analysis')
            for key in ['cases_sha256','study_sha256','manifest_sha256']:
                if local[key] != source[key]:
                    raise ValueError('local/Jev protocol mismatch: '+key)
            validate_resume(local['rows'], cases)
            if len(local['rows']) != count:
                raise ValueError('local/Jev count mismatch')
            task['comparisons']['jev_vs_'+name] = {
                'reference':name, 'comparison':'jev', 'secondary_posthoc':True,
                'reference_sha256':digest(local_blob),
                'reference_checkpoint_verification': original_run.get('checkpoint_verification'),
                **paired_bootstrap(local['rows'], rows, confidence=.95)}
        if slug == 'openai_categories':
            groups = defaultdict(list)
            for row in rows:
                groups[row['policy_category']].append(row)
            task['policy_categories'] = {key:moderation_metrics(value) for key,value in sorted(groups.items())}
        output['tasks'].append(task)
    output['model_resolved'] = sorted(identities)
    return output


if __name__ == '__main__':
    result = analyze()
    dest = RESULTS/'jev_analysis.json'
    dest.write_text(json.dumps(result, indent=2)+'\n')
    print('Complete:', result['complete'], 'tasks:', sum(t['complete'] for t in result['tasks']))
