"""Post-hoc near-duplicate sensitivity check; frozen primary scores stay unchanged."""
import json,re
from collections import Counter,defaultdict
from pathlib import Path
from public_workflow import ROOT,norm,read_jsonl
from evaluate_workflow_suite import detailed_metrics


def grams(text):
    text=norm(text)
    return {text[i:i+5] for i in range(max(1,len(text)-4))}


def audit():
    analysis=json.loads((ROOT/'bench/results/workflow-suite/analysis.json').read_text())
    out={'design':'Post-hoc audit after early results; no retraining, prompt changes, or primary-score changes.',
         'thresholds':{'character_5gram_jaccard':.85,'digit_masked_normalized_exact_match':True},
         'limitation':'These lexical rules find repeated or nearly repeated templates, not every semantic overlap or pretraining contamination.',
         'tasks':[]}
    for task in analysis['tasks']:
        folder=ROOT/'train/workflow-suite'/task['slug'];train=read_jsonl(folder/'train.jsonl');test=read_jsonl(folder/'heldout.jsonl')
        trainsets=[grams(c['state']['request']) for c in train];index=defaultdict(set);digit_index=defaultdict(list)
        for i,g in enumerate(trainsets):
            for word in g:index[word].add(i)
            digit_index[re.sub(r'\d+','NUMBER',norm(train[i]['state']['request']))].append(i)
        flagged=[]
        for c in test:
            g=grams(c['state']['request']);overlap=Counter(i for word in g for i in index.get(word,()))
            best=max(((n/(len(g)+len(trainsets[i])-n),i) for i,n in overlap.items()),default=(0,None))
            digit=digit_index.get(re.sub(r'\d+','NUMBER',norm(c['state']['request'])),[])
            if best[0]>=.85 or digit:
                flagged.append({'test_id':c['id'],'best_jaccard':best[0],
                  'nearest_train_id':train[best[1]]['id'] if best[1] is not None else None,
                  'digit_match_train_ids':[train[i]['id'] for i in digit]})
        blocked={x['test_id'] for x in flagged};summaries={}
        for target,run in task['runs'].items():
            rows=[r for r in json.loads(Path(run['input']).read_text())['rows'] if r['id'] not in blocked]
            summaries[target]=detailed_metrics(rows,task['labels']) if rows else None
        out['tasks'].append({'slug':task['slug'],'original_test_n':len(test),'flagged_n':len(flagged),
          'retained_n':len(test)-len(flagged),'flagged':flagged,'retained_summaries':summaries})
    path=ROOT/'bench/results/workflow-suite/similarity_audit.json';path.write_text(json.dumps(out,indent=2)+'\n')
    for t in out['tasks']:print(t['slug'],'flagged',t['flagged_n'],'retained',t['retained_n'])

if __name__=='__main__':audit()
