"""Serial local safety evaluations; published competitors are reference-only."""
import argparse
import json
import time
from pathlib import Path

from evaluate_workflow_suite import LocalTarget
from moderation_metrics import moderation_metrics
from public_workflow import ROOT, digest, read_jsonl, parse_answer, validate_case_file
from telemetry import Recorder

DATA=ROOT/'train/moderation-study'
RESULTS=ROOT/'bench/results/moderation-study'

def evaluate(checkpoint,name):
    manifest=json.loads((DATA/'study_manifest.json').read_text())
    preflight=json.loads((RESULTS/'preflight.json').read_text())
    target=LocalTarget(checkpoint.resolve())
    from moderation_padding import install_agent_padding
    install_agent_padding(128)
    if target.agent.cfg['max_len']!=2048 or target.agent.cfg['head_max_len']!=256:
        raise ValueError('model budgets differ from protocol')
    warm=read_jsonl(DATA/'aegis_1000/development.jsonl')[0]
    for slug in manifest['test_counts']:
        dest=RESULTS/(slug+'_'+name+'.json')
        if dest.exists():raise FileExistsError(dest)
        folder=DATA/slug;path=folder/'heldout.jsonl';sha=validate_case_file(path,folder)
        cases=read_jsonl(path);rows=[]
        shortened=set(preflight[str(path.relative_to(ROOT))]['truncated_ids'])
        rec=Recorder(time.strftime('%Y%m%dT%H%M%S')+'_moderation_'+slug+'_'+name,'public/moderation-study',
                     meta={'dataset':slug,'target':name,'cases_sha256':sha,'config_sha256':target.config_hash,
                           'weights_sha256':target.identity,'sequence_length':2048,'head_max_len':256,'pad_to_multiple':128,
                           'warmup':'one Aegis development example per dataset','serial':True})
        rec.attach(__file__,folder/'manifest.json',DATA/'study_manifest.json')
        try:
            for i,c in enumerate([warm]+cases):
                start=time.perf_counter();answers={};usage={};resolved=target.identity;error=None;prediction=None;probs=None
                try:
                    answers,usage,resolved=target.ask(c);prediction,probs=parse_answer(answers,['safe','unsafe'])
                except Exception as exc:error=type(exc).__name__+': '+str(exc)[:250]
                latency=(time.perf_counter()-start)*1000
                rec.record('laya',name,c['state'],c['questions'],answers,model_resolved=resolved,case_id=c['id'],
                           split='development' if i==0 else 'heldout',expected=c['expected'],label_source=c['label_source'],
                           latency_ms=latency,usage=usage,error=error)
                if i==0:
                    if error:raise RuntimeError('warmup failed: '+error)
                    continue
                rows.append({'id':c['id'],'text_group':c['text_group'],'expected':c['expected']['intent'],
                             'prediction':prediction,'probabilities':probs,'error':error,'latency_ms':latency,'usage':usage,
                             'model_resolved':resolved,'categories':c.get('categories',[]),
                             'policy_category':c.get('policy_category'),'input_truncated':c['id'] in shortened})
                if i%100==0:print(slug,name,str(i)+'/'+str(len(cases)),flush=True)
                if len(rows)>=5 and all(r['error'] for r in rows[-5:]):raise RuntimeError('five consecutive failures')
            result={'dataset':slug,'target':name,'cases_sha256':sha,'manifest_sha256':digest((folder/'manifest.json').read_bytes()),
                    'study_sha256':digest((DATA/'study_manifest.json').read_bytes()),'checkpoint':str(checkpoint.resolve()),
                    'config_sha256':target.config_hash,'weights_identity':target.identity,'run_id':rec.run_id,'pad_to_multiple':128,
                    'summary':moderation_metrics(rows),'rows':rows}
            dest.write_text(json.dumps(result,indent=1)+'\n');rec.attach(dest)
            print('COMPLETE',slug,name,'F1',result['summary']['harmful_f1'],'accuracy',result['summary']['accuracy'],flush=True)
        finally:rec.close(outcome=f'evaluated {len(rows)}/{len(cases)}')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True,type=Path)
    p.add_argument('--name',required=True,choices=['base','fine1000','fine5000']);a=p.parse_args();evaluate(a.checkpoint,a.name)
