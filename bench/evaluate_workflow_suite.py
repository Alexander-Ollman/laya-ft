"""Evaluate one frozen workflow split using local inference or the Jev API."""
import argparse
import hashlib
import json
import math
import re
import time
from pathlib import Path
from public_workflow import ROOT, digest, norm, read_jsonl, parse_answer, summarize, validate_case_file
from telemetry import Recorder


def wilson(success,total):
    if not total:return None
    p=success/total;z=1.959963984540054;den=1+z*z/total
    center=(p+z*z/(2*total))/den;half=z*math.sqrt(p*(1-p)/total+z*z/(4*total*total))/den
    return [center-half,center+half]


def detailed_metrics(rows,labels,positive=None):
    result=summarize(rows,labels);per={}
    for label in labels:
        tp=sum(r['expected']==label and r['prediction']==label and not r['error'] for r in rows)
        fp=sum(r['expected']!=label and r['prediction']==label and not r['error'] for r in rows)
        tn=sum(r['expected']!=label and r['prediction'] not in (label,None) and not r['error'] for r in rows)
        support=sum(r['expected']==label for r in rows);fn=support-tp;negative=len(rows)-support
        per[label]={'support':support,'negative_support':negative,'tp':tp,'fp':fp,'fn':fn,'tn':tn,'precision':tp/(tp+fp) if tp+fp else None,
          'recall':tp/support if support else None,'recall_wilson_95':wilson(tp,support),
          'false_positive_rate':fp/negative if negative else None,'false_positive_wilson_95':wilson(fp,negative)}
    result['per_class']=per;result['majority_baseline']=max(v['support'] for v in per.values())/len(rows)
    if positive:result['safety_positive_class']={'label':positive,**per[positive]}
    return result


class LocalTarget:
    def __init__(self,checkpoint):
        import laya
        self.agent=laya.load(str(checkpoint));self.model=checkpoint.name
        with (checkpoint/'model.safetensors').open('rb') as f:self.identity='sha256:'+hashlib.file_digest(f,'sha256').hexdigest()
        self.config_hash=digest((checkpoint/'rl_agent_config.json').read_bytes())
    def ask(self,case):
        out=self.agent.system_one(case['state'],case['questions'])
        return out['answers'],out.get('usage',{}),self.identity


def evaluate(args):
    data=args.data.resolve();manifest=json.loads((data/'manifest.json').read_text())
    casespath=data/(args.split+'.jsonl');caseshash=validate_case_file(casespath,data)
    cases=read_jsonl(casespath);labels=manifest['labels'];slug=manifest['slug']
    if args.checkpoint:target=LocalTarget(args.checkpoint.resolve())
    else:
        from run_bench import load_env,make_target
        load_env();target=make_target(args.target)
    runid=time.strftime('%Y%m%dT%H%M%S')+'_suite_'+slug+'_'+args.split+'_'+re.sub(r'[^A-Za-z0-9.-]','_',args.target)
    output=ROOT/'bench/results/workflow-suite';output.mkdir(exist_ok=True)
    resultpath=output/(runid+'.json')
    rec=Recorder(runid,'public/workflow-suite/'+slug,meta={'dataset':manifest['dataset'],'cases_sha256':caseshash,'target':args.target,'protocol':'workflow-suite-v1','warmup':'first development request','config_sha256':getattr(target,'config_hash',None)})
    rec.attach(__file__,data/'manifest.json');rows=[];warm=read_jsonl(data/'development.jsonl')[0]
    try:
        for i,case in enumerate([warm]+cases):
            start=time.perf_counter();answers={};usage={};resolved=None;error=None;prediction=None;probs=None
            try:
                answers,usage,resolved=target.ask(case);prediction,probs=parse_answer(answers,labels)
            except Exception as exc:error=type(exc).__name__+': '+str(exc)[:250]
            latency=(time.perf_counter()-start)*1000
            rec.record('laya' if args.checkpoint else 'jev',target.model,case['state'],case['questions'],answers,
              model_resolved=resolved,case_id=case['id'],split=case['split'],expected=case['expected'],label_source=case['label_source'],latency_ms=latency,usage=usage,error=error)
            if i==0:
                if error:raise RuntimeError('warmup failed: '+error)
                continue
            rows.append({'id':case['id'],'text_group':digest(norm(case['state']['request']).encode()),'expected':case['expected']['intent'],'prediction':prediction,'probabilities':probs,'error':error,'latency_ms':latency,'usage':usage,'model_resolved':resolved})
            if i%100==0:print(f'{slug} {args.target} {i}/{len(cases)}',flush=True)
            if len(rows)>=5 and all(r['error'] for r in rows[-5:]):raise RuntimeError('five consecutive request failures')
        result={'dataset':slug,'target':args.target,'split':args.split,'cases':str(casespath),'cases_sha256':caseshash,
          'manifest_sha256':digest((data/'manifest.json').read_bytes()),'summary':detailed_metrics(rows,labels,manifest['protocol'].get('positive_class')),'rows':rows}
        resultpath.write_text(json.dumps(result,indent=1)+'\n');rec.attach(resultpath)
        print(json.dumps({k:v for k,v in result['summary'].items() if k!='per_class'},indent=2),flush=True)
        print('saved',resultpath,flush=True)
    finally:rec.close(outcome=f'evaluated {len(rows)}/{len(cases)}')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--target',required=True)
    p.add_argument('--checkpoint',type=Path);p.add_argument('--split',choices=['heldout','development'],default='heldout')
    evaluate(p.parse_args())
