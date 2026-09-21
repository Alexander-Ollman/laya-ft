"""Tokenizer-only audit of complete frozen inputs; never infer labels."""
import json
from transformers import AutoTokenizer
from laya.common import build_sequence
from public_workflow import ROOT,read_jsonl,validate_case_file

def main():
    data=ROOT/'train/moderation-study'
    if not (data/'study_manifest.json').exists():raise ValueError('preparation incomplete')
    tok=AutoTokenizer.from_pretrained(str(ROOT/'models/laya-english-wide/tokenizer'));out={}
    for p in sorted(data.glob('*/*.jsonl')):
        validate_case_file(p,p.parent)
        rows=read_jsonl(p);short=[];lengths=[]
        for c in rows:
            q=c['questions']['intent'];q={'t':'choice','ins':q['instructions'],'crit':q['criteria']}
            full,_=build_sequence(tok,c['state'],q,100000,100000)
            actual,ma=build_sequence(tok,c['state'],q,2048,256)
            fullhead,_=build_sequence(tok,{},q,100000,100000)
            shorthead,_=build_sequence(tok,{},q,100000,256)
            if len(ma)!=2 or fullhead!=shorthead:raise ValueError('question truncation '+c['id'])
            if len(full)>len(actual):short.append(c['id'])
            lengths.append(len(full))
        out[str(p.relative_to(ROOT))]={'n':len(rows),'input_truncated':len(short),'truncated_ids':short,
                                     'head_truncated':0,'max_tokens':max(lengths)}
    folder=ROOT/'bench/results/moderation-study';folder.mkdir(exist_ok=True)
    (folder/'preflight.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({k:{n:v for n,v in val.items() if n!='truncated_ids'} for k,val in out.items()},indent=1))

if __name__=='__main__':main()
