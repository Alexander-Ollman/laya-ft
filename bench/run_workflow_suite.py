"""Two auditable lanes: sequential local GPU stages, and hosted Jev evaluations."""
import argparse,json,subprocess,sys,time
from pathlib import Path
from public_workflow import ROOT

SLUGS=['injection','sms','emotion','counterfactual','massive']

def main():
    p=argparse.ArgumentParser();p.add_argument('--lane',choices=['local','jev'],required=True);a=p.parse_args()
    output=ROOT/'bench/results/workflow-suite';output.mkdir(exist_ok=True)
    statuspath=output/(a.lane+'_status.json')
    if statuspath.exists():raise FileExistsError('Review existing lane status before a deliberate recovery')
    status={'lane':a.lane,'state':'running','started':time.strftime('%Y-%m-%dT%H:%M:%S%z'),'stages':[]}
    def save():statuspath.write_text(json.dumps(status,indent=2)+'\n')
    def run(name,command):
        item={'name':name,'command':command,'started':time.time()};status['stages'].append(item);save()
        print('starting',name,flush=True)
        with (output/(name+'.log')).open('x') as log:
            proc=subprocess.Popen([sys.executable,'-u',*command],cwd=ROOT/'bench',stdout=log,stderr=subprocess.STDOUT)
            item['pid']=proc.pid;save();code=proc.wait()
        item.update(exit_code=code,seconds=round(time.time()-item['started'],1));save()
        if code:raise RuntimeError(name+' failed; inspect its log')
    save()
    try:
        for slug in SLUGS:
            data=str(ROOT/'train/workflow-suite'/slug)
            if a.lane=='jev':run(slug+'_jev',['evaluate_workflow_suite.py','--data',data,'--target','jev'])
            else:
                run(slug+'_base',['evaluate_workflow_suite.py','--data',data,'--target','laya:base','--checkpoint',str(ROOT/'models/laya-english-wide')])
                model=ROOT/'models'/('laya-suite-'+slug+'-s7')
                run(slug+'_train',['train_public_laya.py','--data',data,'--out',str(model),'--seed','7'])
                run(slug+'_fine',['evaluate_workflow_suite.py','--data',data,'--target','laya:fine','--checkpoint',str(model)])
        status['state']='completed'
    except BaseException as exc:status.update(state='failed',error=str(exc));raise
    finally:save()

if __name__=='__main__':main()
