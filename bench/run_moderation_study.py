"""One sequential GPU lane; status/logs persist independently of the UI."""
import json
import subprocess
import sys
import time
from pathlib import Path
from public_workflow import ROOT

def main():
    results=ROOT/'bench/results/moderation-study';status_path=results/'status.json'
    if status_path.exists():raise FileExistsError('moderation lane already started')
    if not (results/'preflight.json').exists():raise ValueError('preflight required')
    base=ROOT/'models/laya-moderation-base-2048'
    source=ROOT/'models/laya-english-wide'
    cfg=json.loads((source/'rl_agent_config.json').read_text());cfg.update(max_len=2048,head_max_len=256)
    if base.exists():
        if json.loads((base/'rl_agent_config.json').read_text())!=cfg:raise ValueError('existing base config differs')
        for filename in ('encoder','tokenizer','model.safetensors'):
            if (base/filename).resolve()!=(source/filename).resolve():raise ValueError('existing base asset differs')
    else:
        base.mkdir()
        for filename in ('encoder','tokenizer','model.safetensors'):(base/filename).symlink_to(source/filename)
        (base/'rl_agent_config.json').write_text(json.dumps(cfg,indent=1)+'\n')
    stages=[]
    for n in (1000,5000):
        stages.append(('train'+str(n),['train_public_laya.py','--data',str(ROOT/'train/moderation-study'/('aegis_'+str(n))),
                      '--out',str(ROOT/'models'/('laya-moderation-aegis-'+str(n)+'-s7')),'--pad-to-multiple','128']))
    for name,model in [('base',base)]+[('fine'+str(n),ROOT/'models'/('laya-moderation-aegis-'+str(n)+'-s7')) for n in (1000,5000)]:
        stages.append(('evaluate_'+name,['evaluate_moderation.py','--name',name,'--checkpoint',str(model)]))
    status={'state':'running','started':time.strftime('%Y-%m-%dT%H:%M:%S%z'),'stages':[]}
    def save():status_path.write_text(json.dumps(status,indent=2)+'\n')
    save()
    for name,command in stages:
        start=time.time();entry={'name':name,'command':command,'started':start};status['stages'].append(entry)
        with (results/(name+'.log')).open('x') as log:
            p=subprocess.Popen([sys.executable,'-u']+command,cwd=ROOT/'bench',stdout=log,stderr=subprocess.STDOUT)
            entry['pid']=p.pid;save();code=p.wait()
        entry.update(exit_code=code,seconds=round(time.time()-start,1));save()
        if code:
            status['state']='failed';save();raise RuntimeError('stage failed: '+name)
    status['state']='completed';status['finished']=time.strftime('%Y-%m-%dT%H:%M:%S%z');save()

if __name__=='__main__':main()
