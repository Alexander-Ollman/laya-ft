"""Create a local review bundle of scores/code, excluding dataset text and weights."""
import hashlib
import json
import time
import zipfile
from pathlib import Path
from public_workflow import ROOT,digest
from telemetry import Recorder,ROOT as TELEMETRY_ROOT

def main():
    results=ROOT/'bench/results/moderation-study';data=ROOT/'train/moderation-study'
    status=json.loads((results/'status.json').read_text())
    if status['state']!='completed' or any(s.get('exit_code')!=0 for s in status['stages']):raise ValueError('run incomplete')
    analysis=json.loads((results/'analysis.json').read_text())
    study=json.loads((data/'study_manifest.json').read_text())
    expected_slugs=set(study['test_counts'])
    tasks={t['slug']:t for t in analysis['tasks']}
    if (analysis.get('complete') is not True or analysis.get('skipped') or set(tasks)!=expected_slugs
        or len(tasks)!=len(analysis['tasks']) or any(t['pending'] or set(t['runs'])!={'base','fine1000','fine5000'} for t in tasks.values())):
        raise ValueError('analysis incomplete')
    if analysis['study_sha256']!=digest((data/'study_manifest.json').read_bytes()) or analysis['preflight_sha256']!=digest((results/'preflight.json').read_bytes()):
        raise ValueError('analysis inputs changed')
    inventory={};manifests={}
    for name,folder in [('base','laya-moderation-base-2048'),('fine1000','laya-moderation-aegis-1000-s7'),('fine5000','laya-moderation-aegis-5000-s7')]:
        model=ROOT/'models'/folder;weights=model/'model.safetensors';config=model/'rl_agent_config.json'
        with weights.open('rb') as stream:sha=hashlib.file_digest(stream,'sha256').hexdigest()
        cfgsha=digest(config.read_bytes());inventory[name]={'weights_sha256':sha,'bytes':weights.stat().st_size,'config_sha256':cfgsha,'config':json.loads(config.read_text())}
        for slug in json.loads((data/'study_manifest.json').read_text())['test_counts']:
            rawpath=results/(slug+'_'+name+'.json')
            run=json.loads(rawpath.read_text())
            if tasks[slug]['runs'][name]['input_sha256']!=digest(rawpath.read_bytes()):raise ValueError('stale analysis '+slug+' '+name)
            if run['weights_identity']!='sha256:'+sha or run['config_sha256']!=cfgsha:raise ValueError('checkpoint changed '+name)
            paths=list(TELEMETRY_ROOT.glob('*/'+run['run_id']+'/manifest.json'))
            if len(paths)!=1:raise ValueError('evaluation manifest missing '+run['run_id'])
            m=json.loads(paths[0].read_text())
            if m['meta']['config_sha256']!=cfgsha or m['meta']['cases_sha256']!=run['cases_sha256']:raise ValueError('archived run mismatch')
            n=study['test_counts'][slug]
            if (run.get('pad_to_multiple')!=128 or m['meta'].get('pad_to_multiple')!=128
                or m['meta']['weights_sha256']!=run['weights_identity'] or m['prompt_records']!=n+1
                or m['outcome']!=f'evaluated {n}/{n}' or m['errors']!=sum(bool(r['error']) for r in run['rows'])):
                raise ValueError('archived run incomplete or differs from protocol')
            manifests[run['run_id']]=m
    (results/'checkpoint_manifest.json').write_text(json.dumps(inventory,indent=2)+'\n')
    (results/'evaluation_manifests.json').write_text(json.dumps(manifests,indent=2)+'\n')
    modules=['public_workflow.py','analyze_public_study.py','evaluate_workflow_suite.py','prepare_workflow_suite.py',
             'train_laya.py','train_public_laya.py','telemetry.py','fetch_moderation_sources.py','prepare_moderation_study.py',
             'preflight_moderation.py','moderation_metrics.py','moderation_padding.py','evaluate_moderation.py','run_moderation_study.py',
             'analyze_moderation.py','build_moderation_report.py','package_moderation_study.py','restore_moderation_evidence.py','test_moderation_replay.py',
             'test_moderation_metrics.py','test_moderation_preparation.py','test_moderation_analysis.py','test_moderation_padding.py']
    paths=[ROOT/'bench'/p for p in modules]+[ROOT/'bench/moderation_references.json',ROOT/'bench/moderation_sources.json',
           ROOT/'report/laya-moderation-study.html',ROOT/'docs/moderation-study-protocol.md',ROOT/'docs/moderation-reference-notes.md',
           data/'prior_exclusion_hashes.json']
    paths+=list(results.rglob('*.json'))+list(results.rglob('*.log'))
    assets={str(p.relative_to(ROOT)):p.read_bytes() for p in paths}
    assets['evidence/study_manifest.json']=(data/'study_manifest.json').read_bytes()
    for p in data.glob('*/manifest.json'):assets['evidence/'+p.parent.name+'/manifest.json']=p.read_bytes()
    assets['README.md']=(
        '# Local Laya chat moderation study\n\n'
        'Open report/laya-moderation-study.html. This bundle is for owner review; nothing has been published.\n\n'
        'Includes local predictions, code, source versions, checkpoint hashes and published reference scores. '
        'Dataset text and model weights are excluded. WildGuard requires authorized Hugging Face access.\n\n'
        'Use package versions in bench/results/moderation-study/runtime.json. From bench/, run:\n\n'
        '```sh\npython fetch_moderation_sources.py\npython prepare_moderation_study.py\n'
        'python restore_moderation_evidence.py --data ../train/moderation-study --evidence ../evidence\n'
        'python -m unittest test_moderation_metrics test_moderation_preparation test_moderation_analysis test_moderation_padding test_moderation_replay\n'
        'python analyze_moderation.py\npython build_moderation_report.py\n```\n\n'
        'Compare regenerated manifests with evidence/. Rescoring uses saved predictions and needs no GPU or provider calls. '
        'The historical execution lane is complete; do not restart it over existing results. '
        'For new training, use train_public_laya.py --data ../train/moderation-study/aegis_1000 --out <new path> --pad-to-multiple 128 --mps-cache-clear-interval 1, '
        'and repeat for aegis_5000. Full method and limits: docs/moderation-study-protocol.md.\n').encode()
    sums={n:digest(blob) for n,blob in sorted(assets.items())}
    destination=ROOT/'report/laya-moderation-review-2026-09-21.zip'
    with zipfile.ZipFile(destination,'x',compression=zipfile.ZIP_DEFLATED) as z:
        for name,blob in sorted(assets.items()):z.writestr(name,blob)
        z.writestr('SHA256SUMS.json',json.dumps(sums,indent=2)+'\n')
    with zipfile.ZipFile(destination) as z:
        if z.testzip():raise ValueError('corrupt archive')
        for name,sha in sums.items():
            if digest(z.read(name))!=sha:raise ValueError('checksum mismatch')
        if any('/raw/' in n or n.endswith(('.jsonl','.safetensors','.env')) for n in z.namelist()):raise ValueError('unexpected data in review bundle')
    rec=Recorder(time.strftime('%Y%m%dT%H%M%S')+'_moderation_review','public/moderation-review',meta={'zip_sha256':digest(destination.read_bytes()),'files':len(assets)})
    rec.attach(destination,ROOT/'docs/moderation-study-protocol.md');rec.close(outcome='complete local review bundle; not published')
    print(destination)

if __name__=='__main__':main()
