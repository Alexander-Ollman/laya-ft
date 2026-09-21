"""Build an updated local evidence ZIP without redistributing restricted text."""
import hashlib,json,time,zipfile
from pathlib import Path
from public_workflow import ROOT,digest
from telemetry import Recorder, ROOT as TELEMETRY_ROOT


def main():
    results=ROOT/'bench/results/workflow-suite'
    analysis=json.loads((results/'analysis.json').read_text())
    if len(analysis['tasks'])!=5 or any(t['pending'] or not t['basecomparison'] or not t['jevcomparison'] for t in analysis['tasks']):
        raise RuntimeError('All fifteen task/model evaluations and comparisons must be complete')
    for lane in ('local','jev'):
        status=json.loads((results/(lane+'_status.json')).read_text())
        if status['state']!='completed' or any(s.get('exit_code')!=0 for s in status['stages']):
            raise RuntimeError('Execution lane incomplete: '+lane)
    inventory={}; evaluation_manifests={}
    base=ROOT/'models/laya-english-wide'
    with (base/'model.safetensors').open('rb') as f:base_sha=hashlib.file_digest(f,'sha256').hexdigest()
    for task in analysis['tasks']:
        slug=task['slug'];folder=ROOT/'models'/('laya-suite-'+slug+'-s7')
        config=folder/'rl_agent_config.json';weights=folder/'model.safetensors'
        with weights.open('rb') as f:sha=hashlib.file_digest(f,'sha256').hexdigest()
        identity=task['runs']['laya:fine']['model_resolved']
        if identity!=['sha256:'+sha]:raise ValueError('evaluated model differs from saved checkpoint: '+slug)
        inventory[slug]={'weights_sha256':sha,'bytes':weights.stat().st_size,'config_sha256':digest(config.read_bytes()),'config':json.loads(config.read_text())}
        for target,run in task['runs'].items():
            runid=Path(run['input']).stem
            matches=list(TELEMETRY_ROOT.glob('*/'+runid+'/manifest.json'))
            if len(matches)!=1:raise ValueError('missing or ambiguous evaluation manifest: '+runid)
            manifest=json.loads(matches[0].read_text())
            if manifest['meta']['cases_sha256']!=task['cases_sha256']:raise ValueError('evaluation cases mismatch')
            if target.startswith('laya:'):
                checkpoint=folder if target=='laya:fine' else base
                expected=sha if target=='laya:fine' else base_sha
                if run['model_resolved']!=['sha256:'+expected]:raise ValueError('evaluation weights mismatch')
                if manifest['meta']['config_sha256']!=digest((checkpoint/'rl_agent_config.json').read_bytes()):raise ValueError('evaluation config mismatch')
            evaluation_manifests[runid]=manifest
    (results/'checkpoint_manifest.json').write_text(json.dumps(inventory,indent=2)+'\n')
    (results/'evaluation_manifests.json').write_text(json.dumps(evaluation_manifests,indent=2)+'\n')
    old=ROOT/'report/jev-laya-public-study-2026-09-19.zip'
    destination=ROOT/'report/jev-laya-six-dataset-review-2026-09-20.zip'
    modules=['build_public_report.py','render_workflow_suite.py','prepare_workflow_suite.py','evaluate_workflow_suite.py',
      'train_public_laya.py','run_workflow_suite.py','analyze_workflow_suite.py','audit_workflow_similarity.py','package_workflow_suite.py',
      'test_suite_training.py','test_workflow_suite.py','test_workflow_analysis.py']
    paths=[ROOT/'bench'/name for name in modules]
    paths += [ROOT/'report/jev-vs-laya.html',ROOT/'docs/workflow-suite-protocol.md',ROOT/'bench/results/workflow_suite_preflight.json',
              ROOT/'bench/results/workflow_suite_runtime.json',ROOT/'train/workflow-suite/prior_holdout_hashes.json']
    paths += list(results.glob('*.json'))+list(results.glob('*_train.log'))
    with zipfile.ZipFile(old) as source:
        assets={n:source.read(n) for n in source.namelist() if n not in ('SHA256SUMS.json','REPRODUCE.txt','bench/results/public_transfer_controls.jsonl')}
    for path in paths:assets[str(path.relative_to(ROOT))]=path.read_bytes()
    assets['README.md']=(
      '# Jev vs. fine-tuned Laya\n\n'
      'Local review package covering Banking77 and five further public workflow datasets.\n\n'
      'Open [the report](report/jev-vs-laya.html). See [the extension protocol](docs/workflow-suite-protocol.md) '
      'and REPRODUCE.txt for source downloads, measurement details and verification.\n\n'
      'Results include gains, uncertainty and limitations. The report has not been publicly released.\n').encode()
    for task in analysis['tasks']:
        # Manifests go in evidence, not preparation's output directory: rerun can
        # fetch licensed sources and recreate frozen splits without overwrite.
        folder=ROOT/'train/workflow-suite'/task['slug']
        assets['evidence/workflow-suite/'+task['slug']+'/manifest.json']=(folder/'manifest.json').read_bytes()
    sums={name:digest(blob) for name,blob in sorted(assets.items())}
    with zipfile.ZipFile(destination,'x',compression=zipfile.ZIP_DEFLATED) as archive:
        for name,blob in sorted(assets.items()):archive.writestr(name,blob)
        archive.writestr('SHA256SUMS.json',json.dumps(sums,indent=2)+'\n')
        archive.writestr('REPRODUCE.txt',
          'LOCAL REVIEW PACKAGE. No public release has occurred.\n'
          'Open report/jev-vs-laya.html. Full protocol: docs/workflow-suite-protocol.md.\n'
          'The original Banking77 assets are included under CC BY4.0.\n'
          'The five extension datasets must be fetched from their pinned sources under their respective terms.\n'
          'No extension source text or model weights are included.\n'
          'The older transfer-control text file is also omitted; its saved scores remain.\n'
          'Install versions in bench/results/public_runtime.json and workflow_suite_runtime.json.\n'
          'From bench/: python prepare_workflow_suite.py\n'
          'Check generated manifests against evidence/workflow-suite/<task>/manifest.json.\n'
          'From bench/: python -m unittest test_public_workflow test_public_analysis test_suite_training test_workflow_suite test_workflow_analysis\n'
          'From bench/: python analyze_workflow_suite.py\n'
          'From bench/: python audit_workflow_similarity.py\n'
          'From bench/: python build_public_report.py\n'
          'Saved predictions can be rescored without new provider calls or model weights after fetching data.\n'
          'Execution status files are historical audit evidence; do not blindly rerun the session runner over them.\n'
          'For new training, use train_public_laya.py --data ../train/workflow-suite/<task> --out <new checkpoint path>.\n'
          'Set JEVTEST_TELEMETRY_ROOT to choose a local archive directory.\n')
    with zipfile.ZipFile(destination) as archive:
        if archive.testzip() is not None:raise ValueError('ZIP validation failed')
        for name,expected in sums.items():
            if digest(archive.read(name))!=expected:raise ValueError('checksum mismatch '+name)
        if any('/raw/' in n or n.endswith('.env') for n in archive.namelist()):raise ValueError('unexpected raw data/credential file')
    rec=Recorder(time.strftime('%Y%m%dT%H%M%S')+'_six_dataset_review','public/workflow-suite-report',meta={'zip_sha256':digest(destination.read_bytes()),'files':len(sums)})
    rec.attach(destination,ROOT/'docs/workflow-suite-protocol.md');rec.close(outcome='local review report and evidence package complete; not published')
    print(destination)

if __name__=='__main__':main()
