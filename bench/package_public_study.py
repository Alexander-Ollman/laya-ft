"""Package the completed report and public reproducibility assets locally.

No upload, Git operation, credentials, private training data, or model weights.
"""
import hashlib
import json
import time
import zipfile
from pathlib import Path
from public_workflow import DATA, ROOT
from telemetry import Recorder


def file_sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    result_dir = ROOT/'bench/results'
    analysis = json.loads((result_dir/'public_study_analysis.json').read_text())
    if not analysis.get('paired_comparison') or len(analysis.get('calibration', [])) < 2:
        raise RuntimeError('Complete the public comparison and calibration before packaging')
    inputs = [Path(r['input']) for r in analysis['runs']]
    controls = []
    for path in sorted(result_dir.glob('2*_laya_*.json')):
        data = json.loads(path.read_text())
        if Path(data.get('cases_file', '')).name == 'public_transfer_controls.jsonl':
            controls.append(path)
    if len(controls) < 3:
        raise RuntimeError('Complete the public transfer controls before packaging')
    checkpoint_names = ['laya-era-distill-1k', 'laya-era-distill-full', 'laya-english-wide', 'laya-banking77-gold-1k']
    checkpoints = {}
    for name in checkpoint_names:
        path = ROOT/'models'/name
        weights = path/'model.safetensors'
        config = path/'rl_agent_config.json'
        checkpoints[name] = {'weights_sha256': file_sha(weights), 'config_sha256': file_sha(config),
                             'bytes': weights.stat().st_size,
                             'configuration': json.loads(config.read_text())}
    (result_dir/'public_checkpoint_manifest.json').write_text(json.dumps(checkpoints, indent=2)+'\n')
    modules = ['public_workflow.py', 'train_public_laya.py', 'train_laya.py', 'run_bench.py',
               'serve_laya.py', 'telemetry.py', 'analyze_public_study.py', 'build_public_report.py',
               'test_public_workflow.py', 'test_public_analysis.py', 'package_public_study.py']
    paths = [ROOT/'bench'/name for name in modules]
    paths += [ROOT/'README.md', ROOT/'docs/public-benchmark-protocol.md', ROOT/'report/jev-vs-laya.html']
    paths += list(DATA.glob('*')) + inputs + controls
    paths += [result_dir/name for name in ['public_study_analysis.json', 'public_runtime.json',
              'public_jev_provenance.json', 'public_checkpoint_manifest.json', 'public_transfer_controls.jsonl']]
    paths.append(result_dir/'continuation_comparison.txt')
    paths = sorted({path for path in paths if path.is_file()})
    manifest = {str(path.relative_to(ROOT)): file_sha(path) for path in paths}
    archive = ROOT/'report/jev-laya-public-study-2026-09-19.zip'
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED) as out:
        for path in paths:
            out.write(path, str(path.relative_to(ROOT)))
        out.writestr('SHA256SUMS.json', json.dumps(manifest, indent=2)+'\n')
        out.writestr('REPRODUCE.txt',
            'Open report/jev-vs-laya.html. See docs/public-benchmark-protocol.md.\n'
            'The public Banking77 path needs no private corpora.\n'
            'Use Python/package versions in bench/results/public_runtime.json.\n'
            'From bench/: python -m unittest test_public_workflow test_public_analysis\n'
            'From bench/: python train_public_laya.py --out ../models/laya-banking77-gold-1k\n'
            'Register that checkpoint using LAYA_EXTRA for serve_laya:app.\n'
            'Set JEVTEST_TELEMETRY_ROOT to choose a local archive directory.\n'
            'Weights are separate; public_checkpoint_manifest.json records their hashes.\n'
            'Historical private results in the report supplement are exploratory and their data are not bundled.\n'
            'This is a local review package; public release has not occurred.\n')
    with zipfile.ZipFile(archive) as check:
        assert check.testzip() is None
        assert not any(Path(name).name == '.env' for name in check.namelist())
        for name, expected in manifest.items():
            assert hashlib.sha256(check.read(name)).hexdigest() == expected
    rec = Recorder(time.strftime('%Y%m%dT%H%M%S')+'_public_study_bundle', 'public/study-report',
                   meta={'archive_sha256': file_sha(archive), 'files': manifest})
    rec.attach(archive, result_dir/'public_study_status.json', result_dir/'train_full.log',
               result_dir/'train_public_banking77.log')
    rec.close(outcome='report and public reproducibility package prepared; not published')
    print(archive)


if __name__ == '__main__':
    main()
