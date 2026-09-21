"""Fetch pinned source data with checksum verification; no dataset scripts."""
import json
from pathlib import Path
import httpx
from public_workflow import ROOT,digest

def main():
    manifest=json.loads(Path(__file__).with_name('moderation_sources.json').read_text())
    folder=ROOT/'train/moderation-study'
    for item in manifest['files']:
        dest=folder/item['path'];dest.parent.mkdir(parents=True,exist_ok=True)
        if dest.exists():blob=dest.read_bytes()
        elif 'repo' in item:
            from huggingface_hub import hf_hub_download
            source=hf_hub_download(item['repo'],item['filename'],repo_type='dataset',revision=item['revision'])
            blob=Path(source).read_bytes()
        else:
            r=httpx.get(item['url'],follow_redirects=True,timeout=120);r.raise_for_status();blob=r.content
        if digest(blob)!=item['sha256']:raise ValueError('source checksum mismatch '+item['path'])
        if not dest.exists():dest.write_bytes(blob)
        print('verified',item['path'])
    blob=(json.dumps(manifest['raw_sources_record'],indent=2)+'\n').encode()
    record=folder/'raw/sources.json'
    if record.exists() and record.read_bytes()!=blob:raise ValueError('source metadata mismatch')
    if not record.exists():record.write_bytes(blob)

if __name__=='__main__':main()
