"""Build the public report and downloadable charts from recorded measurements."""
import json
import re
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from build_moderation_report import build_page, TARGETS
from public_workflow import ROOT

COLORS = ['#798995', '#24a39a', '#153f57']
LABELS = ['Before fine-tuning', '1,000 training decisions', '5,000 training decisions']

def charts(analysis):
    tasks = {t['slug']: t for t in analysis['tasks']}
    assets = ROOT/'assets'; assets.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,
                         'axes.spines.right':False,'axes.spines.left':False,'svg.fonttype':'none',
                         'svg.hashsalt':'laya-moderation-20260921'})
    names = [('aegis_prompt','Aegis · user input'),('aegis_response','Aegis · assistant response'),
             ('toxicchat','ToxicChat · user input'),('wildguard_prompt','WildGuard · user input'),
             ('wildguard_response','WildGuard · assistant response'),('beavertails','BeaverTails · assistant response'),
             ('openai_known','OpenAI · known-label subset*')]
    fig, ax = plt.subplots(figsize=(11,9.3),layout='constrained')
    yy=np.arange(len(names));width=.23
    for j,target in enumerate(TARGETS):
        values=[100*tasks[s]['runs'][target]['summary']['harmful_f1'] for s,_ in names]
        bars=ax.barh(yy+(j-1)*width,values,height=.20,color=COLORS[j],label=LABELS[j])
        ax.bar_label(bars,labels=[f'{v:.1f}%' for v in values],padding=4,fontsize=10)
    ax.set_yticks(yy,[n for _,n in names]);ax.invert_yaxis();ax.set_xlim(0,100)
    ax.set_xticks(range(0,101,20));ax.set_xlabel('Harmful-content F1 (%) · higher is better')
    ax.grid(axis='x',alpha=.16);ax.set_axisbelow(True);ax.tick_params(axis='y',length=0,pad=12)
    ax.legend(loc='lower center',bbox_to_anchor=(.4,1.01),ncol=1,frameon=False)
    fig.suptitle('Fine-tuning improves detection across these tests',fontsize=17,fontweight='bold')
    fig.supxlabel('*859 cases with known binary labels; not the published full-set OpenAI benchmark.',fontsize=10)
    save(fig,assets/'detection-f1')
    fig,axes=plt.subplots(1,2,figsize=(11,5.4),layout='constrained')
    for ax,key,title,subtitle in [(axes[0],'recall','Harmful prompts caught','Higher is better · 200 harmful prompts'),
                                  (axes[1],'false_positive_rate','Harmless prompts flagged','Lower is better · 250 harmless prompts')]:
        values=[100*tasks['xstest']['runs'][t]['summary']['harmful_class'][key] for t in TARGETS]
        bars=ax.bar(range(3),values,color=COLORS,width=.60);ax.bar_label(bars,labels=[f'{v:.1f}%' for v in values],padding=5,fontweight='bold')
        ax.set_ylim(0,105);ax.set_yticks(range(0,101,20));ax.set_xticks(range(3),['Before','After 1,000','After 5,000'])
        ax.set_title(title+'\n'+subtitle,fontsize=12,pad=15);ax.set_ylabel('% of prompts');ax.grid(axis='y',alpha=.16);ax.set_axisbelow(True)
    fig.suptitle('Better detection also brings more false alarms',fontsize=17,fontweight='bold')
    save(fig,assets/'false-alarm-tradeoff')

def save(fig,path):
    fig.savefig(path.with_suffix('.svg'),metadata={'Date':None})
    fig.savefig(path.with_suffix('.png'),dpi=170,metadata={'Software':'Laya fine-tuning study'})
    plt.close(fig)

def main():
    results=ROOT/'bench/results/moderation-study'
    load=lambda p:json.loads(p.read_text())
    analysis=load(results/'analysis.json')
    if not analysis.get('complete') or analysis.get('skipped'):raise ValueError('validated complete results required')
    study_path=ROOT/'train/moderation-study/study_manifest.json'
    if not study_path.exists():study_path=ROOT/'evidence/study_manifest.json'
    page=build_page(analysis,load(study_path),load(ROOT/'bench/moderation_references.json'),load(results/'data_audit.json'))
    charts(analysis)
    page=page.replace('local review draft','Laya fine-tuning field notes')
    page=page.replace('This report has not been publicly released. ','')
    page=page.replace('Prepared for owner review. This moderation experiment extends the earlier Jev and Laya workflow study.',
                      'Alexander Ollman · Independent research · September 2026. Code, source versions and recorded predictions are available on GitHub.')
    page=page.replace('href="../','href="')
    style='''<style>
    body{background:#f5f6f8}main{max-width:1120px}header{border-top:0;margin-top:30px}h1{max-width:900px}
    .topbar{background:#153f57;color:white}.topbar nav{max-width:1120px;margin:auto;padding:18px 32px;display:flex;gap:24px;flex-wrap:wrap;align-items:center}.topbar a{color:white;text-decoration:none;font-size:14px}.topbar strong{margin-right:auto}
    .stats{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin:28px 0}.stat{padding:20px;background:white;border:1px solid #dde3e9;border-radius:8px}.stat b{display:block;font-size:29px;color:#153f57}.stat span{font-size:13px;color:#566572}
    figure{margin:28px 0;background:white;border:1px solid #dde3e9;border-radius:10px;padding:20px}.chart{width:100%;height:auto;display:block}.chart-scroll{overflow:auto}figcaption{font-size:13px;color:#566572;margin-top:15px}.chart-links{display:flex;gap:16px;margin-top:10px;font-size:13px}.callout{border-radius:6px;background:#e9f1ef}.scroll{background:white;border-radius:6px}section{scroll-margin-top:20px}
    @media(max-width:600px){.topbar nav{padding:16px 18px;gap:16px}.stats{gap:8px}.stat{padding:12px 8px}.stat b{font-size:23px}.stat span{font-size:12px}figure{padding:12px}.chart{min-width:680px}.chart-scroll{border-bottom:1px solid #ddd}main{padding-top:20px}}
    </style>'''
    page=page.replace('</head>',style+'<meta name="description" content="A reproducible local Laya fine-tuning study: six safety datasets, two training sizes, detection gains and the cost of false alarms."></head>')
    nav='''<div class="topbar"><nav aria-label="Main navigation"><strong>Laya / field notes</strong><a href="#results">Results</a><a href="#false-alarms">False alarms</a><a href="https://github.com/Alexander-Ollman/laya-ft/blob/research/moderation-publication/docs/reproduce.md">Reproduce</a><a href="https://github.com/Alexander-Ollman/laya-ft">GitHub</a><a href="https://alexander-ollman.github.io/qwen3.8-on-rtx3090/">Qwen field notes</a></nav></div>'''
    page=page.replace('<body><main>','<body>'+nav+'<main>')
    page=page.replace('</header>','''<div class="stats"><div class="stat"><b>6</b><span>Public dataset sources</span></div><div class="stat"><b>67,890</b><span>Scored decisions</span></div><div class="stat"><b>0</b><span>Request failures</span></div></div></header>''')
    def figure(name,alt,caption):
        return f'<figure><div class="chart-scroll" tabindex="0" aria-label="Scrollable chart"><img class="chart" src="assets/{name}.svg" alt="{alt}"></div><figcaption>{caption} Exact values appear in the table below. On narrow screens, scroll the chart horizontally.</figcaption><div class="chart-links"><a href="assets/{name}.svg" download>Download SVG</a><a href="assets/{name}.png" download>Download PNG</a></div></figure>'
    page=page.replace('<h2>What changed with fine-tuning?</h2>','<h2 id="results">What changed with fine-tuning?</h2>')
    page=page.replace("<div class=\"scroll\"><table><thead><tr><th>Test</th><th>Examples</th>",figure('detection-f1','Grouped bars compare harmful-content F1 before training and after 1,000 and 5,000 decisions across seven tasks.','All three models use the same test cases. Aegis uses held-out data from the training source; other datasets test transfer.')+"<div class=\"scroll\"><table><thead><tr><th>Test</th><th>Examples</th>",1)
    page=page.replace('<h2>Does it block harmless requests?</h2>','<h2 id="false-alarms">Does it block harmless requests?</h2>'+figure('false-alarm-tradeoff','XSTest harmful recall rises from 61.5 to 84.0 to 91.5 percent, while harmless false alarms rise from 24.8 to 65.2 to 47.2 percent.','The larger fine-tune catches more harmful prompts, but still flags nearly half of the harmless prompts in this challenge set.'))
    # Link readable documentation on GitHub; JSON evidence remains directly downloadable.
    page=re.sub(r'href="docs/([^\"]+\.md)"',r'href="https://github.com/Alexander-Ollman/laya-ft/blob/research/moderation-publication/docs/\1"',page)
    page=page.replace('<footer>','<section><h2>Reproduce the results</h2><p>The repository includes training and evaluation code, pinned source versions, saved predictions, and verification tests. Start with the <a href="https://github.com/Alexander-Ollman/laya-ft/blob/research/moderation-publication/docs/reproduce.md">step-by-step reproduction guide</a>. Rescoring the recorded predictions requires no GPU or paid model calls. Download gated datasets through their original publishers with your own access.</p></section><footer>')
    (ROOT/'index.html').write_text(page)
    (ROOT/'.nojekyll').touch()
    print(ROOT/'index.html')

if __name__=='__main__':main()
