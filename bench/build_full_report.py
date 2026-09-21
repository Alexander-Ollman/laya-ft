#!/usr/bin/env python3
"""Build the combined publication from the preserved report and saved analyses.

Run from any directory: python bench/build_full_report.py
Requires matplotlib only for the four generated chart files. No inference runs.
"""
from pathlib import Path
import html
import hashlib
import json
import re

ROOT = Path(__file__).resolve().parents[1]
COLORS = ['#98aaa8', '#247a72', '#bc7649']
TASK_NAMES = {
    'injection': 'Prompt injection', 'sms': 'SMS spam', 'sms_spam': 'SMS spam',
    'emotion': 'Emotion', 'counterfactual': 'Product feedback',
    'amazon_counterfactual': 'Product feedback', 'massive': 'Assistant routing',
    'aegis_prompt': 'Aegis · prompts', 'aegis_response': 'Aegis · responses',
    'toxicchat': 'ToxicChat · prompts', 'wildguard_prompt': 'WildGuard · prompts',
    'wildguard_response': 'WildGuard · responses', 'beavertails': 'BeaverTails · responses',
    'openai_known': 'OpenAI Moderation · known binary labels',
    'openai_categories': 'OpenAI Moderation · known category decisions', 'xstest': 'XSTest · prompts',
}


def read(relative):
    return json.loads((ROOT / relative).read_text())


def pct(value):
    return f'{100 * value:.1f}%'


def table(headers, rows):
    return '<div class="scroll"><table><thead><tr>' + ''.join(f'<th scope="col">{h}</th>' for h in headers) + '</tr></thead><tbody>' + ''.join('<tr>' + ''.join(f'<td>{v}</td>' for v in row) + '</tr>' for row in rows) + '</tbody></table></div>'


def figure(name, alt, caption):
    return f'<figure><div class="chart-scroll" tabindex="0" aria-label="Scrollable chart"><a href="assets/{name}.svg"><img src="assets/{name}.svg" alt="{html.escape(alt)}" loading="lazy"></a></div><figcaption>{caption} Scroll horizontally on a narrow screen. <a href="assets/{name}.png">Download PNG</a> · <a href="assets/{name}.svg">SVG</a></figcaption></figure>'


def charts(workflow, banking):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11, 'svg.hashsalt': 'laya-ft-full-report', 'axes.spines.top': False, 'axes.spines.right': False})
    tasks = workflow['tasks']
    fig, ax = plt.subplots(figsize=(11.5, 6.3), layout='constrained')
    y = np.arange(len(tasks))
    for offset, key, label, color in zip([-.25, 0, .25], ['laya:base', 'laya:fine', 'jev'], ['Laya before', 'Laya fine-tuned', 'Jev'], COLORS):
        values = [t['runs'][key]['summary']['macro_f1'] * 100 for t in tasks]
        bars = ax.barh(y + offset, values, height=.23, color=color, label=label)
        ax.bar_label(bars, labels=[f'{v:.1f}' for v in values], padding=4, fontsize=9)
    ax.set_yticks(y, [TASK_NAMES.get(t['slug'], t['slug']) for t in tasks])
    ax.invert_yaxis(); ax.set_xlim(0, 110); ax.set_xticks(range(0, 101, 20)); ax.set_xlabel('Macro-F1 (%) · higher is better')
    ax.set_title('Five public tasks · one fine-tune per task', loc='left', weight='bold', pad=25)
    ax.legend(loc='lower center', bbox_to_anchor=(.5, 1.0), ncol=3, frameon=False)
    ax.grid(axis='x', alpha=.14); ax.set_axisbelow(True)
    save(fig, 'workflow-comparison')
    runs = {r['target']: r['summary'] for r in banking['runs'] if r['split'] == 'heldout'}
    keys = ['laya:laya-english', 'laya:laya-english-wide', 'laya:laya-banking77-gold-1k', 'jev']
    values = [runs[k]['accuracy'] * 100 for k in keys]
    fig, ax = plt.subplots(figsize=(11.5, 4.7), layout='constrained')
    bars = ax.barh(['Laya · default settings', 'Laya · wider answer options', 'Laya · fine-tuned (1,001 examples)', 'Jev'], values, color=['#cad2d1', *COLORS], height=.62)
    ax.bar_label(bars, labels=[f'{v:.1f}%' for v in values], padding=6)
    ax.invert_yaxis(); ax.set_xlim(0, 100); ax.set_xlabel('Accuracy (%) · all 3,080 test questions')
    ax.set_title('Banking77 · training closes most of the gap', loc='left', weight='bold', pad=18)
    ax.grid(axis='x', alpha=.14); ax.set_axisbelow(True)
    save(fig, 'banking-comparison')


def save(fig, name):
    import matplotlib.pyplot as plt
    (ROOT / 'assets').mkdir(exist_ok=True)
    fig.savefig(ROOT / f'assets/{name}.svg', metadata={'Date': None}, bbox_inches='tight')
    fig.savefig(ROOT / f'assets/{name}.png', dpi=180, bbox_inches='tight')
    plt.close(fig)


def moderation_chart(tasks, jev_summary, order):
    import matplotlib.pyplot as plt
    import numpy as np
    y = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(12, 8.5), layout='constrained')
    columns = [('base', 'Laya before', '#98aaa8'), ('fine1000', 'Laya · 1k', '#6d9bac'), ('fine5000', 'Laya · 5k', '#247a72'), ('jev', 'Jev', '#bc7649')]
    for j, (key, label, color) in enumerate(columns):
        values = []
        positions = []
        for i, slug in enumerate(order):
            summary = jev_summary(slug) if key == 'jev' else tasks[slug]['runs'][key]['summary']
            if summary is not None:
                values.append(summary['harmful_f1'] * 100)
                positions.append(y[i] + (j - 1.5) * .2)
        if values:
            bars = ax.barh(positions, values, height=.18, color=color, label=label)
            ax.bar_label(bars, labels=[f'{v:.1f}' for v in values], fontsize=8, padding=3)
    labels = [TASK_NAMES[s].replace('OpenAI Moderation · known binary labels', 'OpenAI · known binary').replace('OpenAI Moderation · known category decisions', 'OpenAI · known categories') for s in order]
    ax.set_yticks(y, labels); ax.invert_yaxis(); ax.set_xlim(0, 110); ax.set_xticks(range(0, 101, 20))
    ax.set_xlabel('Harmful-content F1 (%) · higher is better')
    ax.set_title('Chat moderation · detection across datasets', loc='left', weight='bold', pad=27)
    ax.legend(loc='lower center', bbox_to_anchor=(.5, 1), ncol=4, frameon=False)
    ax.grid(axis='x', alpha=.14); ax.set_axisbelow(True)
    save(fig, 'moderation-comparison')


def moderation_section(analysis, references):
    tasks = {t['slug']: t for t in analysis['tasks']}
    # The Jev extension is optional while a run is incomplete. Never fill missing scores.
    jev_path = ROOT / 'bench/results/moderation-study/jev_analysis.json'
    jev_data = json.loads(jev_path.read_text()) if jev_path.exists() else {}
    jev_tasks = {t['slug']: t for t in jev_data.get('tasks', [])}
    if jev_data:
        if not jev_data.get('complete') or jev_data.get('study_sha256') != analysis.get('study_sha256') or jev_tasks.keys() != tasks.keys():
            raise ValueError('Complete matching Jev analysis required for publication')
        result_dir = jev_path.parent
        for slug, task in jev_tasks.items():
            if not task.get('complete') or task['n'] != tasks[slug]['n']:
                raise ValueError('Incomplete Jev task: ' + slug)
            if hashlib.sha256((result_dir / (slug + '_jev.json')).read_bytes()).hexdigest() != task['input_sha256']:
                raise ValueError('Jev evidence changed: ' + slug)
            for target in ['base', 'fine1000', 'fine5000']:
                expected = tasks[slug]['runs'][target]['input_sha256']
                if task['comparisons']['jev_vs_' + target]['reference_sha256'] != expected or hashlib.sha256((result_dir / (slug + '_' + target + '.json')).read_bytes()).hexdigest() != expected:
                    raise ValueError('Laya comparison evidence changed: ' + slug)

    def jev_summary(slug):
        t = jev_tasks.get(slug, {})
        return t.get('summary') or t.get('runs', {}).get('jev', {}).get('summary')

    def jev_value(slug, key='harmful_f1'):
        s = jev_summary(slug)
        return pct(s[key]) if s is not None else 'Not measured yet'

    order = ['aegis_prompt', 'aegis_response', 'toxicchat', 'wildguard_prompt', 'wildguard_response', 'beavertails', 'openai_known', 'openai_categories']
    rows = [[TASK_NAMES[s], f"{tasks[s]['n']:,}", *[pct(tasks[s]['runs'][m]['summary']['harmful_f1']) for m in ['base', 'fine1000', 'fine5000']], jev_value(s)] for s in order]
    body = '''<section id="moderation"><h2>Chat moderation: stronger detection, more false alarms</h2>
<p>A moderation model must catch harmful material while letting ordinary conversation through. We trained two more Laya models on 1,000 and 5,000 labelled decisions from Aegis, then tested them on six safety datasets. These models use the datasets’ existing labels; no Jev answers were used for training.</p>
<p>The score here is <strong>harmful-content F1</strong>, which balances catching harmful examples against incorrectly flagging harmless ones. It is a different measure from the macro-F1 used in the five-task comparison above.</p>'''
    moderation_chart(tasks, jev_summary, order)
    body += figure('moderation-comparison', 'Harmful-content F1 for base Laya, two Aegis fine-tunes and measured Jev results across moderation datasets. Exact scores follow in the table.', 'Measured moderation results. Each model is tested on the same saved examples within a row. No vendor-advertised scores are mixed into this chart.')
    body += table(['Test', 'Decisions', 'Laya before', 'Laya · 1k', 'Laya · 5k', 'Jev · measured here'], rows)
    body += '<p class="note">OpenAI Moderation above uses 859 examples with known binary labels, not the full 1,680-example published benchmark. Missing category labels are unknown, not safe. The separate category evaluation covers 9,298 known category decisions. Aegis response and WildGuard tasks also omit missing or unusable labels; exact counts and rules are in the full moderation report.</p>'
    if jev_tasks:
        total = sum(t['n'] for t in jev_tasks.values())
        failures = sum(t['summary']['errors'] for t in jev_tasks.values())
        identity = ', '.join(jev_data['model_resolved'])
        body += f'<p class="note">Jev model: {html.escape(identity)}. The run covers {total:,} decisions with {failures} request {"failure" if failures == 1 else "failures"}. Failed requests remain in the evidence and error counts. All 67,890 Laya moderation decisions completed without request failures.</p>'
        ap = jev_summary('aegis_prompt')['harmful_f1']
        tc = jev_summary('toxicchat')['harmful_f1']
        wr = jev_summary('wildguard_response')['harmful_f1']
        body += f'<p><strong>The moderation comparison is less favourable to Laya than the workflow results.</strong> On the same test cases, Jev scores {pct(tc)} on ToxicChat and {pct(wr)} on WildGuard responses, compared with {pct(tasks["toxicchat"]["runs"]["fine5000"]["summary"]["harmful_f1"])} and {pct(tasks["wildguard_response"]["runs"]["fine5000"]["summary"]["harmful_f1"])} for the larger Laya fine-tune. Aegis prompt scores are closer: {pct(ap)} for Jev and {pct(tasks["aegis_prompt"]["runs"]["fine5000"]["summary"]["harmful_f1"])} for fine-tuned Laya. Learning the training domain does not guarantee equally strong results elsewhere.</p>'
        body += '<p>Jev’s moderation column comes from fresh hosted-model requests on the same frozen test cases and question wording as Laya. The saved model identities, errors and paired uncertainty estimates are available in the <a href="bench/results/moderation-study/jev_analysis.json">Jev moderation analysis</a>. Jev is evaluated as a classifier, not used as a source of training labels. These are secondary comparisons added after reviewing the Laya results; the descriptive 95% paired intervals are not corrected for testing multiple tasks. The Jev requests use four concurrent hosted calls; Laya runs locally one request at a time. Latency is therefore not a controlled speed comparison. Jev receives the full source text, with provider-side truncation unknown; Laya has a 2,048-token limit.</p>'
    else:
        body += '<p class="note">Jev moderation scores have not been added yet. The Jev figures elsewhere on this page measure other tasks and must not be treated as moderation results.</p>'
    body += '<h3>What happens to harmless requests?</h3><p>XSTest includes 250 harmless prompts that can look risky out of context. Fine-tuning raised Laya’s false-alarm rate from 24.8% to 65.2% with 1,000 training decisions and 47.2% with 5,000. Better detection alone does not establish a dependable general-purpose safety filter.</p>'
    false_rows = []
    for m, label in [('base', 'Laya before'), ('fine1000', 'Laya · 1k'), ('fine5000', 'Laya · 5k')]:
        h = tasks['xstest']['runs'][m]['summary']['harmful_class']
        false_rows.append([label, f"{h['fp']}/{h['negative_support']}", pct(h['false_positive_rate']), f"{h['tp']}/{h['support']}", pct(h['recall'])])
    js = jev_summary('xstest')
    if js:
        h = js['harmful_class']; false_rows.append(['Jev', f"{h['fp']}/{h['negative_support']}", pct(h['false_positive_rate']), f"{h['tp']}/{h['support']}", pct(h['recall'])])
        body += f'<p>Jev flags {h["fp"]} of the {h["negative_support"]} harmless prompts ({pct(h["false_positive_rate"])}) and catches {h["tp"]} of the {h["support"]} harmful prompts ({pct(h["recall"])}). Those two measures belong together: blocking less is only helpful if harmful requests are still caught.</p>'
    body += table(['Model', 'Harmless prompts flagged', 'False-alarm rate ↓', 'Harmful prompts caught', 'Harmful recall ↑'], false_rows)
    body += figure('false-alarm-tradeoff', 'False-alarm rates for Laya on harmless XSTest prompts rise after fine-tuning. Exact counts are in the preceding table.', 'The XSTest task judges prompts. Published XSTest response-refusal scores answer a different question.')
    body += '<h3>How do OpenAI and Mistral compare?</h3><p>The figures below are <strong>advertised results from the model publishers</strong>. We did not rerun these models. They provide context, but differences in policies, retained examples and scoring mean they are not a controlled leaderboard against our Laya and Jev measurements.</p>'
    card = references['sources']['mistral_card']
    ref_rows = [[r['dataset'] + (' · prompts' if r['task'] == 'prompt_harmfulness' else ' · responses'), f"{r['Shieldstral-3B']:.1f}%", f"{r['GPT-OSS-Safeguard-20B']:.1f}%"] for r in card['scores'] if r['dataset'] not in ['XSTest', 'XSTest Harm']]
    body += table(['Published benchmark', 'Mistral Shieldstral 3B', 'OpenAI safeguard 20B · evaluated by Mistral'], ref_rows)
    body += f'<p class="note">Source: <a href="{html.escape(card["url"])}">Mistral’s pinned model card</a>. F1 as reported; Shieldstral threshold 0.5 and safeguard at high reasoning effort. The OpenAI Moderation row uses a different selection from our known-label subset.</p>'
    oai = references['sources']['openai_report']
    body += table(['OpenAI’s own published evaluation', 'OpenAI Moderation', 'ToxicChat'], [[r['model'], f"{r['OpenAI Mod (2022)']:.1f}%", f"{r['ToxicChat']:.1f}%"] for r in oai['scores'] if r['model'].startswith('gpt-oss-safeguard')])
    body += f'<p class="note">Source: <a href="{html.escape(oai["url"])}">OpenAI technical report, Table 2</a>. These are OpenAI’s experiments, separate from Mistral’s safeguard column. Exact policy and missing-label treatment are not established as matching this study.</p>'
    body += '<p class="note">The Aegis budgets count labelled decisions, not distinct conversations; the 1,000-decision set is nested inside the 5,000-decision set. We used one training seed and the final checkpoint after three passes through each set. Training excludes exact normalized overlaps with the frozen tests. Eight long WildGuard responses are truncated; the detailed report includes a sensitivity check. These checks do not establish whether a base model saw a dataset during pretraining.</p>'
    body += '<p>Aegis fine-tuning improves detection within the training domain and on several other datasets. It also exposes a practical limit: broader blocking can help a detection score while making a chat product frustrating to use. The next useful experiment would measure whether targeted harmless examples and a development-set threshold can reduce false alarms without losing the gains.</p><p><a href="moderation.html">Read the full moderation study</a> · <a href="docs/reproduce.md">Repeat the moderation experiment</a> · <a href="bench/results/moderation-study/analysis.json">Inspect all Laya scores and uncertainty ranges</a></p></section>'
    return body


def build():
    original = (ROOT / 'report/jev-vs-laya.html').read_text()
    workflow = read('bench/results/workflow-suite/analysis.json')
    banking = read('bench/results/public_study_analysis.json')
    charts(workflow, banking)
    doc = original.replace('href="../', 'href="').replace("href='../", "href='")
    doc = doc.replace('updated 20 September 2026 · review draft', 'public benchmarks · fine-tuning · chat moderation')
    doc = doc.replace('using six public datasets and a separate set of curated workflow examples.', 'using six public workflow datasets, a separate curated pilot, and six moderation datasets. OpenAI and Mistral’s published safety-model scores provide additional context.')
    nav = '<nav aria-label="Report sections"><a href="#banking">Customer support</a><a href="#five-datasets">Five public tasks</a><a href="#moderation">Chat moderation</a><a href="#earlier">Earlier benchmarks</a><a href="#methods">Methods</a><a href="docs/reproduce-full.md">Reproduce the results</a></nav>'
    doc = doc.replace('</header>', nav + '</header>', 1)
    doc = doc.replace('<h2>Customer support:', '<h2 id="banking">Customer support:', 1)
    doc = doc.replace('<h3>How confident', figure('banking-comparison', 'Banking77 accuracy: Laya default 40.0 percent, wider answer options 51.3 percent, fine-tuned 79.4 percent, Jev 80.0 percent.', 'The fine-tuning comparison uses the wider-options base. All four conditions use the same held-out test set.') + '<h3>How confident', 1)
    doc = doc.replace('<h3>Safety check:', figure('workflow-comparison', 'Grouped macro-F1 bars compare base Laya, task-specific fine-tuned Laya and Jev across five public tasks. Exact values are in the preceding table.', 'Each fine-tuned bar represents a separate task-specific model. Macro-F1 gives each class equal weight.') + '<h3>Safety check:', 1)
    doc = doc.replace('<h2>What this means in practice</h2>', moderation_section(read('bench/results/moderation-study/analysis.json'), read('bench/moderation_references.json')) + '<h2>What this means in practice</h2>')
    doc = doc.replace('<h2>How this fits', '<h2 id="earlier">How this fits', 1).replace('<h2>Methods and supporting data</h2>', '<h2 id="methods">Methods and supporting data</h2>')
    doc = doc.replace('<a href="bench/results/continuation_comparison.txt">See the full workflow results.</a> This private dataset is not included in the public reproducibility package.', 'The private workflow dataset is not included in the public reproducibility package. Its figures remain a historical pilot, not a reproducible public benchmark.')
    doc = re.sub(r'<footer>.*?</footer>', '<footer>Independent research · Results apply to the models, datasets and settings described here. <a href="https://github.com/Alexander-Ollman/laya-ft">Code and evidence</a> · <a href="docs/reproduce-full.md">Full reproduction guide</a> · <a href="moderation.html">Standalone moderation report</a></footer>', doc, flags=re.S)
    css = 'figure{margin:32px 0;background:white;padding:18px;border:1px solid var(--line);border-radius:12px}.chart-scroll{overflow-x:auto}figure img{min-width:800px;display:block;width:100%;height:auto}figcaption{font-size:13px;color:var(--muted);margin-top:10px}nav{display:flex;flex-wrap:wrap;gap:10px 22px;margin-top:24px;font-size:14px}h2[id],section[id]{scroll-margin-top:20px}@media(max-width:600px){figure{padding:6px;margin:24px -8px}figcaption{padding:8px}}'
    doc = doc.replace('</style>', css + '</style>', 1)
    doc = re.sub(r'''href=(["'])(docs/[^"']+\.md)\1''', lambda m: 'href="https://github.com/Alexander-Ollman/laya-ft/blob/research/full-report/' + m[2] + '"', doc)
    (ROOT / 'index.html').write_text(doc)
    print('Built index.html and six chart assets from saved evidence.')


if __name__ == '__main__':
    build()
