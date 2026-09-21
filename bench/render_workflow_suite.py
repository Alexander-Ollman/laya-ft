"""Plain-English report section for the five-dataset extension."""
import json
from public_workflow import ROOT

TASKS={'injection':('Prompt-injection detection','deepset/prompt-injections'),
 'sms':('SMS spam filtering','ucirvine/sms_spam'),'emotion':('Emotion classification','dair-ai/emotion'),
 'counterfactual':('“What if” product feedback','SetFit/amazon_counterfactual'),
 'massive':('Assistant request routing','AmazonScience/massive')}


def pct(x):return '—' if x is None else f'{100*x:.1f}%'


def render():
    path=ROOT/'bench/results/workflow-suite/analysis.json'
    if not path.exists():return ''
    analysis=json.loads(path.read_text())
    tasks=analysis['tasks'];rows=[];safety=[];details=[];intervals=[];complete=[]
    for item in tasks:
        slug=item['slug'];title,repo=TASKS[slug];runs=item['runs']
        def summary(target):return runs.get(target,{}).get('summary',{})
        base,fine,jev=(summary(t) for t in ('laya:base','laya:fine','jev'))
        if base and fine and jev:complete.append(item)
        if item.get('basecomparison'):
            c=item['basecomparison'];lo,hi=c['macro_f1_interval']
            intervals.append(f'<tr><td>{title}</td><td>{100*c["macro_f1_delta_fine_minus_reference"]:+.1f} points</td><td>{100*lo:+.1f} to {100*hi:+.1f}</td></tr>')
        n=item.get('test',fine.get('n',base.get('n','—')))
        rows.append(f'<tr><td><a href="https://huggingface.co/datasets/{repo}">{title}</a></td><td>{n}</td>'
                    f'<td>{pct(base.get("macro_f1"))}</td><td>{pct(fine.get("macro_f1"))}</td><td>{pct(jev.get("macro_f1"))}</td></tr>')
        for target,label in [('laya:base','Laya before'),('laya:fine','Laya fine-tuned'),('jev','Jev')]:
            s=summary(target)
            if not s:continue
            latency='—' if s.get('p50_ms') is None else f'{s["p50_ms"]:.0f} ms'
            details.append(f'<tr><td>{title} · {label}</td><td>{pct(s["accuracy"])}</td><td>{pct(s["majority_baseline"])}</td>'
                           f'<td>{latency}</td><td>{s["errors"]}</td></tr>')
            if slug=='injection':
                p=s['safety_positive_class'];neg=s['n']-p['support']
                safety.append(f'<tr><td>{label}</td><td>{p["tp"]}/{p["support"]}</td><td>{p["fn"]}</td><td>{p["fp"]}/{neg}</td></tr>')
    progress=f'{len(complete)} of five comparisons are complete. Remaining results are still being measured.'
    if len(complete)==5:
        gains=sum(x['runs']['laya:fine']['summary']['macro_f1']>x['runs']['laya:base']['summary']['macro_f1'] for x in complete)
        progress=f'Fine-tuning improved Laya’s score on {gains} of the five tasks in this run. The individual results below show where it helped and where it did not.'
        clear=sum(x['basecomparison']['macro_f1_interval'][0]>0 for x in complete)
        progress+=f' The 99% uncertainty ranges remain above zero for {clear} of those improvements.'
    return '''<section id="five-datasets"><h2>Does the improvement hold across five more tasks?</h2>
<p>We extended the comparison to five public datasets: malicious prompt detection, SMS spam, emotional tone, product-review feedback, and assistant request routing. Each task gets its own Laya fine-tune, starting from the same base model. We use up to 1,000 training examples per task and keep test examples separate. Jev uses its hosted model without additional training or examples in the prompt.</p>
<p>'''+progress+'''</p>
<p>The main score below is <strong>macro-F1</strong>: it gives each answer category equal weight and accounts for both missed matches and wrong matches. Higher is better. This matters for tasks such as spam filtering, where simply choosing the most common answer can look deceptively accurate.</p>
<div class="scroll"><table><thead><tr><th>Task</th><th>Test examples</th><th>Laya before</th><th>Laya fine-tuned</th><th>Jev</th></tr></thead><tbody>'''+''.join(rows)+'''</tbody></table></div>
<p class="note">Training uses 1,000 examples per task, except injection detection with 446. Test sets are fixed samples of 1,000 where larger; injection uses all 116 official test examples and product feedback all 670. SMS uses a custom split because its source has no official test set. These are new, directly paired runs; scores from the earlier benchmark references are separate.</p>
<h3>Safety check: catching malicious prompts without blocking ordinary ones</h3>
<p>The prompt-injection task asks whether supplied text tries to override an assistant’s instructions or redirect its job. We count prompts labelled as attacks that were caught or missed, and benign prompts incorrectly flagged.</p>
<div class="scroll"><table><thead><tr><th>Model</th><th>Attacks caught</th><th>Attacks missed</th><th>False alarms on benign prompts</th></tr></thead><tbody>'''+''.join(safety)+'''</tbody></table></div>
<p class="note">This small dataset tests classification of labelled prompt text. It does not establish that an LLM protected by the classifier is safe from successful attacks. Some examples lack the full system and task context needed to settle borderline cases.</p>
<details><summary>Accuracy, response times, uncertainty and dataset notes</summary>
<div class="scroll"><table><thead><tr><th>Task</th><th>Fine-tuning gain in macro-F1</th><th>99% uncertainty range, points</th></tr></thead><tbody>'''+''.join(intervals)+'''</tbody></table></div>
<div class="scroll"><table><thead><tr><th>Task / model</th><th>Accuracy</th><th>Always choose most common</th><th>Typical response time</th><th>Failed requests</th></tr></thead><tbody>'''+''.join(details)+'''</tbody></table></div>
<p>We use a separate test for each task rather than blending everything into one accuracy score. Paired uncertainty intervals compare the models on the same examples. The five main fine-tune-versus-base comparisons use 99% intervals to allow for testing several tasks; other comparisons use descriptive 95% intervals. This is one training seed per task, so further runs could measure how sensitive the gains are to training randomness.</p>
<p>Duplicate examples, conflicting training labels, official test and validation texts, and previously benchmarked examples are excluded from training. Dataset familiarity during the original models’ pretraining remains unknown. Labels come from each source dataset; they are not all manually assigned.</p>
<p>A later check also looked for nearly repeated wording and templates that differ only in numbers. It flagged four SMS test examples, one product-review example and one assistant-routing example; none in injection or emotion. This is an additional sensitivity check, not a change to the original test. The <a href="../bench/results/workflow-suite/similarity_audit.json">audit records the examples and scores after excluding them</a>. It cannot detect every semantic overlap.</p>
<p>Emotion permits research and educational use. The original product-feedback dataset specifies non-commercial use. Those datasets and the resulting weights are not included in a public redistribution package by this experiment. Other dataset terms and exact source versions are recorded in the protocol.</p>
<p><a href="../docs/workflow-suite-protocol.md">Read the five-dataset protocol</a> · <a href="../bench/results/workflow-suite/analysis.json">Inspect every score and uncertainty interval</a></p></details></section>'''
