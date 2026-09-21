"""Build the independent HTML report from saved results, never invented scores."""
import html
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "bench" / "results"


def esc(value):
    return html.escape(str(value), quote=True)


def pct(value):
    return f"{100*value:.1f}%"


def public_results(analysis):
    found = {}
    for entry in analysis['runs']:
        if entry['split'] == 'heldout':
            path = Path(entry['input'])
            data = json.loads(path.read_text())
            data['summary'] = entry['summary']
            found[data["target"]] = (path, data)
    return found


def main():
    from analyze_public_study import analyze
    from public_workflow import DATA
    from render_workflow_suite import render as render_suite
    analysis = analyze(RESULTS, DATA, 'laya:laya-english-wide', 'laya:laya-banking77-gold-1k')
    (RESULTS/'public_study_analysis.json').write_text(json.dumps(analysis, indent=2)+'\n')
    found = public_results(analysis)
    targets = [("laya:laya-english", "Laya · default settings"),
               ("laya:laya-english-wide", "Laya · before fine-tuning"),
               ("laya:laya-banking77-gold-1k", "Laya · after fine-tuning"),
               ("jev", "Jev")]
    names = dict(targets)
    overview = []
    table = []
    for key, label in targets:
        if key not in found:
            table.append(f"<tr><td>{esc(label)}</td><td colspan='6'>Pending measurement</td></tr>")
            continue
        path, data = found[key]
        s = data["summary"]
        latency = f"{s['p50_ms']:.0f}" if s['p50_ms'] is not None else '—'
        if key != "laya:laya-english":
            overview.append(f"<tr><td>{esc(label)}</td><td>{pct(s['accuracy'])}</td><td>{latency} ms</td></tr>")
        tail_latency = f"{s['p95_ms']:.0f}" if s['p95_ms'] is not None else '—'
        table.append(f"<tr><td>{esc(label)}</td><td>{pct(s['accuracy'])}</td><td>{pct(s['macro_f1'])}</td>"
                     f"<td>{s['correct']:,}/{s['n']:,}</td><td>{latency}</td><td>{tail_latency}</td><td>{s['errors']}</td></tr>")
    pair = analysis.get('paired_comparison')
    uncertainty = ('Paired uncertainty is pending both completed Laya evaluations.' if not pair else
                   f"Fine-tune minus base: {100*pair['accuracy_delta_fine_minus_base']:+.1f} percentage points; "
                   f"95% grouped paired bootstrap interval {100*pair['percentile_95'][0]:+.1f} to {100*pair['percentile_95'][1]:+.1f} points. "
                   f"5,000 resamples over {pair['n_text_groups']:,} normalized-text groups, seed 7. "
                   "This measures test-sample uncertainty conditional on the two checkpoints, not training-seed variability.")
    jev_pair = analysis.get('jev_comparison')
    if jev_pair:
        uncertainty += (f" Secondary comparison, fine-tuned Laya minus Jev: {100*jev_pair['accuracy_delta_fine_minus_base']:+.1f} points "
                        f"(95% interval {100*jev_pair['percentile_95'][0]:+.1f} to {100*jev_pair['percentile_95'][1]:+.1f}).")
        if jev_pair['percentile_95'][0] <= 0 <= jev_pair['percentile_95'][1]:
            uncertainty += " That interval includes zero; this run establishes neither superiority nor equivalence between them."
    calibration = []
    for item in analysis['calibration']:
        before, after = item['heldout_before'], item['heldout_after']
        calibration.append(f"<tr><td>{esc(names.get(item['target'], item['target']))}</td><td>{item['temperature']:.3f}</td>"
                           f"<td>{before['nll']:.3f} → {after['nll']:.3f}</td>"
                           f"<td>{before['ece_10_equal_width']:.3f} → {after['ece_10_equal_width']:.3f}</td></tr>")
    controls = {}
    for path in sorted(RESULTS.glob('2*_laya_*.json')):
        data = json.loads(path.read_text())
        if Path(data.get('cases_file', '')).name == 'public_transfer_controls.jsonl':
            controls[data['target']] = data['rows']
    control_rows = []
    for target, rows in controls.items():
        values = []
        for dataset in ('agnews', 'emotiondair'):
            selected = [r for r in rows if r['category'] == dataset]
            values.append(f"{sum(r['pass'] for r in selected)}/{len(selected)}")
        control_rows.append(f"<tr><td>{esc(names.get(target, target))}</td><td>{values[0]}</td><td>{values[1]}</td></tr>")
    files = "".join(f"<li><a href='../bench/results/{esc(p.name)}'>{esc(names.get(d['target'], d['target']))}: saved predictions and metrics</a></li>"
                    for p, d in found.values()) or "<li>Run outputs will appear here when evaluation completes.</li>"
    metrics = []
    for path, data in found.values():
        s = data['summary']
        cost = (f"${s['cost_usd']:.4f}" if s.get('cost_usd') is not None else
                "No API fee" if data['target'].startswith('laya:') else "Not reported")
        metrics.append(f"<tr><td>{esc(names.get(data['target'], data['target']))}</td><td>{s['multiclass_brier']:.4f}</td><td>{s['nll']:.4f}</td>"
                       f"<td>{s['ece_10_equal_width']:.4f}</td><td>{s['probability_rows']:,}</td><td>{cost}</td></tr>"
                       if s['probability_rows'] else f"<tr><td>{esc(names.get(data['target'], data['target']))}</td><td colspan='5'>No valid full probability distributions</td></tr>")
    pilot = []
    pilot_scores = []
    for key, label in [('laya_laya-english','Before fine-tuning'), ('laya_laya-era-distill-1k','Fine-tuned on 1,007 requests'),
                       ('laya_laya-era-distill-full','Fine-tuned on 2,646 requests')]:
        paths = [p for p in sorted(RESULTS.glob('2*_' + key + '.json'))
                 if Path(json.loads(p.read_text()).get('cases_file', '')).name == 'cases.jsonl']
        if paths:
            d = json.loads(paths[-1].read_text())
            rows = d.get('rows', [])
            qs = [q for r in rows for q in r['questions'].values()]
            pilot_scores.append(sum(q['correct'] for q in qs)/len(qs))
            pilot.append(f"<tr><td>{label}</td><td>{pct(pilot_scores[-1])}</td><td>{sum(q['correct'] for q in qs)}/{len(qs)}</td>"
                         f"<td>{sum(r['pass'] for r in rows)}/{len(rows)}</td></tr>")
        else:
            pilot_scores.append(None)
            pilot.append(f"<tr><td>{label}</td><td colspan='3'>Training / evaluation pending</td></tr>")
    page = """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Jev vs. fine-tuned Laya — a practical comparison</title>
<style>
:root{color-scheme:light;--ink:#17252f;--muted:#566572;--line:#d6dedf;--blue:#17666c;--paper:#f4f5f0}
*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font:17px/1.65 system-ui,-apple-system,sans-serif}
main{max-width:1050px;margin:auto;padding:60px 32px 100px}header{border-top:6px solid var(--blue);padding-top:24px;margin-bottom:46px}
.eyebrow{font-size:12px;letter-spacing:.13em;text-transform:uppercase;color:var(--muted)}h1{font-size:clamp(34px,5vw,58px);line-height:1.08;letter-spacing:-.04em;max-width:850px;margin:20px 0}
h2{font-size:27px;line-height:1.2;margin:48px 0 18px}h3{font-size:19px}p{max-width:78ch}a{color:var(--blue);text-underline-offset:3px}
.lead{font-size:21px;max-width:75ch}.note{color:var(--muted);font-size:14px}.callout{border-left:4px solid var(--blue);padding:10px 22px;background:#e8efea}
.scroll{overflow-x:auto}table{width:100%;border-collapse:collapse;font-size:14px;font-variant-numeric:tabular-nums}th,td{text-align:left;padding:13px 12px;border-bottom:1px solid var(--line);vertical-align:top}th{font-size:11px;text-transform:uppercase;letter-spacing:.05em}td:not(:first-child){white-space:nowrap}
code{font-size:.85em;background:#e8ebe5;padding:2px 5px;overflow-wrap:anywhere}details{margin-top:40px;border-top:1px solid var(--line);padding-top:20px}summary{cursor:pointer;font-weight:650}footer{border-top:1px solid var(--line);margin-top:50px;padding-top:20px;color:var(--muted);font-size:13px}
@media(max-width:600px){main{padding:30px 18px 60px}body{font-size:16px}.lead{font-size:19px}th,td{padding:10px 8px}}
@media print{body{background:white}main{padding:0}details{display:block}.scroll{overflow:visible}a{color:inherit}}
</style></head><body><main>
<header><div class="eyebrow">Independent research · updated 20 September 2026 · review draft</div>
<h1>Jev vs. fine-tuned Laya</h1>
<p class="lead">How much can a small amount of good training data improve a local AI model?</p>
<p>We compare Jev with Laya before and after fine-tuning, using six public datasets and a separate set of curated workflow examples.</p></header>
<section class="callout"><p><strong>Fine-tuning brought Laya close to Jev on customer-support routing.</strong> With 1,001 labelled examples, Laya improved from 51.3% to 79.4% accuracy. Jev scored 80.0% on the same test. That small gap is not enough to call a winner.</p></section>
<p>We then tried five more public tasks. Fine-tuning improved Laya on all five. It scored higher than Jev on prompt-injection detection, SMS spam, emotion and product-review feedback; Jev stayed ahead on assistant request routing. Each task used its own fine-tuned model. <a href="#five-datasets">See the five-task comparison.</a></p>
<h2>Customer support: picking the right category</h2>
<p>Imagine a banking assistant deciding where to send a customer's message. Is it about a lost card, a cash withdrawal, or a bank transfer? The model must choose the right category from 77 possibilities.</p>
<p>We used <a href="https://huggingface.co/datasets/PolyAI/banking77">Banking77</a>, a public dataset with human-labelled customer questions. Laya trained on just 13 examples per category: 1,001 in total. We then tested both models on all 3,080 questions in the dataset's separate test set.</p>
<div class="scroll"><table><thead><tr><th>Model</th><th>Correct answers</th><th>Typical response time</th></tr></thead><tbody>OVERVIEW</tbody></table></div>
<p>Laya gained <strong>28.2 percentage points</strong> from fine-tuning. Both Laya versions had enough room to read the full list of categories, so the comparison measures the benefit of training under the same settings.</p>
<p>The fine-tuned Laya model answered in about 70 milliseconds on a local Apple M4 Max. Jev took about 187 milliseconds through its hosted API. Those timings include different local and network overheads, so they describe this setup rather than a universal speed advantage.</p>
<p class="note">Laya's default settings scored 40.0%. Increasing the space available for the category list raised that to 51.3% before training. We use that stronger starting point throughout the main comparison.</p>
<h3>How confident can we be in the result?</h3>
<p>The improvement over base Laya is clear in this test: our estimated range is about 26 to 30 percentage points. The gap between fine-tuned Laya and Jev is much smaller. The data are consistent with Laya being about two points behind or one point ahead. More tests would be needed to establish a reliable difference.</p>
<h3>Did learning banking make Laya worse at other tasks?</h3>
<p>We also checked 100 news headlines and 100 examples of emotional language. Neither set was used for this fine-tune. News accuracy fell by one answer, while emotion accuracy rose by two. These small checks show little change, but cannot establish how the model will perform on every other task.</p>
<div class="scroll"><table><thead><tr><th>Model</th><th>News: correct out of 100</th><th>Emotion: correct out of 100</th></tr></thead><tbody>CONTROLS</tbody></table></div>
<h2>A second test: everyday AI workflow decisions</h2>
<p>We also fine-tuned Laya on curated workflow datasets covering request routing, tool selection and safety checks. This was a separate model and experiment from the public banking test.</p>
<p>On a fixed set of 145 decision questions, accuracy rose from <strong>71.7% before training to 82.8% with the smaller training set, then 91.7% with the larger set</strong>. Jev previously scored 97.2% on those questions.</p>
<div class="scroll"><table><thead><tr><th>Laya training</th><th>Accuracy</th><th>Correct answers</th><th>Cases with every answer correct</th></tr></thead><tbody>PILOT</tbody></table></div>
<p>This is encouraging evidence that useful examples can improve Laya across several workflow decisions. The workflow test had already informed earlier question wording, however, so these results are exploratory. The separate public banking experiment provides the cleaner comparison.</p>
<details><summary>About the workflow data and its limits</summary>
<p>The training pool combines existing workflow datasets with additional generated examples. A local Qwen model supplied the training labels. The public Banking77 experiment uses the dataset's original human labels. These are different sources of supervision, and neither Laya model was trained on Jev's answers.</p>
<p>There were 2,646 unique requests in the larger workflow pool. Thirteen unusable answer sets were skipped. The older training setup separated individual questions for development checks; questions from the same request could appear in both training and development. Its development score therefore should not be treated as an independent test.</p>
<p>Tool selection and request routing improved, but fine-tuning did not fix the model's tendency to lose information at the end of long inputs. Safety results were mixed: moderation improved, while the injection check remained one answer below the original model.</p>
<p><a href="../bench/results/continuation_comparison.txt">See the full workflow results.</a> This private dataset is not included in the public reproducibility package.</p></details>
WORKFLOW_SUITE
<h2>What this means in practice</h2>
<p>For a well-defined task with a fixed set of possible answers, a modest amount of relevant training data can make a substantial difference. The banking experiment and all five additional tasks support that conclusion. Fine-tuned Laya also scored above Jev on four of the five additional tasks, while Jev retained the lead on assistant routing.</p>
<p>The choice still depends on the workload. Fine-tuning takes labelled examples, training time and a way to run the resulting model. Jev provides a hosted option. Accuracy, response time, running costs and the consequences of a wrong answer all matter.</p>
<p>These results come from one training run per experiment. They do not establish that either model will perform equally well on new topics, longer conversations or production traffic. We kept the public test questions out of fine-tuning, although we cannot know whether either original model encountered them during its earlier training.</p>
<h2>How this fits with existing benchmarks</h2>
<p>The <a href="https://github.com/kraayenjon/awesome-jev#benchmarks-and-evaluations">awesome-jev collection</a> points to independent tests of routing, spam detection, information extraction and other AI workflows. Earlier local runs against several of these benchmarks helped identify where Laya needed improvement.</p>
<details><summary>Earlier benchmark results and source links</summary>
<p>The Jev figures below come from the benchmark authors' published results. The Laya figures are our earlier local runs of the base model. They use different tasks and test sizes from the new fine-tuning study, so they should be read separately.</p>
<div class="scroll"><table><thead><tr><th>Protocol / task</th><th>Items</th><th>Jev, published</th><th>Laya, local</th></tr></thead><tbody>
<tr><td><a href="https://github.com/AbdelStark/jev-benchmarks">Classification pilot</a> · AG News, 4 labels</td><td>100</td><td>91.0%</td><td>97.0%</td></tr>
<tr><td>Same pilot · emotion, 6 labels</td><td>100</td><td>48.0%</td><td>40.0%</td></tr>
<tr><td>Same pilot · banking, 72 labels</td><td>100</td><td>87.0%</td><td>2.0% stock / 16.0% wider</td></tr>
<tr><td><a href="https://github.com/bitnovus/jev-spam-eval">Email spam</a> · original seed-42 binary protocol</td><td>500</td><td>98.0%</td><td>97.2%</td></tr>
<tr><td><a href="https://github.com/nibzard/decision-model-benchmark">Decision Model Benchmark</a> · SMS spam</td><td>300</td><td>93.0%</td><td>86.7%</td></tr>
<tr><td>Same benchmark · 77-way banking intent</td><td>300</td><td>76.3%</td><td>36.0%</td></tr>
<tr><td><a href="https://github.com/TokenTrim/jev-agent-failure-benchmark">Agent failure attribution</a> · who / when / what</td><td>300</td><td>90.7 / 80.0 / 34.3%</td><td>64.3 / 29.3 / 10.3%</td></tr>
<tr><td><a href="https://github.com/vclic/smoking-extraction-benchmark">Smoking-history extraction</a> · all 10 fields correct</td><td>1,000</td><td>92.4%</td><td>25.4%</td></tr>
</tbody></table></div>

<p class="note">The classification test used the same saved list of examples across models. The earlier run records report matching examples for email, agent failure and smoking-history extraction. Decision Model Benchmark used three repeats for the author's Jev results and one local Laya repeat. Some source projects have since expanded their tests. For SMS spam, always choosing the most common answer would score 87.7%, above this Laya result.</p>
<p><a href="https://github.com/jourdanlabs/assay-001">ASSAY-001</a> studies routing accuracy and confidence using saved inputs and outputs. <a href="https://github.com/anessbelbati/jev-rerank-bench">Jev Rerank Bench</a> tests which retrieved documents are most relevant. <a href="https://github.com/Gaurav-Gosain/jev-sec-bench">Jev Security Bench</a> examines prompt injection and vulnerable code. These are useful references; we have not run new fine-tuned Laya comparisons on them here.</p></details>
<h2>Methods and supporting data</h2>
<p>The sections below contain the details needed to check the results or repeat the public experiment.</p>
<details><summary>Training setup and complete scores</summary>
<p>We selected 1,001 training questions and 770 separate development questions from Banking77's original training set. We removed overlap with the official test set, duplicate training questions and conflicting labels. All 3,080 official test questions remain in the final evaluation.</p>
<p>The public fine-tune started from base Laya and made three passes through the training set. We used the final saved model, without choosing a version based on test results. Both the base and fine-tuned comparisons allowed 1,024 tokens in total and 768 for the answer options. A token is a small unit of text used by the model.</p>
<div class="scroll"><table><thead><tr><th>Model</th><th>Accuracy</th><th>Macro-F1</th><th>Correct</th><th>Median ms</th><th>95th percentile ms</th><th>Failed requests</th></tr></thead><tbody>PRIMARY</tbody></table></div>
<p class="note">Macro-F1 gives each category equal weight when combining precision and recall. The 95th-percentile time is the response time below which 95% of requests finished. All requests ran one at a time.</p>
<p class="note">UNCERTAINTY</p>
<p class="note">The uncertainty calculation resamples the test examples 5,000 times, keeping duplicate texts together. It estimates variation from the test sample, not variation across repeated training runs. Dataset files, model versions and settings are recorded in the linked evidence.</p></details>
<details><summary>Confidence scores and cost</summary>
<p>A model can be confidently wrong. We therefore checked whether its predicted probabilities matched how often its answers were correct. Fine-tuning improved Laya's confidence estimates substantially on the banking test.</p>
<div class="scroll"><table><thead><tr><th>Model</th><th>Brier ↓</th><th>NLL ↓</th><th>ECE ↓</th><th>Answers scored</th><th>API cost for this test</th></tr></thead><tbody>METRICS</tbody></table></div>
<p class="note">Lower is better for all three measures. Brier measures probability errors across the categories. NLL penalizes assigning very little probability to the right answer. ECE compares stated probability with observed accuracy. We use the probability of the chosen answer, not the model's separate confidence field. Small rounding differences are normalized; unusable probability maps are excluded from these measures. Failed requests still count against accuracy.</p>
<p>Local Laya has no per-request API fee, but hardware, electricity and training still cost money. Jev's reported API charge for these 3,080 questions was about $0.13.</p>
<h3>Adjusting confidence using a separate development set</h3>
<p>We also tested a simple confidence adjustment, known as temperature scaling. We chose the adjustment using only the 770 development questions, then applied it to the test results. It changes the probabilities, not the answers.</p>
<div class="scroll"><table><thead><tr><th>Model</th><th>Adjustment</th><th>NLL before → after</th><th>ECE before → after</th></tr></thead><tbody>CALIBRATION</tbody></table></div>
<p class="note">The adjustment helped base Laya substantially. For fine-tuned Laya, NLL improved slightly while ECE became slightly worse. These calculations use rounded saved probabilities and approximate the adjustment available from full model scores. They do not change the served models.</p></details>
<details><summary>Download the evidence and repeat the experiment</summary><ul>FILES<li><a href="../docs/public-benchmark-protocol.md">Full method and reproduction commands</a></li><li><a href="../train/public-banking77/manifest.json">Dataset sources and exact split records</a></li><li><a href="../bench/results/public_study_analysis.json">Scores, uncertainty ranges and confidence analysis</a></li><li><a href="../bench/results/public_runtime.json">Hardware and software versions</a></li><li><a href="../bench/results/public_jev_provenance.json">Verification of the Jev test requests</a></li></ul>
<p class="note">Banking77: Casanueva et al., <a href="https://arxiv.org/abs/2003.04807">Efficient Intent Detection with Dual Sentence Encoders</a> (2020), PolyAI, CC BY 4.0.</p></details>
<footer>Draft for review · Results apply to the models, datasets and settings described here. This report has not been publicly released.</footer>
</main></body></html>"""
    for key, value in {"WORKFLOW_SUITE": render_suite(), "OVERVIEW": ''.join(overview), "PRIMARY": ''.join(table),
                       "METRICS": ''.join(metrics) or "<tr><td colspan='6'>Pending measurement</td></tr>",
                       "CALIBRATION": ''.join(calibration) or "<tr><td colspan='4'>Pending development evaluation</td></tr>",
                       "CONTROLS": ''.join(control_rows) or "<tr><td colspan='3'>Pending control evaluation</td></tr>",
                       "UNCERTAINTY": esc(uncertainty),
                       "PILOT": ''.join(pilot), "FILES": files}.items():
        page = page.replace(key, value)
    path = ROOT / "report" / "jev-vs-laya.html"
    archive = ROOT / "report" / "archive" / "jev-vs-laya-v2.html"
    if path.exists() and not archive.exists():
        archive.parent.mkdir(exist_ok=True)
        shutil.copy2(path, archive)
    path.write_text(page)
    print(path)


if __name__ == '__main__':
    main()
