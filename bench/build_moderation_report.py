"""Build a local, plain-English moderation report from validated measurements."""
import html
import json

from public_workflow import ROOT

RESULTS = ROOT / "bench/results/moderation-study"
TARGETS = ("base", "fine1000", "fine5000")
NAMES = {"base": "Laya before", "fine1000": "Laya · 1,000 decisions", "fine5000": "Laya · 5,000 decisions"}
TASKS = {
    "aegis_prompt": ("Aegis · user input", "Same source as training"),
    "aegis_response": ("Aegis · assistant response", "Same source as training"),
    "toxicchat": ("ToxicChat · user input", "Transfer"),
    "wildguard_prompt": ("WildGuard · user input", "Transfer"),
    "wildguard_response": ("WildGuard · assistant response", "Transfer"),
    "beavertails": ("BeaverTails · assistant response", "Transfer"),
    "openai_known": ("OpenAI · known-label subset", "Transfer; adapted subset"),
    "openai_categories": ("OpenAI · supplied category labels", "Separate category questions"),
    "xstest": ("XSTest · benign and harmful prompts", "False-alarm check"),
}
CSS = """
:root{color-scheme:light;--ink:#17252f;--muted:#566572;--line:#d6dedf;--blue:#17666c;--paper:#f4f5f0}
*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font:17px/1.65 system-ui,-apple-system,sans-serif}
main{max-width:1050px;margin:auto;padding:60px 32px 100px}header{border-top:6px solid var(--blue);padding-top:24px;margin-bottom:46px}
.eyebrow{font-size:12px;letter-spacing:.13em;text-transform:uppercase;color:var(--muted)}h1{font-size:clamp(34px,5vw,58px);line-height:1.08;letter-spacing:-.04em;max-width:850px;margin:20px 0}
h2{font-size:27px;line-height:1.2;margin:48px 0 18px}h3{font-size:19px}p{max-width:78ch}a{color:var(--blue);text-underline-offset:3px}
.lead{font-size:21px;max-width:75ch}.note{color:var(--muted);font-size:14px}.callout{border-left:4px solid var(--blue);padding:10px 22px;background:#e8efea}
.scroll{overflow-x:auto}table{width:100%;border-collapse:collapse;font-size:14px;font-variant-numeric:tabular-nums}th,td{text-align:left;padding:13px 12px;border-bottom:1px solid var(--line);vertical-align:top}th{font-size:11px;text-transform:uppercase;letter-spacing:.05em}td:not(:first-child){white-space:nowrap}
details{margin-top:32px;border-top:1px solid var(--line);padding-top:20px}summary{cursor:pointer;font-weight:650}footer{border-top:1px solid var(--line);margin-top:50px;padding-top:20px;color:var(--muted);font-size:13px}
@media(max-width:600px){main{padding:30px 18px 60px}body{font-size:16px}.lead{font-size:19px}th,td{padding:10px 8px}}
@media print{body{background:white}main{padding:0}.scroll{overflow:visible}a{color:inherit}}
"""


def esc(value):
    return html.escape(str(value), quote=True)


def pct(value):
    return "Pending" if value is None else f"{100 * value:.1f}%"


def interval(values):
    return "—" if not values else f"{100 * values[0]:.1f}–{100 * values[1]:.1f}%"


def table(headers, rows):
    return '<div class="scroll"><table><thead><tr>' + ''.join('<th>' + esc(h) + '</th>' for h in headers) + \
        '</tr></thead><tbody>' + ''.join('<tr>' + ''.join('<td>' + str(cell) + '</td>' for cell in row) + '</tr>' for row in rows) + '</tbody></table></div>'


def summary(task, target):
    return task.get("runs", {}).get(target, {}).get("summary", {})


def build_page(analysis, study, references, audit):
    tasks = {t["slug"]: t for t in analysis.get("tasks", [])}
    counts = study.get("test_counts", {})
    complete = sum(target in task.get("runs", {}) for task in tasks.values() for target in TARGETS)
    total = len(counts) * len(TARGETS)
    progress = f"{complete} of {total} dataset/model evaluations are complete." if total else "Measurements are pending."
    conclusions = []
    primary = []
    for slug in ("aegis_prompt", "aegis_response"):
        for fine in ("fine1000", "fine5000"):
            comparison = tasks.get(slug, {}).get("comparisons", {}).get(fine + "_vs_base")
            if comparison:
                primary.append(comparison)
    if len(primary) == 4:
        improved = sum(c["harmful_f1_delta_fine_minus_reference"] > 0 for c in primary)
        lower = sum(c["harmful_f1_delta_fine_minus_reference"] < 0 for c in primary)
        clear = sum(c["harmful_f1_interval"][0] > 0 for c in primary)
        if improved == clear == 4:
            conclusions.append("Both fine-tuned models improved on the base model for Aegis user inputs and assistant responses. All four improvements are supported by the 98.75% uncertainty ranges. Aegis is also the source of the training data; these tests use held-out examples.")
        else:
            conclusions.append(f"On Aegis, fine-tuning raised harmful-content F1 in {improved} of four input/output comparisons and lowered it in {lower}. The 98.75% uncertainty ranges support an improvement in {clear} comparisons.")
    transfer = [tasks.get(slug, {}) for slug in ("toxicchat", "wildguard_prompt", "wildguard_response", "beavertails", "openai_known")]
    if all(summary(t, "base") and summary(t, "fine5000") for t in transfer):
        gains = sum(summary(t, "fine5000")["harmful_f1"] > summary(t, "base")["harmful_f1"] for t in transfer)
        conclusions.append(f"The 5,000-decision model had higher harmful-content F1 than the base model on {gains} of the five transfer tasks. That score alone does not establish reliable moderation: catching more harmful content can come with more false alarms on benign requests.")
    budget_dips = []
    for slug in ("toxicchat", "wildguard_prompt", "wildguard_response", "beavertails", "openai_known"):
        c = tasks.get(slug, {}).get("comparisons", {}).get("fine5000_vs_fine1000")
        if c and c["harmful_f1_delta_fine_minus_reference"] < 0:
            uncertainty = c["harmful_f1_interval"]
            if uncertainty[0] <= 0 <= uncertainty[1]:
                budget_dips.append(TASKS[slug][0].replace(" · ", " "))
    if budget_dips:
        conclusions.append("More training data did not improve every score. The 5,000-decision model scored slightly below the 1,000-decision model on " + " and ".join(budget_dips) + ". The 95% uncertainty ranges include no difference, so these results do not establish a decline.")
    xbase = summary(tasks.get("xstest", {}), "base")
    if xbase:
        bp = xbase["harmful_class"]
        base_benign = xbase["n"] - bp["support"]
        observations = []
        increased = False
        for target, size in (("fine1000", "1,000"), ("fine5000", "5,000")):
            xs = summary(tasks.get("xstest", {}), target)
            if not xs:
                continue
            xp = xs["harmful_class"]
            benign = xs["n"] - xp["support"]
            increased = increased or xp["fp"] > bp["fp"]
            observations.append(f"After {size} training decisions, it flagged {xp['fp']}/{benign} harmless prompts ({pct(xp['false_positive_rate'])}) and caught {xp['tp']}/{xp['support']} harmful prompts ({pct(xp['recall'])}).")
        if observations:
            lead = "Better detection came with more false alarms." if increased else "Detection and false alarms both matter."
            conclusions.insert(0, lead + f" On XSTest, the base model flagged {bp['fp']}/{base_benign} harmless prompts ({pct(bp['false_positive_rate'])}) and caught {bp['tp']}/{bp['support']} harmful prompts ({pct(bp['recall'])}). " + " ".join(observations) + " These false alarms would affect people asking harmless questions; the results do not establish readiness for unattended moderation.")
    if not conclusions:
        conclusions.append("The comparison is still running. Pending cells contain no estimated scores, and conclusions will update when the validated measurements are available.")
    if analysis.get("skipped"):
        conclusions.append(f"{len(analysis['skipped'])} result files did not pass evidence validation and are excluded; see the analysis record for details.")
    main_rows = []
    for slug, (title, scope) in TASKS.items():
        if slug in ("openai_categories", "xstest"):
            continue
        task = tasks.get(slug, {})
        main_rows.append([esc(title) + '<br><span class="note">' + esc(scope) + '</span>', counts.get(slug, "Pending"),
                          *[pct(summary(task, target).get("harmful_f1")) for target in TARGETS]])
    xrows = []
    xtask = tasks.get("xstest", {})
    for target in TARGETS:
        s = summary(xtask, target)
        if not s:
            xrows.append([esc(NAMES[target]), "Pending", "Pending", "Pending", "Pending"])
            continue
        p = s["harmful_class"]
        benign = s["n"] - p["support"]
        xrows.append([esc(NAMES[target]), f'{p["fp"]}/{benign} ({pct(p["false_positive_rate"])})',
                      interval(p["false_positive_wilson_95"]), f'{p["tp"]}/{p["support"]}', p["negative_errors"]])
    safety_rows, timing_rows, comparisons, category_rows, sensitivity = [], [], [], [], []
    for slug, (title, _) in TASKS.items():
        task = tasks.get(slug, {})
        for target in TARGETS:
            run = task.get("runs", {}).get(target)
            if not run:
                continue
            s = run["summary"]
            p = s["harmful_class"]
            label = esc(title + " · " + NAMES[target])
            safety_rows.append([label, pct(p["precision"]) if p["precision"] is not None else "Undefined",
                                f'{p["tp"]}/{p["support"]} ({pct(p["recall"])})',
                                f'{p["fp"]}/{s["n"]-p["support"]} ({pct(p["false_positive_rate"])})', s["errors"]])
            timing_rows.append([label, pct(s["accuracy"]), '—' if s.get("p50_ms") is None else f'{s["p50_ms"]:.0f} ms',
                                '—' if s.get("p95_ms") is None else f'{s["p95_ms"]:.0f} ms',
                                pct(s["average_precision"]["value"]) if s["average_precision"]["value"] is not None else "Undefined"])
            if slug == "openai_categories":
                for category, cs in run.get("policy_categories", {}).items():
                    cp = cs["harmful_class"]
                    category_rows.append([esc(category), esc(NAMES[target]), cs["n"], pct(cs["harmful_f1"]),
                                          pct(cp["recall"]), pct(cp["false_positive_rate"])])
            u = run.get("untruncated_sensitivity", {})
            if u.get("n_excluded"):
                sensitivity.append([label, u["n_excluded"], u["n_retained"], pct(s["harmful_f1"]),
                                    pct((u.get("summary") or {}).get("harmful_f1"))])
        for c in task.get("comparisons", {}).values():
            low, high = c["harmful_f1_interval"]
            comparisons.append([esc(title), esc(NAMES[c["fine"]] + " vs " + NAMES[c["reference"]]),
                                f'{100*c["harmful_f1_delta_fine_minus_reference"]:+.1f}',
                                f'{100*low:+.1f} to {100*high:+.1f}', f'{100*c["confidence"]:g}%'])
    mref = references.get("sources", {}).get("mistral_card", {})
    reference_rows = [[esc(r["dataset"]), "User input" if r["task"] == "prompt_harmfulness" else "Assistant response",
                       f'{r["Shieldstral-3B"]:.1f}%', f'{r["GPT-OSS-Safeguard-20B"]:.1f}%']
                      for r in mref.get("scores", []) if r["dataset"] not in ("XSTest", "XSTest Harm")]
    oref = references.get("sources", {}).get("openai_report", {})
    openai_rows = [[esc(r["model"]), f'{r["OpenAI Mod (2022)"]:.1f}%', f'{r["ToxicChat"]:.1f}%']
                   for r in oref.get("scores", []) if r["model"].startswith("gpt-oss-safeguard")]
    training_rows = []
    for size in (1000, 5000):
        record = audit.get("sets", {}).get(f"aegis_{size}/train", {})
        provenance = record.get("label_source_counts", {})
        training_rows.append([f"{size:,}", record.get("unique_prompt_groups", "Not audited"),
                              provenance.get("human", "—"), provenance.get("llm_jury", "—"), provenance.get("refusal_data_augmentation", "—")])
    oai_rows = []
    for target in TARGETS:
        s = summary(tasks.get("openai_categories", {}), target)
        oai_rows.append([esc(NAMES[target]), s.get("n", counts.get("openai_categories", "Pending")), pct(s.get("harmful_f1"))])
    return f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Can fine-tuned Laya moderate chat?</title><style>{CSS}</style></head><body><main>
<header><div class="eyebrow">Independent research · 21 September 2026 · local review draft</div>
<h1>Can fine-tuned Laya moderate chat?</h1>
<p class="lead">We compare two sizes of safety training data for a small local model, then measure what it learns and whether that learning carries over to other datasets.</p>
<p class="note">{esc(progress)} This report has not been publicly released. Published competitor scores are references, not new model runs.</p></header>
<div class="callout">{''.join('<p>'+esc(c)+'</p>' for c in conclusions)}</div>
<h2>What changed with fine-tuning?</h2>
<p>Each fine-tune starts from the same Laya base. One uses 1,000 labelled decisions; the other uses 5,000, including the smaller set. Half the decisions judge user input and half judge an assistant response in context. Test examples stay out of training.</p>
<p>The score below is <strong>harmful-content F1</strong>. It balances catching harmful content with avoiding false alarms. Higher is better. It is not accuracy, and it is not the macro-F1 used in our earlier workflow study.</p>
{table(['Test','Examples','Laya before','After 1,000','After 5,000'], main_rows)}
<p class="note">“Same source as training” uses separate Aegis test data. “Transfer” uses a different dataset. Input and response safety remain separate: a safe refusal can answer an unsafe request.</p>
<h2>Does it block harmless requests?</h2>
<p>XSTest includes 250 benign prompts that can look risky out of context, alongside 200 harmful contrasts. This check shows unnecessary flags on the benign prompts and how many harmful contrasts were caught.</p>
{table(['Model','Benign prompts flagged','95% range for false alarms','Harmful prompts caught','Failed benign requests'],xrows)}
<p class="note">A failed request is reported separately, not counted as a correct safe decision. This measures classification of prompts; it is not a response-refusal benchmark or proof of protection against attacks.</p>
<h2>OpenAI labels need two separate views</h2>
<p>The original 1,680 texts have incomplete category labels. Our main table uses the 859 texts with a known binary answer: at least one positive category, or all eight categories explicitly negative. The other 821 are not silently labelled safe.</p>
<p>We also ask separate category questions wherever an annotation exists. The table below summarizes those labelled text/category decisions. Several decisions can refer to the same text; they are not independent new texts.</p>
{table(['Model','Known category decisions','Harmful-content F1'],oai_rows)}
<p class="note">Neither view reproduces the published full-set OpenAI moderation score. Unknown categories are omitted, and our policy wording is explicit.</p>
<details><summary>Recall, false alarms and local response times</summary>
<p>Recall shows how much labelled harmful content was caught. False alarms count benign content incorrectly flagged. Precision measures how often a harmful verdict was correct. Full confidence intervals and confusion counts are in the linked analysis.</p>
{table(['Test / model','Precision','Harmful examples caught','False alarms','Failed requests'],safety_rows)}
{table(['Test / model','Accuracy','Median response','95th percentile','Average precision'],timing_rows)}
<p class="note">Timing comes from serial local calls after a warm-up on a shared workstation with other services running. It can vary with background load and is not a direct comparison with vendor latency. Average precision evaluates the ordering of unsafe probabilities across thresholds and uses only valid probability outputs; it is not a test-tuned operating threshold.</p></details>
<details><summary>Uncertainty, category breakdown and long inputs</summary>
<p>Each comparison resamples the same source prompts for both models, keeping repeated decisions together. Four primary Aegis comparisons use 98.75% intervals to account for testing both tasks at both training sizes. Transfer and 5,000-versus-1,000 comparisons use descriptive 95% intervals. A range crossing zero does not settle which model is better.</p>
{table(['Test','Comparison','F1 change, points','Uncertainty range, points','Confidence'],comparisons)}
<h3>Known OpenAI categories</h3>{table(['Category','Model','Decisions','Harmful F1','Recall','False-positive rate'],category_rows)}
<h3>Inputs longer than the model budget</h3>
<p>All questions and answer choices fit. Eight WildGuard response inputs were shortened by the 2,048-token limit; the main scores keep them. The sensitivity view below excludes those inputs without changing training or the primary test. One training example in the smaller set and three in the larger set were also shortened.</p>
{table(['Test / model','Excluded','Retained','Full-test F1','Unshortened-input F1'],sensitivity) if sensitivity else '<p class="note">Sensitivity measurements are pending.</p>'}
<p class="note">The sensitivity subset has descriptive scores, not its own paired uncertainty interval. Original source content remains local.</p></details>
<details><summary>Published safety models: context, not a leaderboard</summary>
<p>These values were reported by the model authors. We did not run Shieldstral or safeguard, and differences in policies, labels, selected examples and scoring prevent a direct win/loss claim.</p>
<h3>Mistral’s model-card evaluation</h3>{table(['Dataset','Task','Shieldstral 3B','Safeguard 20B'],reference_rows)}
<p class="note"><a href="{esc(mref.get('url','#'))}">Mistral source</a>: F1 as published. Shieldstral uses a 0.5 threshold; safeguard uses high reasoning effort. Dataset-specific retained rows are not established as identical to ours. Published XSTest response/refusal scores are deliberately excluded from our benign-prompt comparison.</p>
<h3>OpenAI’s separate evaluation</h3>{table(['Model','OpenAI moderation F1','ToxicChat F1'],openai_rows)}
<p class="note"><a href="{esc(oref.get('url','#'))}">OpenAI technical report, Table 2</a>: different policy prompts and evaluation from Mistral’s column. Table 2 does not specify its F1 averaging or reasoning effort. Our known-label subset must not be compared as if it were that full-set task.</p></details>
<details><summary>Training data, limitations and evidence</summary>
<p>The 1,000 and 5,000 budgets count labelled decisions, not unique conversations. The labels preserve the dataset’s mixture of human judgments, a model jury and generated refusal data.</p>
{table(['Training decisions','Unique source prompts','Human labels','Model-jury labels','Refusal-augmentation labels'],training_rows)}
<p>Both checkpoints start independently from the same pinned base, use three fixed epochs and one training seed, and keep the final checkpoint. A separate 400-decision development set comes from official validation data. No test-based model or threshold selection is used.</p>
<p>Two incomplete training attempts stopped because memory use and step times kept growing. Neither produced a checkpoint or any test results. The second attempt used masked padding to multiples of 128 tokens, but padding alone did not resolve the memory growth. The frozen data, token limit and learning settings stayed fixed. On 20 development examples, the padding check produced identical choices and probabilities; this checks those examples, not equivalence of entire training runs. See the <a href="../bench/results/moderation-study/runtime_restart.json">restart record</a> and <a href="../bench/results/moderation-study/padding_validation.json">padding check</a>.</p>
<p>Training restarted with the same memory policy for both sizes: synchronize and release unused MPS cache after each optimizer step, with <code>--mps-cache-clear-interval 1</code>. In a <a href="../bench/results/moderation-study/cache_validation.json">three-batch training-only check</a>, this reduced driver-allocated memory for the longest batch from 59.17 GB to 16.22 GB while live tensor memory stayed at 3.38 GB. This check addresses memory use; model quality is measured separately in the held-out results above.</p>
<p>Matching training content is removed using normalized hashes of both prompts and responses, including official test rows omitted from scoring and prior benchmark examples. This cannot rule out paraphrases or knowledge acquired during the base model’s original training.</p>
<p>One broad policy guides the generic tasks. Source datasets draw safety boundaries differently, so transfer results reflect those differences as well as model performance. One training seed does not tell us how stable a result is across repeated training. This study does not establish readiness for unattended moderation.</p>
<p>ToxicChat and BeaverTails have non-commercial terms; WildGuard requires authorized access. Dataset text and model weights are not published by this report.</p>
<p><a href="../docs/moderation-study-protocol.md">Study protocol</a> · <a href="../docs/moderation-reference-notes.md">Reference and source notes</a> · <a href="../bench/moderation_references.json">Published reference values</a> · <a href="../bench/results/moderation-study/analysis.json">Validated results</a> · <a href="../bench/results/moderation-study/data_audit.json">Independent data audit</a> · <a href="../bench/results/moderation-study/preflight.json">Input-length audit</a></p></details>
<footer>Prepared for owner review. This moderation experiment extends the earlier Jev and Laya workflow study.</footer>
</main></body></html>'''


def main():
    def load(path):
        return json.loads(path.read_text()) if path.exists() else {}
    page = build_page(load(RESULTS / "analysis.json"), load(ROOT / "train/moderation-study/study_manifest.json") or load(ROOT / "evidence/study_manifest.json"),
                      load(ROOT / "bench/moderation_references.json"), load(RESULTS / "data_audit.json"))
    output = ROOT / "report/laya-moderation-study.html"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(page)
    print(output)


if __name__ == "__main__":
    main()
