import unittest
from evaluate_moderation_jev import validate_resume


class ResumeTests(unittest.TestCase):
    def setUp(self):
        self.case = dict(id='a', text_group='hash', expected={'intent':'safe'}, categories=['x'])
        self.row = dict(id='a', text_group='hash', expected='safe', categories=['x'],
                        policy_category=None, model_resolved='jev1', error=None)

    def test_valid_and_partial(self):
        self.assertEqual(validate_resume([], [self.case]), set())
        self.assertEqual(validate_resume([self.row], [self.case]), {'jev1'})

    def test_changes_rejected(self):
        for field in ['id', 'text_group', 'expected', 'categories', 'policy_category']:
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_resume([dict(self.row, **{field:'changed'})], [self.case])

    def test_missing_identity_and_extra_rows(self):
        with self.assertRaises(ValueError):
            validate_resume([dict(self.row, model_resolved=None)], [self.case])
        with self.assertRaises(ValueError):
            validate_resume([self.row], [])

    def test_identity_change(self):
        with self.assertRaises(ValueError):
            validate_resume([self.row, dict(self.row, model_resolved='jev2')], [self.case]*2)


class AnalysisTests(unittest.TestCase):
    def test_matching_results_and_tampered_hash(self):
        import json
        import tempfile
        from pathlib import Path
        from test_moderation_analysis import fixture
        from analyze_moderation_jev import analyze, MODEL
        from moderation_metrics import moderation_metrics
        with tempfile.TemporaryDirectory() as directory:
            data, results, _, _, source = fixture(Path(directory))
            from public_workflow import digest
            source['summary'] = moderation_metrics(source['rows'])
            runs = {}
            for name in ['base', 'fine1000', 'fine5000']:
                blob = json.dumps(dict(source, target=name)).encode()
                (results/('aegis_prompt_'+name+'.json')).write_bytes(blob)
                runs[name] = {'input_sha256':digest(blob), 'summary':source['summary']}
            (results/'analysis.json').write_text(json.dumps({'complete':True,
                'study_sha256':source['study_sha256'], 'tasks':[{'slug':'aegis_prompt','runs':runs}]}))
            source.update(target='jev', dataset='aegis_prompt', model_requested=MODEL,
                          model_resolved=[source['weights_identity']], summary=moderation_metrics(source['rows']))
            path = results/'aegis_prompt_jev.json'
            path.write_text(json.dumps(source))
            output = analyze(data, results)
            self.assertTrue(output['complete'])
            self.assertEqual(output['tasks'][0]['comparisons']['jev_vs_base']['harmful_f1_delta_fine_minus_reference'], 0)
            local_path = results/'aegis_prompt_base.json'
            original_blob = local_path.read_bytes()
            tampered = json.loads(original_blob)
            tampered['target'] = 'fine5000'
            local_path.write_text(json.dumps(tampered))
            with self.assertRaisesRegex(ValueError, 'original validated analysis'):
                analyze(data, results)
            local_path.write_bytes(original_blob)
            source['cases_sha256'] = 'bad'
            path.write_text(json.dumps(source))
            with self.assertRaisesRegex(ValueError, 'cases_sha256'):
                analyze(data, results)

    def test_pending_results_not_scored(self):
        import tempfile
        from pathlib import Path
        from test_moderation_analysis import fixture
        from analyze_moderation_jev import analyze
        with tempfile.TemporaryDirectory() as directory:
            data, results, _, _, _ = fixture(Path(directory))
            output = analyze(data, results)
            self.assertFalse(output['complete'])
            self.assertFalse(output['tasks'][0]['complete'])


if __name__ == '__main__':
    unittest.main()
