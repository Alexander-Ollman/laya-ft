"""CPU-only checks for public trainer inputs; no model packages are imported."""
import subprocess
import sys
import unittest
from pathlib import Path

from train_public_laya import DATA, parse_args, run_identity, token_budgets, validate_splits


class PublicTrainerInputsTest(unittest.TestCase):
    def test_default_and_explicit_arguments(self):
        defaults = parse_args(["--out", "checkpoint"])
        self.assertEqual(defaults.data, DATA)
        self.assertEqual(defaults.seed, 7)
        args = parse_args(["--out", "checkpoint", "--data", "five-datasets/emotion", "--seed", "19"])
        self.assertEqual(args.data, Path("five-datasets/emotion"))
        self.assertEqual(args.seed, 19)

    def test_model_libraries_not_needed_for_import_or_help(self):
        code = "import sys; import train_public_laya; assert 'torch' not in sys.modules; assert 'laya' not in sys.modules"
        subprocess.run([sys.executable, "-c", code], cwd=Path(__file__).parent, check=True)

    def test_identity_preserves_banking_and_separates_datasets(self):
        self.assertEqual(run_identity({"dataset": "PolyAI/banking77"}), ("banking77", "public/PolyAI-banking77"))
        self.assertEqual(run_identity({"dataset": "deepset/prompt-injections", "slug": "prompt-injection"}),
                         ("prompt-injection", "public/deepset-prompt-injections"))
        self.assertEqual(run_identity({"dataset": "dair-ai/emotion"})[0], "dair-ai-emotion")

    def test_configurable_token_budgets(self):
        self.assertEqual(token_budgets({}), {"max_len": 1024, "head_max_len": 768})
        self.assertEqual(token_budgets({"protocol": {"max_len": 512, "head_max_len": 256}}),
                         {"max_len": 512, "head_max_len": 256})
        for invalid in (0, -1, "512", True):
            with self.assertRaises(ValueError):
                token_budgets({"protocol": {"max_len": invalid}})

    def test_rejects_normalized_overlap_between_every_split_pair(self):
        def case(text):
            return {"state": {"request": text}}
        validate_splits([case("train")], [case("development")], [case("heldout")])
        for left, right in ((0, 1), (0, 2), (1, 2)):
            splits = [[case("first")], [case("second")], [case("third")]]
            splits[left] = [case("Same Request!")]
            splits[right] = [case("same request")]
            with self.assertRaisesRegex(ValueError, "overlaps across splits"):
                validate_splits(*splits)


if __name__ == "__main__":
    unittest.main()
