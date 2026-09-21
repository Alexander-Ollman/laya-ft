"""CPU-only padding invariants; GPU/model equivalence is checked separately."""
import unittest

import torch

from moderation_padding import pad_batch


def batch(length=129, mask_dtype=torch.long):
    return {
        "input_ids": torch.arange(2*length, dtype=torch.long, device="cpu").reshape(2, length),
        "attention_mask": torch.ones((2, length), dtype=mask_dtype, device="cpu"),
        "marker_pos": torch.tensor([[3, 9], [4, 8]], device="cpu"),
        "marker_mask": torch.ones((2, 2), dtype=torch.bool, device="cpu"),
        "target": torch.tensor([[.9, .1], [.1, .9]], device="cpu"),
        "qtype": torch.tensor([0, 0], device="cpu"),
        "meta": [{"id": "one"}, {"id": "two"}],
    }


class ModerationPaddingTests(unittest.TestCase):
    def test_rounds_only_sequence_axis_and_masks_every_added_token(self):
        original = batch()
        original["attention_mask"][1, -7:] = 0
        before_ids = original["input_ids"].clone()
        before_mask = original["attention_mask"].clone()
        padded = pad_batch(original, pad_id=17, multiple=128)
        self.assertEqual(tuple(padded["input_ids"].shape), (2, 256))
        self.assertEqual(tuple(padded["attention_mask"].shape), (2, 256))
        self.assertTrue(torch.equal(padded["input_ids"][:, :129], before_ids))
        self.assertTrue(torch.equal(padded["attention_mask"][:, :129], before_mask))
        self.assertTrue(torch.all(padded["input_ids"][:, 129:] == 17).item())
        self.assertEqual(torch.count_nonzero(padded["attention_mask"][:, 129:]).item(), 0)
        self.assertTrue(torch.equal(original["input_ids"], before_ids))
        self.assertTrue(torch.equal(original["attention_mask"], before_mask))
        self.assertEqual(tuple(original["input_ids"].shape), (2, 129))
        for key in ("marker_pos", "marker_mask", "target", "qtype", "meta"):
            self.assertIs(padded[key], original[key])

    def test_zero_passthrough_and_already_aligned_sequences(self):
        original = batch(128)
        self.assertIs(pad_batch(original, pad_id=0, multiple=0), original)
        aligned = pad_batch(original, pad_id=0, multiple=128)
        self.assertEqual(tuple(aligned["input_ids"].shape), (2, 128))
        self.assertTrue(torch.equal(aligned["input_ids"], original["input_ids"]))
        for invalid in (-128, True, 128., "128"):
            with self.subTest(multiple=invalid), self.assertRaises(ValueError):
                pad_batch(original, pad_id=0, multiple=invalid)

    def test_trainer_padding_is_opt_in(self):
        from train_public_laya import parse_args
        self.assertEqual(parse_args(["--out", "checkpoint"]).pad_to_multiple, 0)
        self.assertEqual(parse_args(["--out", "checkpoint", "--pad-to-multiple", "128"]).pad_to_multiple, 128)

    def test_dtypes_devices_and_short_or_maximal_lengths_preserved(self):
        for length, expected in ((1, 128), (127, 128), (128, 128), (129, 256), (2048, 2048)):
            for mask_dtype in (torch.long, torch.bool, torch.float32):
                with self.subTest(length=length, dtype=mask_dtype):
                    original = batch(length, mask_dtype)
                    padded = pad_batch(original, pad_id=0, multiple=128)
                    self.assertEqual(padded["input_ids"].shape[1], expected)
                    for key in ("input_ids", "attention_mask"):
                        self.assertEqual(padded[key].dtype, original[key].dtype)
                        self.assertEqual(padded[key].device, original[key].device)

    def test_masked_token_aggregation_and_marker_tokens_unchanged(self):
        original = batch(129)
        original["attention_mask"][1, 115:] = 0
        padded = pad_batch(original, pad_id=10000, multiple=128)
        # A large sentinel catches accidental attention on padded IDs.
        for rows in (original, padded):
            rows["masked_sum"] = (rows["input_ids"]*rows["attention_mask"]).sum(dim=1)
            rows["marker_ids"] = torch.gather(rows["input_ids"], 1, rows["marker_pos"])
        self.assertTrue(torch.equal(original["masked_sum"], padded["masked_sum"]))
        self.assertTrue(torch.equal(original["marker_ids"], padded["marker_ids"]))


if __name__ == "__main__":
    unittest.main()
