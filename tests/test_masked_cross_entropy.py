"""CPU regression tests; run with unittest discovery from the repository root."""
import unittest

import torch
import torch.nn.functional as F

from metrics.loss_functions import MaskedCrossEntropyLoss


class MaskedCrossEntropyTests(unittest.TestCase):
    def test_valid_pixels_preserve_original_values_and_gradients(self):
        # One all-void image alongside a valid/mixed image must still train.
        logits = torch.linspace(-2, 2, 2 * 2 * 2 * 19, dtype=torch.float64)
        logits = logits.reshape(2, 2, 2, 19).requires_grad_()
        labels = torch.tensor([[[[19], [19]], [[19], [19]]],
                               [[[0], [19]], [[18], [3]]]])
        mask = labels != 19
        valid = mask.squeeze(-1)
        # Original log_softmax/gather formula on the selected 2D matrix.
        selected = logits[valid]
        targets = labels.squeeze(-1)[valid].unsqueeze(-1)
        original = -F.log_softmax(selected, dim=1).gather(1, targets)
        for mean in (True, False):
            with self.subTest(mean=mean):
                actual = MaskedCrossEntropyLoss(mean=mean)(logits, (labels, mask))
                expected = original.mean() if mean else original
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                actual_grad, = torch.autograd.grad(actual.sum(), logits, retain_graph=True)
                expected_grad, = torch.autograd.grad(expected.sum(), logits, retain_graph=True)
                torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=0)
                self.assertEqual(torch.count_nonzero(actual_grad[~valid]).item(), 0)
        reference = F.cross_entropy(selected, targets.squeeze(-1))
        torch.testing.assert_close(MaskedCrossEntropyLoss()(logits, (labels, mask)), reference)

    def test_no_mask_preserves_cross_entropy(self):
        logits = torch.tensor([[1., 2., -1.], [0., 1., 3.]])
        labels = torch.tensor([1, 2])
        for ground_truth in (labels, (labels,), (labels, None)):
            torch.testing.assert_close(MaskedCrossEntropyLoss()(logits, ground_truth),
                                       F.cross_entropy(logits, labels))

    def test_all_void_stops_before_optimizer_update(self):
        model = torch.nn.Linear(2, 19)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.1)
        inputs = torch.ones(1, 2)
        # Populate AdamW's moments so returning zero would not be a safe skip.
        F.cross_entropy(model(inputs), torch.tensor([0])).backward()
        optimizer.step()
        optimizer.zero_grad()
        before = [p.detach().clone() for p in model.parameters()]
        steps = [optimizer.state[p]['step'].clone() for p in model.parameters()]
        with self.assertRaisesRegex(ValueError, 'no valid target pixels'):
            loss = MaskedCrossEntropyLoss()(model(inputs),
                (torch.tensor([[19]]), torch.tensor([[False]])))
            loss.backward()
            optimizer.step()
        for p, snapshot, step in zip(model.parameters(), before, steps):
            torch.testing.assert_close(p, snapshot, rtol=0, atol=0)
            torch.testing.assert_close(optimizer.state[p]['step'], step, rtol=0, atol=0)
            self.assertIsNone(p.grad)

    def test_all_void_unreduced_output_remains_empty(self):
        logits = torch.zeros(1, 2, 2, 19, requires_grad=True)
        labels = torch.full((1, 2, 2, 1), 19)
        losses = MaskedCrossEntropyLoss(mean=False)(logits, (labels, labels != 19))
        self.assertEqual(tuple(losses.shape), (0, 1))
        self.assertTrue(torch.isfinite(losses).all())


if __name__ == '__main__':
    unittest.main()
