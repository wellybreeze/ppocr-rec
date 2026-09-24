from __future__ import annotations


def _levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            ins = cur[j - 1] + 1
            delete = prev[j] + 1
            sub = prev[j - 1] + (ca != cb)
            cur.append(min(ins, delete, sub))
        prev = cur
    return prev[-1]


class RecMetrics:
    def __init__(self, ignore_space: bool = True):
        self.ignore_space = ignore_space
        self.reset()

    def reset(self):
        self.correct = 0
        self.total = 0
        self.edit_sum = 0.0

    def update(self, preds: list, gts: list):
        for pred, gt in zip(preds, gts):
            p = pred[0] if isinstance(pred, (tuple, list)) else str(pred)
            t = gt[0] if isinstance(gt, (tuple, list)) else str(gt)
            if self.ignore_space:
                p = p.replace(" ", "")
                t = t.replace(" ", "")
            denom = max(len(p), len(t), 1)
            self.edit_sum += _levenshtein(p, t) / denom
            if p == t:
                self.correct += 1
            self.total += 1

    def results(self) -> dict[str, float]:
        n = max(self.total, 1)
        return {
            "acc": self.correct / n,
            "norm_edit_dis": 1.0 - self.edit_sum / n,
        }
