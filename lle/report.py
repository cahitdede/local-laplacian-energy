"""Uniform PASS/FAIL bookkeeping."""
import time


class Report:
    def __init__(self, title):
        self.title, self.rows, self.t0 = title, [], time.time()
        print("\n" + "=" * 78 + "\n" + title + "\n" + "=" * 78, flush=True)

    def check(self, name, ok, detail=""):
        ok = bool(ok)
        self.rows.append((name, ok))
        line = "   %s  %s" % ("PASS" if ok else "FAIL", name)
        if detail:
            line += "\n         " + detail
        print(line, flush=True)
        return ok

    def note(self, text):
        print("         " + text, flush=True)

    def summary(self):
        good = sum(ok for _, ok in self.rows)
        print("   -- %d / %d passed  (%.1f s)" % (good, len(self.rows), time.time() - self.t0),
              flush=True)
        return self.rows
