"""
Fast structural checks across every entry point. No data, no GPU, seconds.

    PYTHONPATH=. python3 scripts/check_consistency.py

Run it after touching anything in `encoding/` or `stats/`. It exists because
this project's entry points share helpers without sharing a test, so a change
that is correct in one module can break another silently -- and has:

* `run_encoding.load_bands` was rewritten to read `args.band_stores`, which
  only `run_encoding.main` sets. `stats.run_permutation` imports that function
  and defines no such attribute, so the permutation entry point was dead on
  arrival and nothing said so until it was run.
* `--n-splits` meant "evaluation folds" under `--eval cv` and "alpha search"
  under `--eval holdout`, so one job selected alphas two different ways in its
  two eval modes, and `run_permutation` carried a comment asking a human to
  keep its own alpha search equal to run_encoding's. It broke the first time
  run_encoding changed.

Each check below is one of those failures turned into an assertion. Add to it
whenever a cross-module assumption is discovered rather than declared.
"""

import sys
import traceback

FAILURES = []


#: Smallest argv that satisfies each module's required flags. Kept explicit so
#: a newly required flag shows up here as a failing check rather than as a
#: silently skipped module.
MIN_ARGV = {
    "encoding.run_prosody_sweep": ["--source", "perlayer_base_emotion",
                                   "--configs", "11"],
    "encoding.run_semantic_sweep": ["--sources", "perlayer_gpt2_k16:8"],
}


def parse_defaults(mod):
    """A namespace of defaults, whichever way this module exposes its CLI.

    The sweeps expose `build_parser()`, run_encoding and run_permutation expose
    `parse_args()`. A check that knows only one of the two silently skips the
    other, which is how a broken entry point passes a green suite.
    """
    argv = MIN_ARGV.get(mod.__name__, [])
    if hasattr(mod, "parse_args"):
        return mod.parse_args(argv)
    if hasattr(mod, "build_parser"):
        return mod.build_parser().parse_args(argv)
    raise AssertionError(f"{mod.__name__} exposes neither parse_args nor build_parser")


def code_of(mod):
    """Module source with comments and docstrings stripped.

    Greping raw source finds the word in the comment explaining why the call
    was removed, which is the opposite of what these checks want to know.
    """
    import io
    import inspect
    import tokenize
    src = inspect.getsource(mod)
    out, prev_end = [], (1, 0)
    for tok in tokenize.generate_tokens(io.StringIO(src).readline):
        if tok.type in (tokenize.COMMENT, tokenize.STRING):
            continue
        out.append(tok.string)
    return " ".join(out)


def check(name):
    def wrap(fn):
        try:
            fn()
            print(f"  ok    {name}")
        except Exception as exc:
            FAILURES.append((name, exc))
            print(f"  FAIL  {name}: {type(exc).__name__}: {exc}")
        return fn
    return wrap


ENTRY_POINTS = [
    "encoding.run_encoding",
    "encoding.run_prosody_sweep",
    "encoding.run_semantic_sweep",
    "stats.run_permutation",
    "stats.analysis",
]

print("entry points import and parse their own defaults")
for mod_name in ENTRY_POINTS:
    @check(mod_name)
    def _(mod_name=mod_name):
        import importlib
        mod = importlib.import_module(mod_name)
        if mod_name != "stats.analysis":       # a library, not a CLI
            parse_defaults(mod)

print("\nshared helpers accept every namespace that reaches them")


@check("load_bands works on a run_permutation namespace")
def _():
    from encoding.run_encoding import resolve_band_stores
    from stats.run_permutation import parse_args
    args = parse_args([])
    stores = resolve_band_stores(args)
    assert set(stores) == {"text", "audio"}, stores
    # The real call also needs feature files; resolving the stores is the part
    # that broke, and it is the part that must not depend on which main() ran.


@check("resolve_band_stores never touches a bare args.band")
def _():
    import inspect
    from encoding.run_encoding import resolve_band_stores
    src = inspect.getsource(resolve_band_stores)
    assert "getattr(args, \"band\"" in src, (
        "resolve_band_stores must use getattr for --band: entry points without "
        "that flag call it through load_bands")


print("\nfold vocabulary means one thing everywhere")


@check("plan_folds gives holdout no evaluation folds")
def _():
    import numpy as np
    from encoding.cv import plan_folds
    ids = np.repeat(np.arange(12), 5)
    ho = plan_folds(ids, "holdout", n_splits=5, alpha_n_splits=None)
    assert ho.evaluation is None, "holdout has no outer loop to split"
    assert len(ho.alpha_search) == 12, "alpha search defaults to leave-one-story-out"
    cv = plan_folds(ids, "cv", n_splits=5, alpha_n_splits=None)
    assert cv.alpha_search is None, "cv builds its alpha folds inside each outer fold"
    assert len(cv.evaluation) == 5


@check("no entry point still calls story_folds directly")
def _():
    import importlib
    offenders = []
    for mod_name in ENTRY_POINTS:
        try:
            src = code_of(importlib.import_module(mod_name))
        except OSError:
            continue
        if "story_folds (" in src or "story_folds(" in src:
            offenders.append(mod_name)
    assert not offenders, (
        f"{offenders} build folds directly, so the meaning of the result is "
        f"set by the call site again. Use encoding.cv.plan_folds.")


@check("sweeps bound the alpha search, run_encoding does not")
def _():
    import importlib
    enc = parse_defaults(importlib.import_module("encoding.run_encoding"))
    assert enc.alpha_n_splits is None, (
        "a final single-configuration model should search alphas "
        "leave-one-story-out")
    for name in ("prosody", "semantic"):
        args = parse_defaults(
            importlib.import_module(f"encoding.run_{name}_sweep"))
        assert args.n_splits is not None, (
            f"the {name} sweep must bound its folds: the fit count is "
            f"n_splits * alpha_n_splits per configuration")


@check("permutation can reuse a published fit instead of refitting")
def _():
    from stats.run_permutation import parse_args, load_observed_fit  # noqa: F401
    args = parse_args([])
    assert hasattr(args, "from_encoding"), (
        "without --from-encoding the permutation refits the observed model, "
        "and its alpha search has to be kept equal to run_encoding's by hand")
    assert args.alpha_n_splits is None, (
        "the fallback alpha search must default to the same thing "
        "run_encoding does, or the two disagree out of the box")


@check("--n-splits legacy spelling still reaches the alpha search")
def _():
    from stats.run_permutation import parse_args
    assert parse_args(["--n-splits", "5"]).alpha_n_splits == 5


print("\nresults record how they were produced")


@check("run_encoding meta names its fold plan")
def _():
    import inspect
    from encoding import run_encoding
    src = inspect.getsource(run_encoding)
    for key in ('"alpha_n_splits"', '"folds"', '"band_stores"'):
        assert key in src, f"meta must record {key} for a later reader to check"


if FAILURES:
    print(f"\n{len(FAILURES)} check(s) failed")
    sys.exit(1)
print("\nall checks passed")
