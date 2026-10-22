import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import icc_reader as R  # noqa: E402
import own_gamut as G  # noqa: E402
import press_model as P  # noqa: E402

PROFILE = os.environ.get("PRESS_ICC", "/app/press.icc")
H = dict(np.load(os.path.join(HERE, "hidden_points.npz")))
_cache = {}
GAMUT_MEAN, GAMUT_P95 = 0.55, 0.90


def profile():
    if "p" not in _cache:
        assert os.path.isfile(PROFILE), "/app/press.icc is missing"
        try:
            _cache["p"] = R.Profile(open(PROFILE, "rb").read())
        except R.ICCError as e:
            _cache["p"] = str(e)
    assert not isinstance(_cache["p"], str), "profile rejected by the strict parser: %s" % _cache["p"]
    return _cache["p"]


def finite(x, what):
    x = np.asarray(x, float)
    assert np.isfinite(x).all(), "%s contains non-finite values" % what
    return x


def a2b(cmyk):
    return finite(profile().a2b(cmyk), "A2B1 output")


def b2a(lab):
    return finite(profile().b2a(lab), "B2A1 output")


def test_profile_is_valid():
    profile()


# A2B1 is the only part of the profile graded against the hidden press.
def test_forward_accuracy():
    e = P.delta_e2000(a2b(H["F"]), H["Flab"])
    print("forward mean %.3f p95 %.3f" % (e.mean(), np.percentile(e, 95)))
    assert e.mean() <= 0.60, "A2B1 mean error %.2f exceeds 0.60" % e.mean()
    assert np.percentile(e, 95) <= 1.80, "A2B1 95th percentile error %.2f exceeds 1.80" % np.percentile(e, 95)


# Every B2A1 check below runs the B2A1 result back through the profile's own A2B1, so a forward-model
# error is graded once (above) and not again here.
def test_inverse_accuracy():
    e = P.delta_e2000(a2b(b2a(H["Ilab"])), H["Ilab"])
    print("inverse mean %.3f p95 %.3f" % (e.mean(), np.percentile(e, 95)))
    assert e.mean() <= 0.65, "A2B1(B2A1(lab)) mean error %.2f exceeds 0.65" % e.mean()
    assert np.percentile(e, 95) <= 1.80, "A2B1(B2A1(lab)) 95th percentile error %.2f exceeds 1.80" % np.percentile(e, 95)


def test_ink_limit():
    for key, what in (("R", "dark and far out-of-gamut colours"), ("Ilab", "printable colours"),
                      ("Gt", "out-of-gamut colours"), ("Nlab", "neutral colours"),
                      ("S0lab", "smoothness colours"), ("S1lab", "smoothness colours")):
        tac = b2a(H[key]).sum(1).max() * 100
        print("max total ink on %s %.2f%%" % (what, tac))
        assert tac <= 300.5, "B2A1 total ink reaches %.1f%% on %s" % (tac, what)


def test_neutral_axis():
    lab = a2b(b2a(H["Nlab"]))
    e = P.delta_e2000(lab, H["Nlab"])
    print("neutral max error %.3f max |a|,|b| %.3f" % (e.max(), np.abs(lab[:, 1:]).max()))
    assert e.max() <= 1.0, "neutral error %.2f exceeds 1.0" % e.max()
    assert np.abs(lab[:, 1:]).max() <= 0.8, "neutral a/b reaches %.2f, limit 0.8" % np.abs(lab[:, 1:]).max()


def test_b2a_smoothness():
    d = np.abs(b2a(H["S0lab"]) - b2a(H["S1lab"])).max(1)
    print("smoothness max jump %.3f, %.2f%% of pairs above 0.16" % (d.max(), 100 * (d > 0.16).mean()))
    assert d.max() <= 0.60, "B2A1 jumps by %.2f between colours 1.5 apart, limit 0.60" % d.max()
    assert (d > 0.16).mean() <= 0.05, "%.1f%% of the colour pairs jump by more than 0.16, limit 5%%" % (100 * (d > 0.16).mean())


def test_gamut_clipping():
    err = P.delta_e2000(a2b(b2a(H["Gt"])), H["Gt"])
    best, _ = G.best_reachable(lambda q: profile().a2b(q), H["Gt"])
    excess = np.maximum(err - best, 0.0)
    print("gamut excess mean %.3f p95 %.3f" % (excess.mean(), np.percentile(excess, 95)))
    assert excess.mean() <= GAMUT_MEAN, "out-of-gamut colours land %.2f farther than the closest colour A2B1 reaches within 300%% ink on average, limit %.2f" % (excess.mean(), GAMUT_MEAN)
    assert np.percentile(excess, 95) <= GAMUT_P95, "95th percentile excess over the closest reachable colour is %.2f, limit %.2f" % (np.percentile(excess, 95), GAMUT_P95)
