"""Cut isolated phonics sounds out of real words spoken by the neural voice.

Each sound comes from a source word and a cut rule:
  "onset"  - the consonant before the vowel, held longer (s from "sun", m from "mud")
  "cv"     - the consonant(s) plus a short clipped vowel (b + "uh" from "bug")
  "vowel"  - the vowel of a vowel-first word, before the closing consonant (a from "at")
  "coda"   - the voiced tail after an initial voiceless consonant (rrr from "fur")
"""
import subprocess
import numpy as np

SR = 24000
HOP = int(0.01 * SR)
WIN = int(0.03 * SR)

CUTS = {
    "a": ("at", "vowel"), "e": ("etch", "vowel"), "i": ("itch", "vowel"),
    "o": ("ox", "vowel"), "u": ("up", "vowel"),
    "s": ("sun", "onset"), "f": ("fun", "onset"), "sh": ("shut", "onset"), "th": ("thumb", "onset"),
    "m": ("mud", "onset"), "n": ("nut", "onset"), "l": ("lull", "onset"), "v": ("van", "onset"),
    "z": ("zap", "onset"), "r": ("fur", "coda"),
    "b": ("bug", "cv"), "c": ("cup", "cv"), "k": ("cup", "cv"), "ck": ("cup", "cv"),
    "d": ("dug", "cv"), "g": ("gum", "cv"), "h": ("hut", "cv"), "j": ("jug", "cv"),
    "p": ("pup", "cv"), "t": ("tub", "cv"), "w": ("one", "cv"), "wh": ("one", "cv"),
    "ch": ("chug", "cv"),
    "st": ("stuff", "cv"), "sl": ("slug", "cv"), "fl": ("fluff", "cv"), "cr": ("crumb", "cv"),
    "dr": ("drum", "cv"),
}


def frames(a):
    """Per 10 ms frame: rms, periodicity (0..1), zero-crossing rate (Hz), share of energy below 500 Hz."""
    n = max(1, (len(a) - WIN) // HOP)
    rms, per, zcr, low = np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
    bins = np.fft.rfftfreq(WIN, 1 / SR)
    lo, hi = int(SR / 400), int(SR / 75)
    for i in range(n):
        x = a[i * HOP:i * HOP + WIN]
        rms[i] = np.sqrt(np.mean(x ** 2))
        zcr[i] = np.mean(np.abs(np.diff(np.sign(x)))) / 2 * SR
        x = x - x.mean()
        ac = np.correlate(x, x, "full")[WIN - 1:]
        per[i] = ac[lo:hi].max() / ac[0] if ac[0] > 1e-9 else 0
        spec = np.abs(np.fft.rfft(x * np.hanning(WIN))) ** 2
        low[i] = spec[bins < 500].sum() / max(spec.sum(), 1e-12)
    return rms, per, zcr, low


def stretch(a, target):
    """Lengthen a steady sound to about `target` seconds without changing pitch."""
    factor = len(a) / SR / target
    if factor >= 1:
        return a
    filters = []
    while factor < 0.5:
        filters.append("atempo=0.5"); factor /= 0.5
    filters.append(f"atempo={factor:.3f}")
    out = subprocess.run(["ffmpeg", "-loglevel", "error", "-f", "f32le", "-ar", str(SR), "-ac", "1",
                          "-i", "-", "-af", ",".join(filters), "-f", "f32le", "-"],
                         input=a.astype(np.float32).tobytes(), capture_output=True, check=True).stdout
    return np.frombuffer(out, np.float32)


def fade(a, fin=0.01, fout=0.04):
    a = a.copy()
    i, o = int(fin * SR), min(len(a) // 2, int(fout * SR))
    a[:i] *= np.linspace(0, 1, i)
    a[-o:] *= np.linspace(1, 0, o)
    return a


def cut(a, rule):
    rms, per, zcr, low = frames(a)
    top = rms.max()
    start = int(np.argmax(rms > 0.006 * top))
    # Vowel = loud, periodic, and not a low hum (nasals and l keep their energy under 500 Hz)
    vowel = [i for i in range(start, len(rms))
             if per[i] > 0.6 and rms[i] > 0.45 * top and zcr[i] < 2500 and low[i] < 0.6]
    v0 = vowel[0] if vowel else start
    if v0 - start > 3:
        # skip quiet lead-in, judged against the consonant's typical loudness (not its peak, which
        # already includes the vowel's rising edge) so weak sounds like f survive
        typical = np.percentile(rms[start:v0], 75)
        start += int(np.argmax(rms[start:v0] > 0.25 * typical))
    if rule == "onset":
        seg = a[start * HOP:v0 * HOP]
        return fade(stretch(seg, 0.55), 0.02, 0.08)
    if rule == "cv":
        return fade(a[start * HOP:(v0 + 9) * HOP], 0.005, 0.05)
    # vowel / coda: the loud voiced stretch, ended where loudness drops off
    end = v0
    while end < len(rms) and per[end] > 0.45 and rms[end] > 0.15 * top:
        end += 1
    seg = a[v0 * HOP:end * HOP]
    return fade(stretch(seg, 0.45 if rule == "vowel" else 0.6), 0.02, 0.08)
