"""Cut isolated phonics sounds out of real words spoken by the neural voice.

Each sound comes from a source word and a cut rule:
  "onset"  - the consonant before the vowel, held longer (s from "sun", m from "mud")
  "cv"     - the consonant(s) plus a short clipped vowel (b + "uh" from "bug")
  "vowel"  - the vowel of a vowel-first word, before the closing consonant (a from "at")
  "coda"   - the voiced tail after an initial voiceless consonant (rrr from "fur")
  "noise"  - a clean breath with the tone of a faint sound (f from "off", th from "bath")
"""
import subprocess
import numpy as np

SR = 24000
HOP = int(0.01 * SR)
WIN = int(0.03 * SR)

CUTS = {
    "a": ("at", "vowel"), "e": ("etch", "vowel"), "i": ("itch", "vowel"),
    "o": ("ox", "vowel"), "u": ("up", "vowel"),
    "s": ("sun", "onset"), "f": ("off", "noise"), "sh": ("shut", "onset"), "th": ("bath", "noise"),
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


def shaped_noise(a, seconds=0.55, level=0.09):
    """A steady breath with the measured tone of the longest fricative in `a`.

    f and th are 30-40 dB quieter than vowels, so amplifying or stretching them
    brings up compression noise. Only their average spectrum is kept, and they
    stay softer than s and sh, as they are in speech.
    """
    rms, per, zcr, low = frames(a)
    mask = (zcr > 2500) & (per < 0.55) & (rms > 0.004 * rms.max())
    best, i = (0, 0), 0
    while i < len(mask):
        j = i
        while j < len(mask) and mask[j]:
            j += 1
        if j - i > best[1] - best[0]:
            best = (i, j)
        i = max(j, i + 1)
    seg = a[best[0] * HOP:(best[0] + int((best[1] - best[0]) * 0.8)) * HOP]
    n = 512
    if len(seg) < n * 2:
        raise ValueError(f"fricative too short to measure: {len(seg) / SR:.3f}s")
    power = np.mean([np.abs(np.fft.rfft(seg[s:s + n] * np.hanning(n))) ** 2
                     for s in range(0, len(seg) - n, n // 4)], axis=0)
    power = np.convolve(power, np.ones(15) / 15, "same")  # smooth away compression artifacts
    size = int(seconds * SR)
    spec = np.fft.rfft(np.random.default_rng(7).standard_normal(size))
    freqs = np.fft.rfftfreq(size, 1 / SR)
    gain = np.sqrt(np.interp(freqs, np.fft.rfftfreq(n, 1 / SR), power)) * (freqs > 800)
    out = np.fft.irfft(spec * gain, size).astype(np.float32)
    return fade(out * level / np.sqrt(np.mean(out ** 2)), 0.04, 0.12)


def cut(a, rule):
    if rule == "noise":
        return shaped_noise(a)
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
        if len(seg) / SR < 0.05:
            raise ValueError(f"consonant cut too short: {len(seg) / SR:.3f}s")
        return fade(stretch(seg, 0.55), 0.02, 0.08)
    if rule == "cv":
        return fade(a[start * HOP:(v0 + 9) * HOP], 0.005, 0.05)
    # vowel / coda: the loud voiced stretch, ended where loudness drops off
    end = v0
    while end < len(rms) and per[end] > 0.45 and rms[end] > 0.15 * top:
        end += 1
    seg = a[v0 * HOP:end * HOP]
    return fade(stretch(seg, 0.45 if rule == "vowel" else 0.6), 0.02, 0.08)
