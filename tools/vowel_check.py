"""Check that every short-vowel word in audio/ really has the short vowel it is spelled with.

Measures the first two formants of each word's vowel, builds per-vowel averages
from all the words, and flags any word whose vowel sits closer to a different
short vowel (e.g. "cat" said like "Kate"). Run after generate_audio.py:
  ./venv/bin/python tools/vowel_check.py
"""
import glob, os, re, subprocess, sys
import numpy as np
from phoneme_cut import frames, HOP, SR

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SR2 = 11025
SHORT = {"a": "æ", "e": "ɛ", "i": "ɪ", "o": "ɑ", "u": "ʌ"}
CVC = re.compile(r"^(sh|ch|th|wh|st|sl|fl|cr|dr|fr|[bcdfghjklmnprstvwz])([aeiou])(ck|sh|ch|th|st|[bdfgklmnpstxz])$")


def lpc(x, order):
    r = np.correlate(x, x, "full")[len(x) - 1:len(x) + order]
    a = np.zeros(order + 1); a[0] = 1; e = r[0]
    for i in range(1, order + 1):
        k = -(r[i] + a[1:i] @ r[i - 1:0:-1]) / e
        a[1:i + 1] += k * np.concatenate([a[i - 1:0:-1], [1]]); e *= 1 - k * k
    return a


def vowel_formants(path):
    raw = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", path, "-ac", "1", "-ar", str(SR),
                          "-f", "f32le", "-"], capture_output=True, check=True).stdout
    a = np.frombuffer(raw, np.float32)
    rms, per, zcr, low = frames(a)
    idx = [i for i in range(len(rms)) if per[i] > 0.6 and rms[i] > 0.3 * rms.max() and low[i] < 0.6]
    if not idx:
        return None
    seg = a[idx[0] * HOP:(idx[-1] + 3) * HOP]
    seg = seg[len(seg) // 5:len(seg) // 2]  # early-middle: past the consonant, before any glide
    x = np.interp(np.arange(0, len(seg), SR / SR2), np.arange(len(seg)), seg)
    out = []
    for s in range(0, len(x) - 256, 64):
        w = x[s:s + 256]; w = np.append(w[0], w[1:] - 0.95 * w[:-1]) * np.hamming(256)
        f = [(np.angle(q) * SR2 / 2 / np.pi, -np.log(abs(q)) * SR2 / np.pi)
             for q in np.roots(lpc(w, 12)) if np.imag(q) > 0.01]
        f = sorted(fr for fr, bw in f if 200 < fr < 4000 and bw < 350)
        if len(f) >= 2:
            out.append(f[:2])
    return np.median(out, axis=0) if out else None


bark = lambda f: 26.81 * f / (1960 + f) - 0.53


def main():
    words = {}
    for path in sorted(glob.glob(os.path.join(ROOT, "audio", "word-*.mp3"))):
        w = os.path.basename(path)[5:-4]
        m = CVC.match(w)
        if m:
            f = vowel_formants(path)
            if f is not None:
                words[w] = (m.group(2), bark(f), f.astype(int))
    centroids = {v: np.median([b for vv, b, _ in words.values() if vv == v], axis=0) for v in SHORT}
    bad = 0
    for w, (v, b, f) in words.items():
        nearest = min(centroids, key=lambda c: np.linalg.norm(b - centroids[c]))
        margin = np.linalg.norm(b - centroids[nearest]) - np.linalg.norm(b - centroids[v])
        if nearest != v:
            bad += 1
            print(f"FLAG {w:6} spelled /{SHORT[v]}/ but sounds closest to /{SHORT[nearest]}/  F1,F2={f.tolist()}")
    print(f"{len(words)} short-vowel words checked, {bad} flagged")
    return bad


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
