"""Pre-render every sound, word and sentence in index.html to audio/*.mp3.

The published page plays these files directly, so it needs no API key and never
falls back to the browser's robotic voice. Everything is one voice: Microsoft's
"Aria" neural voice (via edge-tts), chosen because its short vowels
(bed/bad, hop/hup, pin/pen) measured the most distinct of the voices tested. Isolated phonics sounds are cut out of real
words she speaks (see phoneme_cut.py), because every TTS engine tried reads bare
sounds as letter names ("ess", "eff") or adds stray vowels.

Setup:
  brew install ffmpeg
  python3 -m venv venv && ./venv/bin/pip install edge-tts soundfile numpy
  ./venv/bin/python tools/generate_audio.py              # everything
  ./venv/bin/python tools/generate_audio.py sound-f hint-th   # only these clips
"""
import asyncio, json, os, re, subprocess, sys, tempfile
import edge_tts
import numpy as np
import soundfile as sf
from phoneme_cut import CUTS, cut

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "audio")
SR = 24000
VOICE = "en-US-AriaNeural"
TWO_LETTER_EXAMPLES = {"sh": "ship", "ch": "chop", "th": "thin", "wh": "whip", "ck": "duck",
                       "st": "stop", "sl": "slip", "fl": "flag", "cr": "crab", "dr": "drum"}
PHRASES = ["You got it!"]
# Words the voice says wrong on their own, and a rendering that fixes them. Multi-word
# texts are spoken whole and the word is cut out at its timestamp. Found with vowel_check.py.
WORD_OVERRIDES = {
    "cat": "The cat is here.",  # alone it comes out close to "Kate"
    "got": "got!",              # crisper than "got." (which speech recognition missed)
}


def slug(text):
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def page_text():
    html = open(os.path.join(ROOT, "index.html"), encoding="utf-8").read()
    words = set(re.findall(r'\{ word: "(\w+)"', html))
    sentences = [json.loads("[" + s + "]") for s in re.findall(r"sentence: \[(.*?)\] \}", html)]
    decoys = json.loads(re.search(r"const decoys = shuffleArray\((\[.*?\])\)", html).group(1))
    for s in sentences:
        words.update(re.sub(r"[.!]", "", w).lower() for w in s)
    words.update(d.lower() for d in decoys)
    words.update(TWO_LETTER_EXAMPLES.values())
    return sorted(words), [" ".join(s) for s in sentences]


def decode(path):
    raw = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", path, "-ac", "1", "-ar", str(SR),
                          "-f", "f32le", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.float32)


def trim(audio, pad=0.12, thresh=0.005):
    # generous pad: soft onsets like the voicing before "g" start well before the first loud sample
    loud = np.where(np.abs(audio) > thresh * np.max(np.abs(audio)))[0]
    if len(loud) == 0:
        raise ValueError("silent render")
    return audio[max(0, loud[0] - int(pad * SR)):min(len(audio), loud[-1] + int(pad * SR))]


def save(audio, name, normalize=True):
    if normalize:
        audio = audio / max(1e-6, np.max(np.abs(audio))) * 0.9
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        sf.write(tmp.name, audio, SR)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", tmp.name, "-ac", "1",
                    "-b:a", "48k", os.path.join(OUT, name + ".mp3")], check=True)
    os.unlink(tmp.name)


async def edge(text, rate="-10%", raw=False):
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        path = tmp.name
    for attempt in range(4):
        try:
            await edge_tts.Communicate(text, VOICE, rate=rate).save(path)
            break
        except Exception:
            if attempt == 3:
                raise
            await asyncio.sleep(2)
    audio = decode(path)
    os.unlink(path)
    return audio if raw else trim(audio)


async def edge_word(word, rate="-10%"):
    text = WORD_OVERRIDES.get(word, f"{word}.")
    if " " not in text:
        return await edge(text, rate)
    marks, chunks = [], []
    async for ch in edge_tts.Communicate(text, VOICE, rate=rate, boundary="WordBoundary").stream():
        if ch["type"] == "audio":
            chunks.append(ch["data"])
        elif ch["type"] == "WordBoundary":
            marks.append((ch["text"].lower().strip(".,!?"), ch["offset"] / 1e7, ch["duration"] / 1e7))
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        tmp.write(b"".join(chunks))
    audio = decode(tmp.name)
    os.unlink(tmp.name)
    offset, duration = next((o, d) for t, o, d in marks if t == word)
    # start in the quietest 10 ms near the boundary (the stop closure) so no trace of the
    # previous word leaks in
    hop = int(0.01 * SR)
    lo = int((offset - 0.08) * SR)
    quiet = [np.sqrt(np.mean(audio[i:i + hop] ** 2)) for i in range(lo, lo + 13 * hop, hop)]
    start = lo + int(np.argmin(quiet)) * hop
    return audio[start:int((offset + duration + 0.1) * SR)]


async def main(only=()):
    os.makedirs(OUT, exist_ok=True)
    wanted = lambda name: not only or name in only
    names = []
    sounds = {}
    for token, (word, rule) in CUTS.items():
        sounds[token] = cut(await edge(f"{word}.", raw=True), rule)
        if wanted(f"sound-{token}"):
            save(sounds[token], f"sound-{token}", normalize=rule != "noise")
        names.append(f"sound-{token}")
    for token, example in TWO_LETTER_EXAMPLES.items():
        if wanted(f"hint-{token}"):
            gap = np.zeros(int(0.25 * SR), np.float32)
            save(np.concatenate([sounds[token], gap, await edge(f"as in {example}.")]), f"hint-{token}")
        names.append(f"hint-{token}")
    if only:
        print(f"re-rendered {len(only)} clips")
        return

    words, sentences = page_text()
    sem = asyncio.Semaphore(6)

    async def run(name, render):
        async with sem:
            save(await render, name)
            names.append(name)
    await asyncio.gather(*[run(f"word-{w}", edge_word(w)) for w in words],
                         *[run(slug(s), edge(s, "-15%")) for s in sentences + PHRASES])

    with open(os.path.join(OUT, "manifest.json"), "w") as f:
        json.dump(sorted(names), f, indent=0)
    print(f"{len(names)} clips written to {OUT}")


if __name__ == "__main__":
    asyncio.run(main(set(sys.argv[1:])))
