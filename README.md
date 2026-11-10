# Phonics Pal

Early reading practice: 2-letter sound matching, word building, and decodable sentences.

**Play it:** https://turnstyle19320.github.io/phonics-pal/

## Voice

Every sound, word, and sentence is a pre-recorded clip in `audio/`, so the page needs no API key and
never falls back to the browser's robotic voice. All clips use one voice, Microsoft's "Aria" neural
voice, picked because its short vowels (bed/bad, hop/hup, pin/pen) measured the most distinct of the
voices tested.

Isolated phonics sounds (/s/, /m/, short a, "buh"…) are cut out of real words Aria speaks ("sun",
"mud", "at", "bug"), because TTS engines read bare sounds as letter names ("ess") or add stray vowels.
Continuous sounds are held with no added vowel; stop sounds keep only a short, clipped vowel.
/f/ and /th/ are so faint that amplifying a cut adds noise, so they are a clean breath built from the
measured tone of Aria's "off" and "bath".

## Pictures

Word Challenge words that can be pictured show a picture so the child knows which word is meant
("sun", not "son"). Pictures are [Twemoji](https://github.com/jdecked/twemoji) graphics
(© Twitter, Inc. and other contributors, licensed [CC-BY 4.0](https://creativecommons.org/licenses/by/4.0/)),
saved in `images/<word>.svg`. Words that can't be pictured clearly (sat, much, when…) show none.

## Regenerating the audio

After changing words or sentences in `index.html`:

```bash
brew install ffmpeg
python3 -m venv venv && ./venv/bin/pip install edge-tts soundfile numpy
./venv/bin/python tools/generate_audio.py
./venv/bin/python tools/vowel_check.py   # flags words whose short vowel sounds off
```

If a word comes out wrong, add a better rendering to `WORD_OVERRIDES` in `tools/generate_audio.py`.
