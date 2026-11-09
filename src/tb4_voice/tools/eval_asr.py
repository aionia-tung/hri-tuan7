#!/usr/bin/env python3
"""Danh gia nhan dang giong noi + phan tich cau lenh tren bo cau lenh mau.

    # 1) tao audio mau bang gTTS (can Internet, chi lam 1 lan) -> test_audio/*.mp3
    python3 tools/eval_asr.py --make
    # 2) danh gia Vosk tu do vs Vosk + grammar, sach va co nhieu trang (SNR 30, 20 dB)
    python3 tools/eval_asr.py

Co the thay bang giong that: ghi file WAV vao test_audio/ va them dong "<file>\t<cau>"
vao test_audio/transcripts.tsv.
Mot cau duoc tinh "dung" neu ket qua parse (intent, lo trinh, khu vuc khong ton tai) trung
voi ket qua parse cua cau goc.
"""

import argparse
import csv
import subprocess
import sys
from pathlib import Path

import numpy as np

PKG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG))

from tb4_voice.command_parser import CommandParser, grammar_phrases, tokens  # noqa: E402
from tb4_voice.speech import SpeechRecognizer  # noqa: E402
from tb4_voice.zones import ZoneMap  # noqa: E402

AUDIO = PKG / 'test_audio'
COMMANDS = [
    'Đi đến khu vực A.',
    'Di chuyển từ khu vực A đến khu vực B.',
    'Đi đến phòng thí nghiệm.',
    'Đi tới hành lang.',
    'Robot hãy đi đến thư viện.',
    'Đi từ phòng học qua hành lang rồi đến phòng thí nghiệm.',
    'Về sảnh chính.',
    'Đi đến khu vực xê.',
    'Đi đến căng tin.',
    'Đi đến phòng y tế.',
    'Đi đến văn phòng.',
    'Di chuyển đến phòng kho.',
    'Đi từ khu vực đê qua khu vực e rồi đến khu vực gờ.',
    'Đi đến nhà bếp.',
    'Đi đến khu vực ích.',
    'Dừng lại.',
]


# Kich ban video demo (source:=files): robot bat dau o Sanh chinh (E, dock)
DEMO = [
    'Đi đến phòng thí nghiệm.',                                  # E -> G
    'Đi đến nhà bếp.',                                           # khu vuc khong ton tai
    'Di chuyển từ khu vực E đến khu vực xê.',                    # dang o G -> qua E roi den C
    'Đi đến khu vực A.',                                         # C -> A
    'Đi từ phòng học qua hành lang rồi đến thư viện.',           # nhieu diem: A -> C -> F
]


def make_audio():
    from gtts import gTTS
    AUDIO.mkdir(exist_ok=True)
    for i, text in enumerate(DEMO, start=1):
        gTTS(text, lang='vi').save(str(AUDIO / f'demo_{i}.mp3'))
        print(f'demo_{i}.mp3', text)
    rows = []
    for i, text in enumerate(COMMANDS, start=1):
        name = f'cmd{i:02d}.mp3'
        gTTS(text, lang='vi').save(str(AUDIO / name))
        rows.append(f'{name}\t{text}')
        print(name, text)
    (AUDIO / 'transcripts.tsv').write_text('\n'.join(rows) + '\n', encoding='utf-8')


def load_pcm(path, rate=16000):
    pcm = subprocess.run(['ffmpeg', '-loglevel', 'error', '-i', str(path), '-ar', str(rate),
                          '-ac', '1', '-f', 's16le', '-'], check=True, capture_output=True).stdout
    return np.frombuffer(pcm, np.int16)


def add_noise(sig, snr_db, seed=0):
    """Tron nhieu trang Gauss voi SNR cho truoc (tinh tren doan co tieng)."""
    x = sig.astype(np.float32)
    voiced = x[np.abs(x) > 0.02 * np.abs(x).max()]
    p_sig = float(np.mean(voiced ** 2)) if voiced.size else 1.0
    noise = np.random.default_rng(seed).normal(0, np.sqrt(p_sig / 10 ** (snr_db / 10)), x.shape)
    pad = np.random.default_rng(seed + 1).normal(0, noise.std(), 8000)   # 0.5 s nen nhieu
    return np.clip(np.concatenate([pad, x + noise, pad]), -32768, 32767).astype(np.int16)


def recognize(rec, sig):
    pcm = sig.tobytes()
    texts = []
    for i in range(0, len(pcm), 8000):
        r = rec.accept(pcm[i:i + 8000])
        if r:
            texts.append(r[0])
    r = rec.flush()
    if r:
        texts.append(r[0])
    return ' '.join(texts)


def wer(ref, hyp):
    r, h = tokens(ref), tokens(hyp)
    d = np.zeros((len(r) + 1, len(h) + 1), int)
    d[:, 0], d[0, :] = range(len(r) + 1), range(len(h) + 1)
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1,
                          d[i - 1, j - 1] + (r[i - 1] != h[j - 1]))
    return d[-1, -1], len(r)


def outcome(cmd):
    return (cmd.intent, tuple(cmd.route), bool(cmd.unknown))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--make', action='store_true', help='tao audio mau bang gTTS')
    ap.add_argument('--zones', default=str(PKG / 'maps' / 'zones.yaml'))
    ap.add_argument('--snr', type=float, nargs='+', default=[30.0, 20.0])
    ap.add_argument('--csv', help='ghi chi tiet tung cau')
    args = ap.parse_args()
    if args.make:
        make_audio()
        return

    zone_map = ZoneMap.load(args.zones)
    parser = CommandParser(zone_map.zones)
    recs = {'tu do': SpeechRecognizer(), 'grammar': SpeechRecognizer(
        grammar=grammar_phrases(zone_map.zones))}
    items = [line.split('\t') for line in
             (AUDIO / 'transcripts.tsv').read_text(encoding='utf-8').splitlines() if line]
    rows, stats = [], {}
    for name, ref in items:
        clean = load_pcm(AUDIO / name)
        expected = outcome(parser.parse(ref))
        conditions = [('sach', clean)] + [(f'SNR {snr:g} dB', add_noise(clean, snr))
                                          for snr in args.snr]
        for cond, sig in conditions:
            for mode, rec in recs.items():
                hyp = recognize(rec, sig)
                cmd = parser.parse(hyp)
                ok = outcome(cmd) == expected
                err, n = wer(ref, hyp)
                s = stats.setdefault((cond, mode), [0, 0, 0, 0])
                s[0] += err
                s[1] += n
                s[2] += ok
                s[3] += 1
                rows.append({'file': name, 'condition': cond, 'mode': mode, 'reference': ref,
                             'hypothesis': hyp, 'route': '>'.join(cmd.route),
                             'unknown': ';'.join(cmd.unknown), 'intent_ok': int(ok)})
                mark = 'OK ' if ok else 'SAI'
                print(f'[{mark}] {cond:<10} {mode:<7} {ref:<52} -> {hyp}  {cmd.route}'
                      f'{" unknown=" + str(cmd.unknown) if cmd.unknown else ""}')
    print('\nDieu kien    Che do    WER     Dung lenh')
    for (cond, mode), (err, n, ok, total) in stats.items():
        print(f'{cond:<12} {mode:<8} {100 * err / n:5.1f}%  {ok}/{total} ({100 * ok / total:.0f}%)')
    if args.csv:
        Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
        with open(args.csv, 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)


if __name__ == '__main__':
    main()
