"""Nhan dang giong noi tieng Viet offline bang Vosk (Kaldi) - micro hoac file WAV.

Grammar (tuy chon): gioi han tu vung cua decoder vao tu lenh + ten khu vuc
(command_parser.grammar_words). Moi tu ngoai grammar thanh "[unk]", giam manh loi
"phòng học" -> "phòng họp", "đi đến" -> "địa điểm" so voi nhan dang tu do.
"""

import json
import subprocess
from pathlib import Path

DEFAULT_MODEL = Path.home() / '.cache' / 'vosk' / 'vosk-model-small-vn-0.4'
SAMPLE_RATE = 16000


class SpeechRecognizer:
    def __init__(self, model_path=DEFAULT_MODEL, grammar=None, sample_rate=SAMPLE_RATE):
        from vosk import KaldiRecognizer, Model, SetLogLevel
        SetLogLevel(-1)
        model_path = Path(model_path).expanduser()
        if not model_path.exists():
            raise FileNotFoundError(
                f'Khong thay model Vosk {model_path} - tai vosk-model-small-vn-0.4 (xem README)')
        self._model = Model(str(model_path))
        self._grammar = json.dumps(list(grammar) + ['[unk]'], ensure_ascii=False) if grammar else None
        self.sample_rate = sample_rate
        self._make = lambda: (KaldiRecognizer(self._model, sample_rate, self._grammar)
                              if self._grammar else KaldiRecognizer(self._model, sample_rate))
        self.reset()

    def reset(self):
        self._rec = self._make()
        self._rec.SetWords(True)

    @staticmethod
    def _parse(result_json):
        res = json.loads(result_json)
        words = res.get('result', [])
        conf = sum(w['conf'] for w in words) / len(words) if words else 0.0
        return res.get('text', '').strip(), conf

    def accept(self, pcm):
        """Nap PCM 16-bit mono; tra ve (text, conf) khi ket thuc mot cau, nguoc lai None."""
        if self._rec.AcceptWaveform(pcm):
            text, conf = self._parse(self._rec.Result())
            return (text, conf) if text else None
        return None

    def partial(self):
        return json.loads(self._rec.PartialResult()).get('partial', '')

    def flush(self):
        text, conf = self._parse(self._rec.FinalResult())
        self.reset()
        return (text, conf) if text else None

    def transcribe_file(self, path):
        """Nhan dang ca file audio (moi dinh dang ffmpeg doc duoc) -> list[(text, conf)]."""
        pcm = subprocess.run(
            ['ffmpeg', '-loglevel', 'error', '-i', str(path), '-ar', str(self.sample_rate),
             '-ac', '1', '-f', 's16le', '-'], check=True, capture_output=True).stdout
        out = []
        for i in range(0, len(pcm), 8000):
            r = self.accept(pcm[i:i + 8000])
            if r:
                out.append(r)
        r = self.flush()
        if r:
            out.append(r)
        return out


def open_microphone(device='default', sample_rate=SAMPLE_RATE):
    """Luong PCM tu micro qua arecord (ALSA/PulseAudio), khong can PortAudio/pyaudio."""
    return subprocess.Popen(
        ['arecord', '-q', '-D', device, '-f', 'S16_LE', '-r', str(sample_rate), '-c', '1',
         '-t', 'raw'], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
