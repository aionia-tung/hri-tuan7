"""Phan tich cau lenh dieu huong tieng Viet (van ban tu ASR) thanh lo trinh qua cac khu vuc.

Vi du:
    "Đi đến khu vực A."                          -> route [A]
    "Di chuyển từ khu vực A đến khu vực B."      -> route [A, B], start = A
    "Đi đến phòng thí nghiệm."                   -> route [G]
    "Đi từ phòng A qua hành lang C rồi đến B"    -> route [A, C, B]
    "Đi đến nhà bếp"                             -> unknown ["nhà bếp"]
    "Dừng lại"                                   -> intent cancel

So khop tren token da bo dau (fold) de chap nhan ca van ban go khong dau; ten khu vuc
duoc so khop theo cum dai nhat truoc ("phòng thí nghiệm" truoc "phòng").
"""

import re
import unicodedata
from dataclasses import dataclass, field
from difflib import SequenceMatcher

# Cach doc chu cai tieng Viet (co dau) - dung cho ca grammar Vosk lan so khop
LETTER_NAMES = {
    'A': ['a'], 'B': ['bê', 'bờ'], 'C': ['xê', 'cờ'], 'D': ['đê', 'dê'], 'E': ['e'],
    'F': ['ép', 'phờ'], 'G': ['gờ'], 'H': ['hát', 'hờ'], 'I': ['i'], 'J': ['gi'],
    'K': ['ca'], 'L': ['lờ'], 'M': ['em', 'mờ'], 'N': ['en', 'nờ'], 'O': ['o'], 'P': ['pê'],
    'Q': ['quy'], 'R': ['rờ'], 'S': ['ét'], 'T': ['tê'], 'U': ['u'], 'V': ['vê'],
    'X': ['ích'], 'Y': ['y'], 'Z': ['dét'],
}
# Chu cai ngoai ban do van dua vao grammar de bat truong hop "khu vuc khong ton tai".
# Bo cac chu de nham voi chu co that (vd "ca" ~ "vực a") va chu ngoai tu vung model (dét).
EXTRA_LETTERS = ['Q', 'X']
# Ten phong khong co tren ban do (de ASR nghe ra thay vi ep thanh ten gan nhat)
EXTRA_PLACES = ['nhà bếp', 'phòng ngủ', 'phòng khách', 'nhà vệ sinh', 'phòng tắm', 'sân',
                'ban công', 'phòng giám đốc', 'thang máy']

FUZZY_RATIO = 0.75

ZONE_PREFIXES = [['khu', 'vực'], ['vùng'], ['phòng']]
START_WORDS = [['từ'], ['xuất', 'phát', 'từ']]
VIA_WORDS = [['đi', 'qua'], ['ghé', 'qua'], ['ngang', 'qua'], ['thông', 'qua'], ['qua'], ['ghé']]
DEST_WORDS = [['đến'], ['tới'], ['về'], ['sang'], ['vào']]
CONNECT_WORDS = [['rồi'], ['sau', 'đó'], ['tiếp', 'theo'], ['và'], ['xong']]
MOVE_WORDS = [['di', 'chuyển'], ['đi'], ['chạy'], ['hãy'], ['robot'], ['đưa'], ['tôi']]
CANCEL_WORDS = [['dừng', 'lại'], ['dừng'], ['hủy'], ['huỷ'], ['thôi'], ['đứng', 'lại']]
FILLER_WORDS = ['nhé', 'ngay', 'luôn', 'giúp', 'với', 'nào', 'cho', 'mình', 'ạ', 'đi']


def fold(text):
    """Chu thuong, bo dau tieng Viet (đ -> d), bo dau cau."""
    text = unicodedata.normalize('NFD', text.lower().replace('đ', 'd').replace('Đ', 'd'))
    text = ''.join(c for c in text if unicodedata.category(c) != 'Mn')
    return re.sub(r'[^a-z0-9\[\] ]+', ' ', text)


def raw_tokens(text):
    """Token goc (chu thuong, giu dau); '[unk]' cua Vosk giu nguyen."""
    return re.findall(r'\[unk\]|\w+', text.lower())


def tokens(text):
    return ['unk' if t == '[unk]' else fold(t).strip() for t in raw_tokens(text)]


def _folded(phrases):
    return [tuple(tokens(' '.join(p) if isinstance(p, list) else p)) for p in phrases]


_START, _VIA, _DEST = _folded(START_WORDS), _folded(VIA_WORDS), _folded(DEST_WORDS)
_CONNECT, _MOVE = _folded(CONNECT_WORDS), _folded(MOVE_WORDS)
_CANCEL, _PREFIX = _folded(CANCEL_WORDS), _folded(ZONE_PREFIXES)
_FILLER = {fold(w).strip() for w in FILLER_WORDS}
_LETTER_OF = {fold(n).strip(): letter for letter, names in LETTER_NAMES.items() for n in names}
# chu cai viet truc tiep ("khu vuc b") cung duoc chap nhan
_LETTER_OF.update({letter.lower(): letter for letter in LETTER_NAMES})


@dataclass
class Mention:
    role: str            # 'start' | 'via' | 'dest'
    text: str            # cum tu goc trong cau
    letter: str = None   # chu cai khu vuc neu nhan ra, None neu khong ton tai


@dataclass
class ParsedCommand:
    text: str
    intent: str = 'none'                         # 'navigate' | 'cancel' | 'none'
    route: list = field(default_factory=list)    # chu cai theo thu tu di
    start: str = None                            # khu vuc xuat phat ("tu A ...")
    unknown: list = field(default_factory=list)  # cum tu chi khu vuc khong ton tai
    mentions: list = field(default_factory=list)


def _match(toks, i, phrases):
    """Cum dai nhat trong phrases khop toks[i:], tra ve do dai (0 neu khong khop)."""
    best = 0
    for p in phrases:
        if len(p) > best and tuple(toks[i:i + len(p)]) == p:
            best = len(p)
    return best


class CommandParser:
    def __init__(self, zones):
        """zones: iterable co .letter, .name, .aliases (vd tb4_voice.zones.Zone)."""
        self.letters = {z.letter.upper() for z in zones}
        self.names = []   # (token tuple, letter), sap xep dai truoc
        for z in zones:
            for phrase in [z.name] + list(z.aliases):
                self.names.append((tuple(tokens(phrase)), z.letter.upper()))
        self.names.sort(key=lambda item: -len(item[0]))
        self.extra_places = [tuple(tokens(p)) for p in EXTRA_PLACES]
        self._keywords = _START + _VIA + _DEST + _CONNECT

    def _name_at(self, toks, i):
        for phrase, letter in self.names:
            if tuple(toks[i:i + len(phrase)]) == phrase:
                return len(phrase), letter
        # So khop mo: ten >= 2 am tiet, sai dung 1 am tiet gan giong (ASR: "thu vien" ->
        # "tu vien"). Nguong 0.75 de "phong hop" (0.67) khong bi hieu thanh "phong hoc".
        for phrase, letter in self.names:
            window = toks[i:i + len(phrase)]
            if len(phrase) < 2 or len(window) < len(phrase):
                continue
            diff = [(a, b) for a, b in zip(window, phrase) if a != b]
            if len(diff) == 1 and SequenceMatcher(None, *diff[0]).ratio() >= FUZZY_RATIO:
                return len(phrase), letter
        return 0, None

    def _is_boundary(self, toks, i):
        return i >= len(toks) or _match(toks, i, self._keywords) > 0

    def parse(self, text):
        cmd = ParsedCommand(text=text)
        raw = raw_tokens(text)
        toks = tokens(text)
        role, after_keyword = 'dest', False
        i = 0
        while i < len(toks):
            # 1. Ten khu vuc co nghia ("phong thi nghiem", "hanh lang") - uu tien cum dai
            n, letter = self._name_at(toks, i)
            if n:
                # "hanh lang C" / "phong hoc A": chu cai di kem phai trung ten, bo qua no
                if i + n < len(toks) and _LETTER_OF.get(toks[i + n]) == letter:
                    n += 1
                cmd.mentions.append(Mention(role, ' '.join(raw[i:i + n]), letter))
                i, after_keyword = i + n, False
                continue
            # 2. Tu khoa vai tro (tu / qua / den) va tu noi (roi, sau do)
            for words, new_role in ((_START, 'start'), (_VIA, 'via'), (_DEST, 'dest')):
                n = _match(toks, i, words)
                if n:
                    role, after_keyword = new_role, True
                    break
            if n:
                i += n
                continue
            n = _match(toks, i, _CONNECT)
            if n:
                i += n
                continue
            # dong tu di chuyen ("di", "di chuyen") cung mo vi tri dich: ASR hay nuot mat
            # "den" ("di den nha bep" -> "di di nha bep")
            n = _match(toks, i, _MOVE)
            if n:
                if not cmd.mentions:
                    role = 'dest'
                i, after_keyword = i + n, True
                continue
            n = _match(toks, i, _CANCEL)
            if n:
                if cmd.intent == 'none':
                    cmd.intent = 'cancel'
                i += n
                continue
            # 3. Tien to + chu cai: "khu vuc A", "phong B", "vung xe"
            n = _match(toks, i, _PREFIX)
            if n:
                j = i + n
                if j < len(toks) and toks[j] in _LETTER_OF:
                    self._add_letter(cmd, role, raw[i:j + 1], _LETTER_OF[toks[j]])
                    i, after_keyword = j + 1, False
                    continue
                # tien to + tu khong phai ten khu vuc -> khu vuc khong ton tai
                k = j
                while not self._is_boundary(toks, k) and toks[k] not in _FILLER and k - j < 3:
                    k += 1
                if k > j:
                    cmd.mentions.append(Mention(role, ' '.join(raw[i:k])))
                    i, after_keyword = k, False
                    continue
                # "khu vuc" khong co ten theo sau (ASR nuot mat "a" trong "vuc a") -> bao
                # khong ro thay vi bo qua am tham va di sai lo trinh
                cmd.mentions.append(Mention(role, ' '.join(raw[i:j]) + ' [unk]'))
                i, after_keyword = j, False
                continue
            # 4. Chu cai dung ngay sau tu khoa: "di den B"
            if after_keyword and toks[i] in _LETTER_OF:
                self._add_letter(cmd, role, raw[i:i + 1], _LETTER_OF[toks[i]])
                i, after_keyword = i + 1, False
                continue
            # 5. Cum tu la sau tu khoa (vd "nha bep", "[unk]") -> khu vuc khong ton tai
            if after_keyword and toks[i] not in _FILLER:
                k = i
                while not self._is_boundary(toks, k) and toks[k] not in _FILLER and k - i < 4:
                    k += 1
                cmd.mentions.append(Mention(role, ' '.join(raw[i:k])))
                i, after_keyword = k, False
                continue
            i += 1   # tu khong lien quan

        cmd.unknown = [m.text.replace('[unk]', '(không rõ)') for m in cmd.mentions
                       if m.letter is None]
        known = [m for m in cmd.mentions if m.letter]
        if cmd.mentions and cmd.intent != 'cancel':
            cmd.intent = 'navigate'
        starts = [m for m in known if m.role == 'start']
        rest = [m for m in known if m.role != 'start']
        # "den B qua C" -> qua C truoc roi moi den B
        if len(rest) > 1 and rest[0].role == 'dest' and all(m.role == 'via' for m in rest[1:]):
            rest = rest[1:] + rest[:1]
        route = []
        for m in starts[:1] + rest:
            if not route or route[-1] != m.letter:
                route.append(m.letter)
        cmd.route = route
        cmd.start = starts[0].letter if starts else None
        return cmd

    def _add_letter(self, cmd, role, toks, letter):
        text = ' '.join(toks)
        cmd.mentions.append(Mention(role, text, letter if letter in self.letters else None))


def zone_expressions(zones):
    """Cac cach goi mot khu vuc: ten, alias, "khu vuc <chu cai>", "phong <chu cai>"..."""
    out = []
    letters = sorted({z.letter.upper() for z in zones} | set(EXTRA_LETTERS))
    for letter in letters:
        # chi cach doc chuan: "gờ"/"hờ"/"bờ" de nham voi nhau khi dua vao grammar
        spoken = LETTER_NAMES[letter][0]
        out += [f'{" ".join(prefix)} {spoken}' for prefix in ZONE_PREFIXES]
    for z in zones:
        out += [z.name.lower()] + [a.lower() for a in z.aliases]
    return out + EXTRA_PLACES


def grammar_phrases(zones):
    """Cum mau cho Vosk grammar (tu khoa + khu vuc). Vosk dung chung lam mo hinh ngon ngu
    nen cac cum van noi duoc voi nhau ("tu A qua C roi den B")."""
    places = zone_expressions(zones)
    phrases = ['dừng lại', 'hủy', 'đứng lại', 'robot hãy', 'xuất phát từ']
    for place in places:
        phrases += [f'đi đến {place}', f'đi tới {place}', f'di chuyển đến {place}',
                    f'di chuyển từ {place}', f'đi từ {place}', f'từ {place}', f'qua {place}',
                    f'đi qua {place}', f'rồi đến {place}', f'sau đó đến {place}',
                    f'về {place}', f'đến {place}']
    return phrases


def grammar_words(zones):
    """Tap tu xuat hien trong grammar (de kiem tra/hien thi)."""
    words = []
    for p in grammar_phrases(zones):
        for w in p.split():
            if w not in words:
                words.append(w)
    return words
