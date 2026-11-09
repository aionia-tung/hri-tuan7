"""Kiem thu command_parser, zones va map_segmentation (khong can ROS)."""

from pathlib import Path

import pytest

from tb4_voice.command_parser import CommandParser, fold, grammar_words
from tb4_voice.zones import ZoneMap

PKG = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def zone_map():
    return ZoneMap.load(PKG / 'maps' / 'zones.yaml')


@pytest.fixture(scope='module')
def parser(zone_map):
    return CommandParser(zone_map.zones)


@pytest.mark.parametrize('text, route, start', [
    # 3 cau vi du trong de bai
    ('Đi đến khu vực A.', ['A'], None),
    ('Di chuyển từ khu vực A đến khu vực B.', ['A', 'B'], 'A'),
    ('Đi đến phòng thí nghiệm.', ['G'], None),
    # bien the
    ('di den khu vuc c', ['C'], None),                      # go khong dau
    ('đi tới khu vực bê', ['B'], None),                     # chu cai doc theo tieng Viet
    ('đi đến hành lang', ['C'], None),
    ('robot hãy đi tới phòng học nhé', ['A'], None),
    ('đến B', ['B'], None),
    ('về sảnh chính', ['E'], None),
    ('đi từ phòng A qua hành lang C rồi đến B', ['A', 'C', 'B'], 'A'),
    ('đi từ thư viện qua căng tin rồi đến phòng y tế', ['F', 'H', 'I'], 'F'),
    ('đến phòng thí nghiệm từ khu vực đê', ['D', 'G'], 'D'),
    ('đi đến phòng kho qua hành lang', ['C', 'B'], None),
    ('đi đến khu vực A rồi đến khu vực A', ['A'], None),    # gop lap lien tiep
    # so khop mo loi ASR that (eval_asr): "thu vien" -> "tu vien"
    ('đi từ phòng học qua hành lang rồi đến từ viện', ['A', 'C', 'F'], 'A'),
])
def test_routes(parser, text, route, start):
    cmd = parser.parse(text)
    assert cmd.intent == 'navigate'
    assert cmd.unknown == []
    assert cmd.route == route
    assert cmd.start == start


@pytest.mark.parametrize('text, unknown', [
    ('đi đến nhà bếp', ['nhà bếp']),
    ('đi đến khu vực dét', ['khu vực dét']),
    ('đi đến khu vực Z', ['khu vực z']),
    ('đi đến khu vực ích', ['khu vực ích']),
    ('di chuyển từ khu vực A đến phòng ngủ', ['phòng ngủ']),
    ('đi đến [unk]', ['(không rõ)']),
    ('di di nhà bếp', ['nhà bếp']),
    ('đi đến phòng họp', ['phòng họp']),
    ('di chuyển từ khu vực đến khu vực bê', ['khu vực (không rõ)']),   # ASR mat chu "a"   # khong bi so khop mo thanh "phong hoc"       # ASR nghe "đến" thanh "đi"
])
def test_unknown_zone(parser, text, unknown):
    cmd = parser.parse(text)
    assert cmd.intent == 'navigate'
    assert cmd.unknown == unknown


@pytest.mark.parametrize('text', ['dừng lại', 'hủy', 'Robot dừng lại ngay'])
def test_cancel(parser, text):
    assert parser.parse(text).intent == 'cancel'


@pytest.mark.parametrize('text', ['', 'xin chào', 'hôm nay trời đẹp', 'robot hãy đi'])
def test_no_command(parser, text):
    cmd = parser.parse(text)
    assert cmd.intent == 'none' and cmd.route == []


def test_fold():
    assert fold('Phòng Thí Nghiệm, Đi!') == 'phong thi nghiem  di '


def test_grammar_has_zone_words(zone_map):
    words = grammar_words(zone_map.zones)
    for w in ['đi', 'đến', 'khu', 'vực', 'thí', 'nghiệm', 'bê', 'xê', 'bếp', 'ích']:
        assert w in words
    assert 'ca' not in words   # K de nham voi "vuc a"


def test_zone_lookup(zone_map):
    assert len(zone_map.zones) >= 3
    for z in zone_map.zones:
        # diem dich nam trong chinh khu vuc do
        assert zone_map.zone_at(*z.goal[:2]).letter == z.letter
    assert zone_map.zone_at(0.0, 0.0).letter == 'E'   # dock nam o sanh chinh
    assert zone_map.zone_at(50.0, 50.0) is None


def test_segmentation_reproducible(tmp_path):
    from tb4_voice.segment_map import main
    main(['--map', str(PKG / 'maps' / 'map.yaml'),
          '--semantics', str(PKG / 'config' / 'zone_semantics.yaml'), '--out', str(tmp_path)])
    fresh = ZoneMap.load(tmp_path / 'zones.yaml')
    saved = ZoneMap.load(PKG / 'maps' / 'zones.yaml')
    assert [(z.letter, z.goal) for z in fresh.zones] == [(z.letter, z.goal) for z in saved.zones]
