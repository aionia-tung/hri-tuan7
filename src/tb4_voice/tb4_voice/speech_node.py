"""Node nhan dang giong noi (Vosk tieng Viet) -> /voice/text (std_msgs/String).

Nguon audio (tham so source):
  mic    doc micro lien tuc qua arecord, Vosk tu cat cau theo khoang lang
  files  lan luot nhan dang cac file trong wav_files (demo/kiem thu lap lai duoc); file sau
         chi phat khi robot xong lenh truoc (/voice_nav/busy = false)
Luon co them topic /voice/wav (String = duong dan file audio) de nhan dang 1 file theo yeu cau.

Ket qua co do tin cay trung binh < min_confidence bi bo qua (tranh nhieu nen thanh lenh).
"""

import threading
from pathlib import Path

import rclpy
from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from std_msgs.msg import Bool, String

from tb4_voice.command_parser import grammar_phrases
from tb4_voice.speech import DEFAULT_MODEL, SpeechRecognizer, open_microphone
from tb4_voice.zones import ZoneMap


class SpeechNode(Node):

    def __init__(self):
        super().__init__('speech_recognition')
        share = Path(get_package_share_directory('tb4_voice'))
        self.declare_parameter('model_path', str(DEFAULT_MODEL))
        self.declare_parameter('zones_file', str(share / 'maps' / 'zones.yaml'))
        self.declare_parameter('use_grammar', True)
        self.declare_parameter('source', 'mic')            # mic | files
        self.declare_parameter('device', 'default')         # thiet bi ALSA cho arecord
        self.declare_parameter('wav_files', [''])
        self.declare_parameter('file_interval_sec', 8.0)    # nghi toi thieu giua cac file
        self.declare_parameter('file_timeout_sec', 900.0)   # doi robot toi da
        self.declare_parameter('min_confidence', 0.55)
        self.declare_parameter('text_topic', '/voice/text')
        p = lambda name: self.get_parameter(name).value  # noqa: E731

        grammar = None
        if p('use_grammar'):
            grammar = grammar_phrases(ZoneMap.load(p('zones_file')).zones)
        self.rec = SpeechRecognizer(p('model_path'), grammar=grammar)
        self.lock = threading.Lock()
        self.min_conf = float(p('min_confidence'))
        self.pub = self.create_publisher(String, p('text_topic'), 10)
        self.raw_pub = self.create_publisher(String, '/voice/raw', 10)
        self.create_subscription(String, '/voice/wav', self.on_wav, 10)
        self.nav_idle = threading.Event()
        self.nav_idle.set()
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(Bool, '/voice_nav/busy', self.on_busy, latched)
        mode = f'grammar {len(grammar)} cum' if grammar else 'tu do'
        self.get_logger().info(f'Vosk {Path(p("model_path")).name} ({mode}), nguon = {p("source")}')

        self.running = True
        if p('source') == 'mic':
            self.thread = threading.Thread(target=self.mic_loop, args=(p('device'),), daemon=True)
        else:
            files = [f for f in p('wav_files') if f]
            self.thread = threading.Thread(
                target=self.file_loop, daemon=True,
                args=(files, float(p('file_interval_sec')), float(p('file_timeout_sec'))))
        self.thread.start()

    def emit(self, result, source):
        if not result:
            return
        text, conf = result
        self.raw_pub.publish(String(data=f'{text} | conf={conf:.2f} | {source}'))
        if text.replace('[unk]', '').strip() == '':
            return
        if conf < self.min_conf:
            self.get_logger().info(f'Bo qua "{text}" (conf {conf:.2f} < {self.min_conf})')
            return
        self.get_logger().info(f'Nghe duoc: "{text}" (conf {conf:.2f}, {source})')
        self.pub.publish(String(data=text))

    def mic_loop(self, device):
        proc = open_microphone(device)
        self.get_logger().info('Dang nghe micro... hay noi vi du "đi đến khu vực A"')
        try:
            while self.running and rclpy.ok():
                chunk = proc.stdout.read(4000)   # 0.125 s
                if not chunk:
                    self.get_logger().error('arecord dung - kiem tra micro (arecord -l)')
                    break
                with self.lock:
                    self.emit(self.rec.accept(chunk), 'mic')
        finally:
            proc.terminate()

    def transcribe(self, path):
        with self.lock:
            results = self.rec.transcribe_file(path)
        for r in results:
            self.emit(r, Path(path).name)

    def on_busy(self, msg):
        if msg.data:
            self.nav_idle.clear()
        else:
            self.nav_idle.set()

    def file_loop(self, files, interval, timeout):
        threading.Event().wait(3.0)   # doi voice_nav khoi dong
        for i, f in enumerate(files):
            if not self.running:
                return
            self.get_logger().info(f'[{i + 1}/{len(files)}] Phat file {Path(f).name}')
            self.transcribe(f)
            threading.Event().wait(interval)
            if not self.nav_idle.wait(timeout):
                self.get_logger().warn('Robot chua xong lenh truoc - van phat file tiep')
        self.get_logger().info('Da phat het file demo')

    def on_wav(self, msg):
        threading.Thread(target=self.transcribe, args=(msg.data,), daemon=True).start()

    def destroy_node(self):
        self.running = False
        super().destroy_node()


def main():
    rclpy.init()
    node = SpeechNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
