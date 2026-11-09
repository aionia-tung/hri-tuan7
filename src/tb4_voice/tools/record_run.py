#!/usr/bin/env python3
"""Ghi lai mot lan chay de ve hinh bao cao: quy dao AMCL, duong du kien, thong bao trang thai.

    python3 tools/record_run.py --out docs/run_demo     # Ctrl+C de dung
Tao docs/run_demo/{trajectory.csv, planned_route.json, status.csv}.
"""

import argparse
import csv
import json
import time
from pathlib import Path

import rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped
from nav_msgs.msg import Path as PathMsg
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from std_msgs.msg import String


class Recorder(Node):
    def __init__(self, out):
        super().__init__('run_recorder')
        self.out = Path(out)
        self.out.mkdir(parents=True, exist_ok=True)
        self.t0 = time.time()
        self.traj = csv.writer(open(self.out / 'trajectory.csv', 'w', newline=''))
        self.traj.writerow(['t', 'x', 'y'])
        self.status = csv.writer(open(self.out / 'status.csv', 'w', newline='', encoding='utf-8'))
        self.status.writerow(['t', 'text'])
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(PoseWithCovarianceStamped, 'amcl_pose', self.on_pose, latched)
        self.create_subscription(PathMsg, '/voice_nav/planned_route', self.on_route, latched)
        self.create_subscription(String, '/voice_nav/status', self.on_status, 10)
        self.create_subscription(String, '/voice/text', self.on_text, 10)

    def now(self):
        return round(time.time() - self.t0, 2)

    def on_pose(self, msg):
        p = msg.pose.pose.position
        self.traj.writerow([self.now(), round(p.x, 3), round(p.y, 3)])

    def on_route(self, msg):
        if msg.poses:
            pts = [[round(p.pose.position.x, 3), round(p.pose.position.y, 3)] for p in msg.poses]
            (self.out / 'planned_route.json').write_text(json.dumps(pts))

    def on_status(self, msg):
        self.status.writerow([self.now(), msg.data])
        self.get_logger().info(msg.data)

    def on_text(self, msg):
        self.status.writerow([self.now(), f'[voice] {msg.data}'])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='docs/run_demo')
    args = ap.parse_args()
    rclpy.init()
    node = Recorder(args.out)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    rclpy.try_shutdown()


if __name__ == '__main__':
    main()
