"""Node dieu huong bang cau lenh giong noi: van ban -> lo trinh khu vuc -> Nav2 NavigateToPose.

Luong xu ly:
  /voice/text (String) -> CommandParser -> lo trinh [A, C, B]
    - khu vuc khong ton tai / khong hieu lenh -> bao loi tren /voice_nav/status, robot dung yen
    - "tu A den B" khi robot dang o vi tri khac -> di qua A truoc; da o trong A -> bo qua A
    - "dung lai" / "huy" -> huy goal hien tai
    - lenh moi khi dang chay -> huy lo trinh cu, thay bang lo trinh moi
  Robot dang dock -> undock (Create 3) truoc khi gui goal dau tien.
  Truoc khi chay: ComputePathToPose tung chang -> /voice_nav/planned_route (duong du kien).

Hien thi RViz:
  /voice_nav/zone_map  OccupancyGrid to mau theo khu vuc (color scheme "costmap")
  /voice_nav/zones     MarkerArray: vien, ten, diem dich tung khu vuc + lo trinh dang chay
  /voice_nav/busy      Bool, true khi con lo trinh dang chay
"""

import math
import subprocess
import unicodedata
from pathlib import Path

import numpy as np
import rclpy
from action_msgs.msg import GoalStatus
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import Point, PoseStamped, PoseWithCovarianceStamped
from irobot_create_msgs.action import Undock
from irobot_create_msgs.msg import DockStatus
from nav2_msgs.action import ComputePathToPose, NavigateToPose
from nav_msgs.msg import OccupancyGrid, Path as PathMsg
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, qos_profile_sensor_data
from std_msgs.msg import Bool, ColorRGBA, String
from tf2_ros import Buffer, TransformException, TransformListener
from visualization_msgs.msg import Marker, MarkerArray

from tb4_voice.command_parser import CommandParser
from tb4_voice.zones import ZoneMap

PALETTE = [(141, 211, 199), (255, 237, 111), (190, 186, 218), (251, 128, 114),
           (128, 177, 211), (253, 180, 98), (179, 222, 105), (252, 205, 229),
           (188, 128, 189), (204, 235, 197), (217, 217, 217), (255, 255, 179)]
STATUS_TEXT = {GoalStatus.STATUS_SUCCEEDED: 'SUCCEEDED', GoalStatus.STATUS_ABORTED: 'ABORTED',
               GoalStatus.STATUS_CANCELED: 'CANCELED'}


def ascii_text(text):
    """RViz2 khong ve duoc dau tieng Viet -> bo dau cho nhan marker."""
    text = unicodedata.normalize('NFD', text.replace('đ', 'd').replace('Đ', 'D'))
    return ''.join(c for c in text if unicodedata.category(c) != 'Mn')


def yaw_to_quat(yaw):
    return 0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)


class VoiceNavNode(Node):

    def __init__(self):
        super().__init__('voice_nav')
        default_zones = Path(get_package_share_directory('tb4_voice')) / 'maps' / 'zones.yaml'
        self.declare_parameter('zones_file', str(default_zones))
        self.declare_parameter('text_topic', '/voice/text')
        self.declare_parameter('undock_first', True)
        self.declare_parameter('goal_timeout_sec', 300.0)
        self.declare_parameter('max_retries', 1)
        self.declare_parameter('dwell_sec', 2.0)          # dung o diem trung gian
        self.declare_parameter('preview_path', True)
        self.declare_parameter('speak', False)            # doc phan hoi bang spd-say
        p = lambda name: self.get_parameter(name).value  # noqa: E731

        self.zone_map = ZoneMap.load(p('zones_file'))
        self.parser = CommandParser(self.zone_map.zones)
        self.undock_first = p('undock_first')
        self.timeout = float(p('goal_timeout_sec'))
        self.max_retries = int(p('max_retries'))
        self.dwell = float(p('dwell_sec'))
        self.preview = p('preview_path')
        self.speak_enabled = p('speak')

        self.nav = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self.planner = ActionClient(self, ComputePathToPose, 'compute_path_to_pose')
        self.undock_client = ActionClient(self, Undock, 'undock')
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.status_pub = self.create_publisher(String, '/voice_nav/status', 10)
        self.marker_pub = self.create_publisher(MarkerArray, '/voice_nav/zones', latched)
        self.grid_pub = self.create_publisher(OccupancyGrid, '/voice_nav/zone_map', latched)
        self.route_pub = self.create_publisher(PathMsg, '/voice_nav/planned_route', latched)
        self.busy_pub = self.create_publisher(Bool, '/voice_nav/busy', latched)
        self.create_subscription(String, p('text_topic'), self.on_text, 10)
        self.docked = None
        # pose AMCL du phong khi TF chua co (node vua khoi dong); amcl_pose la transient local
        self.amcl_pose = None
        self.create_subscription(PoseWithCovarianceStamped, 'amcl_pose', self.on_amcl,
                                 QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self.create_subscription(DockStatus, 'dock_status', self.on_dock, qos_profile_sensor_data)

        # Trang thai lo trinh
        self.generation = 0        # tang moi lenh moi -> bo qua callback cua lenh cu
        self.legs = []             # cac Zone con phai di
        self.current = None        # Zone dang di toi
        self.goal_handle = None
        self.goal_start = None
        self.retries = 0
        self.last_feedback_log = 0.0
        self.create_timer(1.0, self.check_timeout)

        self.publish_zone_map()
        self.publish_markers()
        names = ', '.join(z.label for z in self.zone_map.zones)
        self.get_logger().info(f'Da nap {len(self.zone_map.zones)} khu vuc: {names}')
        self.say(f'San sang. Cac khu vuc: {names}', speak=False)

    # ------------------------------------------------------------------ phan hoi
    def say(self, text, speak=True, error=False):
        # rclpy cam doi muc log tai cung mot dong goi -> tach 2 nhanh
        if error:
            self.get_logger().warn(text)
        else:
            self.get_logger().info(text)
        self.status_pub.publish(String(data=text))
        if speak and self.speak_enabled:
            subprocess.Popen(['spd-say', '-l', 'vi', '-r', '-20', ascii_text(text)],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # ------------------------------------------------------------------ cau lenh
    def on_dock(self, msg):
        self.docked = msg.is_docked

    def on_text(self, msg):
        text = msg.data.strip()
        if not text:
            return
        cmd = self.parser.parse(text)
        self.get_logger().info(f'Lenh: "{text}" -> intent={cmd.intent} route={cmd.route} '
                               f'start={cmd.start} unknown={cmd.unknown}')
        if cmd.intent == 'cancel':
            self.cancel_all('Đã hủy lệnh, robot dừng lại.')
            return
        if cmd.intent == 'none':
            self.say(f'Không hiểu lệnh "{text}". Hãy nói ví dụ: đi đến khu vực A.', error=True)
            return
        if cmd.unknown:
            valid = ', '.join(z.label for z in self.zone_map.zones)
            self.say(f'Không tìm thấy khu vực "{", ".join(cmd.unknown)}" trên bản đồ. '
                     f'Các khu vực hợp lệ: {valid}.', error=True)
            return
        if not cmd.route:
            self.say('Không xác định được khu vực đích.', error=True)
            return

        legs = [self.zone_map.by_letter[letter] for letter in cmd.route]
        here = self.robot_zone()
        if cmd.start and len(legs) > 1 and here is not None and here.letter == cmd.start:
            self.get_logger().info(f'Robot dang o {here.label} -> bo qua diem xuat phat')
            legs = legs[1:]
        if self.current is not None or self.legs:
            self.cancel_all(None)   # lenh moi thay lenh cu
        self.generation += 1
        self.legs = legs
        self.retries = 0
        route_txt = ' -> '.join(z.label for z in legs)
        where = f' (robot đang ở {here.label})' if here else ''
        self.say(f'Đã nhận lệnh: đi {route_txt}{where}.')
        self.publish_markers()
        if self.preview:
            self.preview_route(self.generation, list(legs))
        self.start_next(self.generation)

    def cancel_all(self, message):
        self.generation += 1
        self.legs = []
        self.current = None
        if self.goal_handle is not None:
            self.goal_handle.cancel_goal_async()
            self.goal_handle = None
        self.route_pub.publish(PathMsg(header=self.header()))
        self.publish_markers()
        if message:
            self.say(message)

    # ------------------------------------------------------------------ vi tri robot
    def on_amcl(self, msg):
        self.amcl_pose = msg.pose.pose

    def robot_pose(self):
        try:
            t = self.tf_buffer.lookup_transform('map', 'base_link', rclpy.time.Time())
            pos, q = t.transform.translation, t.transform.rotation
        except TransformException:
            if self.amcl_pose is None:
                return None
            pos, q = self.amcl_pose.position, self.amcl_pose.orientation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        return pos.x, pos.y, yaw

    def robot_zone(self):
        pose = self.robot_pose()
        return self.zone_map.zone_at(pose[0], pose[1]) if pose else None

    def header(self):
        msg = PoseStamped().header
        msg.frame_id = 'map'
        msg.stamp = self.get_clock().now().to_msg()
        return msg

    def zone_pose(self, zone, from_xy=None):
        """Pose dich cua khu vuc; huong = huong di toi (tranh xoay tai cho o cuoi)."""
        x, y, yaw_deg = zone.goal
        yaw = math.radians(yaw_deg)
        if from_xy is not None and math.hypot(x - from_xy[0], y - from_xy[1]) > 0.3:
            yaw = math.atan2(y - from_xy[1], x - from_xy[0])
        pose = PoseStamped(header=self.header())
        pose.pose.position.x, pose.pose.position.y = float(x), float(y)
        o = pose.pose.orientation
        o.x, o.y, o.z, o.w = yaw_to_quat(yaw)
        return pose

    # ------------------------------------------------------------------ chuoi goal
    def start_next(self, gen):
        if gen != self.generation:
            return
        if self.undock_first and self.docked:
            if not self.undock_client.wait_for_server(timeout_sec=2.0):
                self.get_logger().warn('Khong co action /undock - bo qua undock')
                self.undock_first = False
            else:
                self.say('Robot đang ở trạm sạc, rời dock trước.', speak=False)
                self.undock_first = False
                fut = self.undock_client.send_goal_async(Undock.Goal())
                fut.add_done_callback(lambda f: self.on_undock_accepted(f, gen))
                return
        if not self.legs:
            self.current = None
            self.publish_markers()
            return
        if not self.nav.wait_for_server(timeout_sec=5.0):
            self.say('Nav2 chưa sẵn sàng (không có action navigate_to_pose).', error=True)
            self.legs, self.current = [], None
            return
        self.current = self.legs.pop(0)
        pose = self.robot_pose()
        target = self.zone_pose(self.current, pose[:2] if pose else None)
        goal = NavigateToPose.Goal(pose=target)
        self.say(f'Đang di chuyển đến {self.current.label}.')
        self.publish_markers()
        self.goal_start = self.get_clock().now()
        fut = self.nav.send_goal_async(goal, feedback_callback=self.on_feedback)
        fut.add_done_callback(lambda f: self.on_goal_accepted(f, gen))

    def on_undock_accepted(self, future, gen):
        handle = future.result()
        if not handle.accepted:
            self.start_next(gen)
            return
        handle.get_result_async().add_done_callback(lambda _: self.start_next(gen))

    def on_goal_accepted(self, future, gen):
        handle = future.result()
        if gen != self.generation:
            if handle.accepted:
                handle.cancel_goal_async()
            return
        if not handle.accepted:
            self.say(f'Nav2 từ chối mục tiêu {self.current.label}.', error=True)
            self.legs, self.current = [], None
            return
        self.goal_handle = handle
        handle.get_result_async().add_done_callback(lambda f: self.on_result(f, gen))

    def on_feedback(self, msg):
        now = self.get_clock().now().nanoseconds * 1e-9
        if now - self.last_feedback_log > 5.0 and self.current is not None:
            self.last_feedback_log = now
            fb = msg.feedback
            self.get_logger().info(f'-> {self.current.letter}: con {fb.distance_remaining:.1f} m, '
                                   f'{fb.number_of_recoveries} recovery')

    def on_result(self, future, gen):
        if gen != self.generation:
            return
        status = future.result().status
        self.goal_handle = None
        zone = self.current
        if status == GoalStatus.STATUS_SUCCEEDED:
            here = self.robot_zone()
            check = '' if here is None or here.letter == zone.letter else \
                f' (AMCL đặt robot ở {here.label})'
            if self.legs:
                self.say(f'Đã đi qua {zone.label}{check}.')
                self.create_oneshot(self.dwell, lambda: self.start_next(gen))
            else:
                self.say(f'Đã đến {zone.label}{check}.')
                self.current = None
                self.publish_markers()
            self.retries = 0
            return
        if status == GoalStatus.STATUS_CANCELED:
            return
        if self.retries < self.max_retries:
            self.retries += 1
            self.get_logger().warn(f'Goal {zone.letter} {STATUS_TEXT.get(status, status)} - thu lai')
            self.legs.insert(0, zone)
            self.start_next(gen)
            return
        self.say(f'Không thể đến {zone.label} ({STATUS_TEXT.get(status, status)}).', error=True)
        self.legs, self.current = [], None
        self.publish_markers()

    def create_oneshot(self, delay, callback):
        timer = None

        def fire():
            timer.cancel()
            callback()
        timer = self.create_timer(max(delay, 0.01), fire)

    def check_timeout(self):
        if self.goal_handle is None or self.goal_start is None:
            return
        elapsed = (self.get_clock().now() - self.goal_start).nanoseconds * 1e-9
        if elapsed > self.timeout:
            self.say(f'Quá thời gian đến {self.current.label}, hủy lệnh.', error=True)
            self.cancel_all(None)

    # ------------------------------------------------------------------ duong du kien
    def preview_route(self, gen, legs):
        """Goi planner cho tung chang (robot -> A -> C -> B), ghep thanh mot Path."""
        if not self.planner.wait_for_server(timeout_sec=1.0):
            return
        segments = []
        pose = self.robot_pose()
        prev_xy = pose[:2] if pose else None

        def request(i, prev_xy):
            if gen != self.generation:
                return
            if i == len(legs):
                path = PathMsg(header=self.header())
                for seg in segments:
                    path.poses.extend(seg)
                self.route_pub.publish(path)
                length = sum(math.dist((a.pose.position.x, a.pose.position.y),
                                       (b.pose.position.x, b.pose.position.y))
                             for a, b in zip(path.poses, path.poses[1:]))
                self.get_logger().info(f'Duong du kien {len(legs)} chang, dai {length:.1f} m')
                return
            goal = ComputePathToPose.Goal(goal=self.zone_pose(legs[i], prev_xy))
            if i > 0:
                goal.use_start = True
                goal.start = self.zone_pose(legs[i - 1])
            fut = self.planner.send_goal_async(goal)

            def accepted(f):
                h = f.result()
                if not h.accepted:
                    return
                h.get_result_async().add_done_callback(done)

            def done(f):
                result = f.result()
                if result.status == GoalStatus.STATUS_SUCCEEDED:
                    segments.append(list(result.result.path.poses))
                request(i + 1, legs[i].goal[:2])
            fut.add_done_callback(accepted)

        request(0, prev_xy)

    # ------------------------------------------------------------------ RViz
    def publish_zone_map(self):
        labels = self.zone_map.labels
        if labels is None:
            return
        grid = OccupancyGrid()
        grid.header = self.header()
        grid.info.resolution = float(self.zone_map.resolution)
        grid.info.height, grid.info.width = labels.shape[:2]
        grid.info.origin.position.x = float(self.zone_map.origin[0])
        grid.info.origin.position.y = float(self.zone_map.origin[1])
        grid.info.origin.orientation.w = 1.0
        n = max(len(self.zone_map.zones), 1)
        # moi khu vuc mot muc 5..95 de color scheme costmap to mau khac nhau; 0 = trong suot
        values = np.zeros(int(labels.max()) + 1, np.int8)
        for i, z in enumerate(self.zone_map.zones):
            values[z.id] = int(5 + 90 * i / max(n - 1, 1))
        grid.data = values[np.flipud(labels)].flatten().tolist()
        self.grid_pub.publish(grid)

    def publish_markers(self):
        markers = MarkerArray()
        markers.markers.append(Marker(action=Marker.DELETEALL))
        header = self.header()
        active = self.current.letter if self.current else None
        pending = [z.letter for z in self.legs]
        for idx, z in enumerate(self.zone_map.zones):
            r, g, b = (c / 255.0 for c in PALETTE[(z.id - 1) % len(PALETTE)])
            hot = z.letter == active
            outline = Marker(header=header, ns='zone_outline', id=z.id, type=Marker.LINE_STRIP)
            outline.scale.x = 0.12 if hot else 0.05
            outline.color = ColorRGBA(r=0.1, g=0.8, b=0.1, a=1.0) if hot else \
                ColorRGBA(r=r * 0.7, g=g * 0.7, b=b * 0.7, a=0.9)
            outline.pose.orientation.w = 1.0
            outline.points = [Point(x=float(x), y=float(y), z=0.05) for x, y in z.polygon]
            if outline.points:
                outline.points.append(outline.points[0])
            label = Marker(header=header, ns='zone_label', id=z.id, type=Marker.TEXT_VIEW_FACING)
            label.pose.position.x, label.pose.position.y = float(z.goal[0]), float(z.goal[1]) + 0.6
            label.pose.position.z = 0.5
            label.pose.orientation.w = 1.0
            label.scale.z = 0.55
            label.color = ColorRGBA(r=0.05, g=0.05, b=0.05, a=1.0)
            label.text = ascii_text(z.label)
            goal = Marker(header=header, ns='zone_goal', id=z.id, type=Marker.CYLINDER)
            goal.pose.position.x, goal.pose.position.y = float(z.goal[0]), float(z.goal[1])
            goal.pose.orientation.w = 1.0
            goal.scale.x = goal.scale.y = 0.35
            goal.scale.z = 0.05
            goal.color = ColorRGBA(r=0.1, g=0.8, b=0.1, a=1.0) if hot else \
                ColorRGBA(r=0.9, g=0.2, b=0.1, a=0.8 if z.letter in pending else 0.5)
            markers.markers += [outline, label, goal]
        # lo trinh dang chay: duong thang noi cac diem dich + so thu tu
        route = ([self.current] if self.current else []) + list(self.legs)
        if route:
            line = Marker(header=header, ns='route', id=0, type=Marker.LINE_STRIP)
            line.scale.x = 0.06
            line.color = ColorRGBA(r=1.0, g=0.5, b=0.0, a=0.9)
            line.pose.orientation.w = 1.0
            pose = self.robot_pose()
            pts = ([pose[:2]] if pose else []) + [z.goal[:2] for z in route]
            line.points = [Point(x=float(x), y=float(y), z=0.1) for x, y in pts]
            markers.markers.append(line)
            for k, z in enumerate(route, start=1):
                num = Marker(header=header, ns='route_step', id=k, type=Marker.TEXT_VIEW_FACING)
                num.pose.position.x, num.pose.position.y = float(z.goal[0]), float(z.goal[1])
                num.pose.position.z = 1.0
                num.pose.orientation.w = 1.0
                num.scale.z = 0.6
                num.color = ColorRGBA(r=1.0, g=0.4, b=0.0, a=1.0)
                num.text = f'#{k}'
                markers.markers.append(num)
        self.marker_pub.publish(markers)
        # publish_markers duoc goi moi khi trang thai doi -> cap nhat luon co busy
        self.busy_pub.publish(Bool(data=bool(route)))


def main():
    rclpy.init()
    node = VoiceNavNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
