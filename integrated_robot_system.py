import math
import random
import sys
import numpy as np
from collections import deque
import threading
import time

import pygame
from pygame import gfxdraw

# ============ إعدادات عامة ============
MAP_PATH = r"E:\tank puthon\tank.png"
WINDOW_W, WINDOW_H = 900, 900
CONTROL_PANEL_W = 300
TOTAL_W = WINDOW_W + CONTROL_PANEL_W
SCALE = 1
FPS = 30

# روبوت
ROBOT_W, ROBOT_H = 40, 24
ROBOT_SPEED_MIN = 0.1
ROBOT_SPEED_MAX = 5.0
ROBOT_SPEED_DEFAULT = 1.2
ROBOT_ROT_SPEED = 2.5

# حسّاسات مُحسّنة - نفس التوزيع المطلوب
SENSOR_ANGLES = [-110, -90, -70, -20, 0, 20, 70, 90, 110, 180]
SENSOR_MAX_DIST = 250
SENSOR_STEP = 3

# حدود التفكير الذكي
SAFE_DISTANCE = 80
EARLY_TURN_DISTANCE = 120
CRITICAL_DISTANCE = 40
REAR_SAFE_DISTANCE = 50

# خريطة الاحتلال
OCCUPIED = 1
FREE = 0
UNKNOWN = -1

# أوضاع التفاعل
MODE_NORMAL = 0
MODE_ADD_OBSTACLE = 1
MODE_REMOVE_OBSTACLE = 2
MODE_DRAW = 3

# ألوان
BG_COLOR = (245,245,245)
CONTROL_BG = (230,230,230)
ROBOT_COLOR = (20,120,200)
RAY_HIT_COLOR = (200,30,30)
RAY_FREE_COLOR = (30,180,30)
REAR_SENSOR_COLOR = (255,100,255)
GRID_FREE_COLOR = (30,180,30,18)
GRID_OCC_COLOR = (200,30,30,35)
BUTTON_COLOR = (200,200,200)
BUTTON_HOVER = (180,180,180)
BUTTON_ACTIVE = (150,150,250)
BUTTON_DRAW_MODE = (100,200,100)
BUTTON_AI_MODE = (255,165,0)
BUTTON_REAL_ROBOT = (100,200,100)
SLIDER_COLOR = (100,100,100)
SLIDER_HANDLE = (50,50,200)
DRAW_COLOR_LIVE = (50,50,50,180)

# ============ إعدادات الاتصال بالعربة الحقيقية ============
try:
    import serial
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False
    print("⚠️ مكتبة serial غير متوفرة - لن تعمل العربة الحقيقية")

# إعدادات الاتصال
SERIAL_PORT = "COM4"  # عدل حسب جهازك
SERIAL_BAUDRATE = 9600
REAL_ROBOT_ENABLED = False
arduino_connection = None

# ============ فئة الاتصال بالعربة الحقيقية ============
class RealRobotController:
    def __init__(self):
        self.connected = False
        self.port = SERIAL_PORT
        self.baudrate = SERIAL_BAUDRATE
        self.connection = None
        self.last_command = None
        self.command_history = []
        
    def connect(self):
        """الاتصال بالعربة الحقيقية"""
        if not SERIAL_AVAILABLE:
            print("❌ مكتبة serial غير متوفرة")
            return False
            
        try:
            self.connection = serial.Serial(self.port, self.baudrate, timeout=1)
            time.sleep(2)  # انتظار استقرار الاتصال
            self.connected = True
            print(f"✅ تم الاتصال بالعربة الحقيقية على {self.port}")
            return True
        except Exception as e:
            print(f"❌ فشل الاتصال: {e}")
            self.connected = False
            return False
            
    def disconnect(self):
        """قطع الاتصال"""
        if self.connection and self.connection.is_open:
            self.connection.close()
        self.connected = False
        print("🔌 تم قطع الاتصال")
        
    def send_command(self, command):
        """إرسال أمر للعربة الحقيقية"""
        if not self.connected or not self.connection:
            return False
            
        try:
            # إرسال الأمر
            self.connection.write(command.encode())
            
            # قراءة التأكيد
            response = self.connection.readline().decode().strip()
            
            # حفظ في التاريخ
            self.command_history.append({
                'command': command,
                'timestamp': time.time(),
                'response': response
            })
            
            # الاحتفاظ بآخر 50 أمر فقط
            if len(self.command_history) > 50:
                self.command_history.pop(0)
                
            self.last_command = command
            print(f"📤 أرسلنا للعربة الحقيقية: {command} | استجابة: {response}")
            return True
            
        except Exception as e:
            print(f"❌ خطأ في إرسال الأمر: {e}")
            self.connected = False
            return False
            
    def get_status(self):
        """الحصول على حالة الاتصال"""
        return {
            'connected': self.connected,
            'port': self.port,
            'last_command': self.last_command,
            'command_count': len(self.command_history)
        }

# ============ خوارزمية DWA المدمجة ============
class DWAConfig:
    def __init__(self):
        self.max_speed = ROBOT_SPEED_MAX
        self.min_speed = 0.0
        self.max_yaw_rate = 40.0 * math.pi / 180.0
        self.max_accel = 1.5
        self.max_delta_yaw_rate = 40.0 * math.pi / 180.0
        self.velocity_resolution = 0.05
        self.yaw_rate_resolution = 0.1 * math.pi / 180.0
        self.to_goal_cost_gain = 0.8
        self.speed_cost_gain = 0.5
        self.obstacle_cost_gain = 2.0
        self.robot_stuck_flag_cons = 0.001
        self.dt = 0.1

class DWAPlanner:
    def __init__(self, config):
        self.config = config
        
    def calc_dynamic_window(self, x, config):
        Vs = [config.min_speed, config.max_speed,
              -config.max_yaw_rate, config.max_yaw_rate]
        Vd = [x[3] - config.max_accel * config.dt,
              x[3] + config.max_accel * config.dt,
              x[4] - config.max_delta_yaw_rate * config.dt,
              x[4] + config.max_delta_yaw_rate * config.dt]
        dw = [max(Vs[0], Vd[0]), min(Vs[1], Vd[1]),
              max(Vs[2], Vd[2]), min(Vs[3], Vd[3])]
        return dw
        
    def predict_trajectory(self, x_init, v, y, config):
        x = np.array(x_init)
        trajectory = np.array(x)
        time = 0
        
        while time <= 3.0:
            x = self.motion(x, [v, y], config.dt)
            trajectory = np.vstack((trajectory, x))
            time += config.dt
            
        return trajectory
        
    def motion(self, x, u, dt):
        x[2] += u[1] * dt
        x[0] += u[0] * math.cos(x[2]) * dt
        x[1] += u[0] * math.sin(x[2]) * dt
        x[3] = u[0]
        x[4] = u[1]
        return x
        
    def calc_obstacle_cost(self, trajectory, ob):
        min_r = float("inf")
        
        for ii in range(len(trajectory)):
            for i in range(len(ob)):
                ox = ob[i][0]
                oy = ob[i][1]
                dx = trajectory[ii, 0] - ox
                dy = trajectory[ii, 1] - oy
                r = math.sqrt(dx**2 + dy**2)
                
                if r <= ROBOT_W/2:
                    return float("inf")
                    
                if r < min_r:
                    min_r = r
                    
        return 1.0 / min_r if min_r < 2*ROBOT_W else 0.0
        
    def calc_to_goal_cost(self, trajectory, goal):
        dx = goal[0] - trajectory[-1, 0]
        dy = goal[1] - trajectory[-1, 1]
        error_angle = math.atan2(dy, dx)
        cost_angle = error_angle - trajectory[-1, 2]
        cost = abs(math.atan2(math.sin(cost_angle), math.cos(cost_angle)))
        return cost
        
    def dwa_control(self, x, config, goal, ob):
        dw = self.calc_dynamic_window(x, config)
        u, trajectory = self.calc_control_and_trajectory(x, dw, config, goal, ob)
        return u, trajectory
        
    def calc_control_and_trajectory(self, x, dw, config, goal, ob):
        x_init = x[:]
        min_cost = float("inf")
        best_u = [0.0, 0.0]
        best_trajectory = np.array([x])
        
        for v in np.arange(dw[0], dw[1], config.velocity_resolution):
            for y in np.arange(dw[2], dw[3], config.yaw_rate_resolution):
                
                trajectory = self.predict_trajectory(x_init, v, y, config)
                
                to_goal_cost = config.to_goal_cost_gain * self.calc_to_goal_cost(trajectory, goal)
                speed_cost = config.speed_cost_gain * (config.max_speed - trajectory[-1, 3])
                ob_cost = config.obstacle_cost_gain * self.calc_obstacle_cost(trajectory, ob)
                
                final_cost = to_goal_cost + speed_cost + ob_cost
                
                if min_cost >= final_cost:
                    min_cost = final_cost
                    best_u = [v, y]
                    best_trajectory = trajectory
                    
        return best_u, best_trajectory

# ============ الذكاء الاصطناعي المطور مع دعم العربة الحقيقية ============
class SmartAIBrain:
    def __init__(self, real_robot_controller=None):
        self.enabled = False
        self.dwa_planner = DWAPlanner(DWAConfig())
        self.real_robot_controller = real_robot_controller
        
        # ذاكرة ذكية متطورة
        self.position_memory = deque(maxlen=200)
        self.failed_paths = deque(maxlen=50)
        self.success_paths = deque(maxlen=50)
        self.obstacle_memory = []
        
        # ذكاء التفكير
        self.stuck_counter = 0
        self.last_position = (0, 0)
        self.exploration_targets = []
        self.current_goal = None
        self.path_attempt_counter = {}
        
        # وضع التفكير المتقدم
        self.thinking_mode = "explore"
        self.decision_confidence = 0.0
        
        # ذاكرة الحساس الخلفي
        self.rear_obstacle_memory = deque(maxlen=20)
        self.reverse_maneuver_counter = 0
        
        # وضع العربة الحقيقية
        self.real_robot_mode = False
        
    def toggle_real_robot_mode(self):
        """تبديل بين المحاكي والعربة الحقيقية"""
        if self.real_robot_controller and self.real_robot_controller.connected:
            self.real_robot_mode = not self.real_robot_mode
            print(f"🔄 تم تبديل الوضع: {'عربة حقيقية' if self.real_robot_mode else 'محاكي'}")
            return True
        else:
            print("❌ العربة الحقيقية غير متصلة")
            return False
        
    def toggle(self):
        """تفعيل/إيقاف الذكاء الاصطناعي"""
        self.enabled = not self.enabled
        if self.enabled:
            self.reset_memory()
            
    def reset_memory(self):
        """إعادة تهيئة الذاكرة"""
        self.position_memory.clear()
        self.failed_paths.clear()
        self.success_paths.clear()
        self.stuck_counter = 0
        self.thinking_mode = "explore"
        self.rear_obstacle_memory.clear()
        self.reverse_maneuver_counter = 0
        
    def is_stuck(self, robot):
        """فحص متقدم للعلق بناءً على الذاكرة"""
        current_pos = (robot.x, robot.y)
        
        distance_moved = math.sqrt((current_pos[0] - self.last_position[0])**2 + 
                                 (current_pos[1] - self.last_position[1])**2)
        
        if distance_moved < 3:
            self.stuck_counter += 1
        else:
            self.stuck_counter = max(0, self.stuck_counter - 2)
            
        recent_positions = list(self.position_memory)[-20:]
        if len(recent_positions) > 15:
            center_x = sum(p[0] for p in recent_positions) / len(recent_positions)
            center_y = sum(p[1] for p in recent_positions) / len(recent_positions)
            avg_distance = sum(math.sqrt((p[0]-center_x)**2 + (p[1]-center_y)**2) 
                             for p in recent_positions) / len(recent_positions)
            
            if avg_distance < 30:
                self.stuck_counter += 2
                
        self.last_position = current_pos
        return self.stuck_counter > 20
        
    def analyze_environment_smart(self, robot, sensors):
        """تحليل ذكي متقدم للبيئة مع الحساس الخلفي"""
        analysis = {
            'sensors': sensors,
            'front_sensors': sensors[:-1],
            'rear_sensor': sensors[-1],
            'safe_directions': [],
            'dangerous_directions': [],
            'best_exploration_angle': 0,
            'escape_route': None,
            'confidence': 0.0,
            'rear_clear': sensors[-1] > REAR_SAFE_DISTANCE,
            'can_reverse': sensors[-1] > REAR_SAFE_DISTANCE * 1.5
        }
        
        self.rear_obstacle_memory.append(sensors[-1])
        
        for i, distance in enumerate(sensors[:-1]):
            angle = SENSOR_ANGLES[i]
            
            if distance > EARLY_TURN_DISTANCE:
                analysis['safe_directions'].append((angle, distance))
            elif distance < SAFE_DISTANCE:
                analysis['dangerous_directions'].append((angle, distance))
                
        if analysis['safe_directions']:
            best_dir = max(analysis['safe_directions'], key=lambda x: x[1])
            analysis['best_exploration_angle'] = best_dir[0]
            analysis['confidence'] = min(1.0, best_dir[1] / SENSOR_MAX_DIST)
            
        if len(analysis['dangerous_directions']) > 6:
            safe_gaps = []
            front_sensors = sensors[:-1]
            for i in range(len(front_sensors)-1):
                if front_sensors[i] > CRITICAL_DISTANCE and front_sensors[i+1] > CRITICAL_DISTANCE:
                    gap_angle = (SENSOR_ANGLES[i] + SENSOR_ANGLES[i+1]) / 2
                    gap_width = min(front_sensors[i], front_sensors[i+1])
                    safe_gaps.append((gap_angle, gap_width))
                    
            if safe_gaps:
                analysis['escape_route'] = max(safe_gaps, key=lambda x: x[1])
            elif analysis['can_reverse']:
                analysis['escape_route'] = (180, sensors[-1])
                
        return analysis
        
    def remember_path_result(self, start_pos, end_pos, success):
        """تذكر نتيجة المسار (نجح أم فشل)"""
        path_key = (int(start_pos[0]/50)*50, int(start_pos[1]/50)*50,
                   int(end_pos[0]/50)*50, int(end_pos[1]/50)*50)
        
        if success:
            self.success_paths.append(path_key)
            if path_key in self.path_attempt_counter:
                del self.path_attempt_counter[path_key]
        else:
            self.failed_paths.append(path_key)
            self.path_attempt_counter[path_key] = self.path_attempt_counter.get(path_key, 0) + 1
            
    def has_tried_path_before(self, start_pos, target_pos):
        """فحص إذا تم تجربة هذا المسار من قبل وفشل"""
        path_key = (int(start_pos[0]/50)*50, int(start_pos[1]/50)*50,
                   int(target_pos[0]/50)*50, int(target_pos[1]/50)*50)
        
        return self.path_attempt_counter.get(path_key, 0) > 3
        
    def find_smart_goal(self, robot, analysis):
        """إيجاد هدف ذكي للاستكشاف"""
        current_pos = (robot.x, robot.y)
        
        potential_goals = []
        
        if analysis['safe_directions']:
            for angle, distance in analysis['safe_directions']:
                goal_angle = math.radians(robot.angle + angle)
                goal_distance = min(distance - 30, 150)
                goal_x = robot.x + goal_distance * math.cos(goal_angle)
                goal_y = robot.y + goal_distance * math.sin(goal_angle)
                
                if 50 < goal_x < WINDOW_W-50 and 50 < goal_y < WINDOW_H-50:
                    if not self.has_tried_path_before(current_pos, (goal_x, goal_y)):
                        potential_goals.append((goal_x, goal_y, distance))
        
        if analysis['can_reverse'] and not potential_goals:
            rear_angle = math.radians(robot.angle + 180)
            rear_distance = min(analysis['rear_sensor'] - 30, 100)
            rear_goal_x = robot.x + rear_distance * math.cos(rear_angle)
            rear_goal_y = robot.y + rear_distance * math.sin(rear_angle)
            
            if 50 < rear_goal_x < WINDOW_W-50 and 50 < rear_goal_y < WINDOW_H-50:
                potential_goals.append((rear_goal_x, rear_goal_y, analysis['rear_sensor']))
                        
        for _ in range(5):
            goal_x = random.randint(100, WINDOW_W-100)
            goal_y = random.randint(100, WINDOW_H-100)
            if not self.has_tried_path_before(current_pos, (goal_x, goal_y)):
                potential_goals.append((goal_x, goal_y, 100))
                
        if potential_goals:
            potential_goals.sort(key=lambda x: -x[2])
            return potential_goals[0][:2]
            
        return (WINDOW_W/2, WINDOW_H/2)
        
    def make_smart_decision(self, robot, sensors):
        """صنع القرار الذكي المتطور مع دعم العربة الحقيقية"""
        if not self.enabled:
            return 0.0, 0.0
            
        current_pos = (robot.x, robot.y)
        self.position_memory.append(current_pos)
        
        analysis = self.analyze_environment_smart(robot, sensors)
        
        if self.is_stuck(robot):
            self.thinking_mode = "escape"
        elif analysis['confidence'] > 0.7:
            self.thinking_mode = "explore"
        else:
            self.thinking_mode = "navigate"
            
        # إرسال الأمر للعربة الحقيقية إذا كانت متصلة
        if self.real_robot_mode and self.real_robot_controller and self.real_robot_controller.connected:
            self.send_command_to_real_robot(robot, sensors, analysis)
            
        if self.thinking_mode == "escape":
            return self.escape_strategy(robot, analysis)
        elif self.thinking_mode == "explore":
            return self.exploration_strategy(robot, analysis)
        else:
            return self.navigation_strategy(robot, analysis)
            
    def send_command_to_real_robot(self, robot, sensors, analysis):
        """إرسال أمر للعربة الحقيقية بناءً على التحليل"""
        if not self.real_robot_controller or not self.real_robot_controller.connected:
            return
            
        # تحديد أفضل اتجاه للحركة
        if analysis['safe_directions']:
            best_dir = max(analysis['safe_directions'], key=lambda x: x[1])
            target_angle = best_dir[0]
            
            # تحويل الزاوية إلى أمر حركة
            if abs(target_angle) < 20:
                command = 'F'  # أمام
            elif target_angle > 0:
                command = 'I'  # أمام + يمين
            else:
                command = 'G'  # أمام + شمال
        else:
            # البحث عن طريق هروب
            if analysis['can_reverse']:
                command = 'B'  # خلف
            elif analysis['escape_route']:
                escape_angle = analysis['escape_route'][0]
                if escape_angle > 0:
                    command = 'R'  # يمين
                else:
                    command = 'L'  # يسار
            else:
                command = 'S'  # توقف
                
        # إرسال الأمر
        self.real_robot_controller.send_command(command)
        
    def escape_strategy(self, robot, analysis):
        """استراتيجية الهروب من العلق مع استخدام الحساس الخلفي"""
        linear = 0.0
        angular = 0.0
        
        if analysis['can_reverse'] and self.reverse_maneuver_counter < 3:
            self.reverse_maneuver_counter += 1
            linear = -robot.speed * 0.6
            angular = 0.0
            return linear, angular
        
        if analysis['escape_route']:
            escape_angle = analysis['escape_route'][0]
            
            if escape_angle == 180 and analysis['can_reverse']:
                linear = -robot.speed * 0.5
                angular = 0.0
            else:
                angle_diff = (escape_angle) % 360
                if angle_diff > 180:
                    angle_diff -= 360
                    
                if abs(angle_diff) > 15:
                    angular = 6.0 if angle_diff > 0 else -6.0
                    linear = 0.3
                else:
                    angular = 0
                    linear = robot.speed * 0.8
        else:
            recent_positions = list(self.position_memory)[-10:]
            if len(recent_positions) > 5:
                center_x = sum(p[0] for p in recent_positions) / len(recent_positions)
                center_y = sum(p[1] for p in recent_positions) / len(recent_positions)
                
                escape_x = robot.x - (center_x - robot.x)
                escape_y = robot.y - (center_y - robot.y)
                
                escape_angle = math.degrees(math.atan2(escape_y - robot.y, escape_x - robot.x))
                angle_diff = (escape_angle - robot.angle) % 360
                if angle_diff > 180:
                    angle_diff -= 360
                    
                angular = 5.0 if angle_diff > 0 else -5.0
                linear = 0.4
                
                if abs(linear) < 0.2 and analysis['can_reverse']:
                    linear = -0.5
                    
            else:
                angular = random.choice([-6.0, 6.0])
                linear = 0.2
                
        return linear, angular
        
    def exploration_strategy(self, robot, analysis):
        """استراتيجية الاستكشاف الذكي"""
        self.reverse_maneuver_counter = 0
        
        if not self.current_goal or random.randint(1, 100) == 1:
            self.current_goal = self.find_smart_goal(robot, analysis)
            
        x = [robot.x, robot.y, math.radians(robot.angle), robot.speed, 0.0]
        goal = list(self.current_goal)
        
        obstacles = []
        for i, distance in enumerate(analysis['front_sensors']):
            if distance < SENSOR_MAX_DIST:
                angle_world = math.radians(robot.angle + SENSOR_ANGLES[i])
                obs_x = robot.x + distance * math.cos(angle_world)
                obs_y = robot.y + distance * math.sin(angle_world)
                obstacles.append([obs_x, obs_y])
                
        if analysis['rear_sensor'] < SENSOR_MAX_DIST:
            angle_world = math.radians(robot.angle + 180)
            obs_x = robot.x + analysis['rear_sensor'] * math.cos(angle_world)
            obs_y = robot.y + analysis['rear_sensor'] * math.sin(angle_world)
            obstacles.append([obs_x, obs_y])
                
        try:
            u, trajectory = self.dwa_planner.dwa_control(x, self.dwa_planner.config, goal, obstacles)
            
            linear = u[0] if u[0] > 0.1 else 0.4
            angular = u[1] * 180.0 / math.pi / 2.0
            
            distance_to_goal = math.sqrt((robot.x - self.current_goal[0])**2 + 
                                       (robot.y - self.current_goal[1])**2)
            if distance_to_goal < 50:
                self.remember_path_result(self.last_position, self.current_goal, True)
                self.current_goal = None
                
        except:
            return self.fallback_strategy(robot, analysis)
            
        return linear, angular
        
    def navigation_strategy(self, robot, analysis):
        """استراتيجية التنقل العادي المحسن مع الحساس الخلفي"""
        linear = 0.0
        angular = 0.0
        
        front_sensors = analysis['front_sensors'][3:6]
        min_front = min(front_sensors)
        
        if min_front > EARLY_TURN_DISTANCE:
            linear = robot.speed
            
            if analysis['safe_directions']:
                best_dir = max(analysis['safe_directions'], key=lambda x: x[1])
                target_angle = best_dir[0]
                
                if abs(target_angle) > 10:
                    angular = 1.5 if target_angle > 0 else -1.5
                    
        elif min_front > SAFE_DISTANCE:
            linear = robot.speed * 0.6
            
            if analysis['safe_directions']:
                best_dir = max(analysis['safe_directions'], key=lambda x: x[1])
                target_angle = best_dir[0]
                angular = 3.0 if target_angle > 0 else -3.0
                
        else:
            linear = 0.1
            if analysis['escape_route']:
                escape_angle = analysis['escape_route'][0]
                if escape_angle == 180 and analysis['can_reverse']:
                    linear = -0.4
                    angular = 0.0
                else:
                    angular = 4.0 if escape_angle > 0 else -4.0
            else:
                front_sensors_only = analysis['front_sensors']
                right_side = sum(front_sensors_only[6:9])
                left_side = sum(front_sensors_only[0:3])
                
                if analysis['can_reverse'] and min(front_sensors_only) < CRITICAL_DISTANCE:
                    linear = -0.3
                    angular = 0.0
                else:
                    angular = 4.0 if right_side > left_side else -4.0
                
        return linear, angular
        
    def fallback_strategy(self, robot, analysis):
        """استراتيجية احتياطية محسنة مع الحساس الخلفي"""
        if analysis['safe_directions']:
            best_dir = max(analysis['safe_directions'], key=lambda x: x[1])
            target_angle = best_dir[0]
            
            linear = robot.speed * 0.7
            angular = 2.0 if target_angle > 0 else -2.0
        else:
            front_sensors_only = analysis['front_sensors']
            least_dangerous = max(enumerate(front_sensors_only), key=lambda x: x[1])
            target_angle = SENSOR_ANGLES[least_dangerous[0]]
            
            if analysis['rear_sensor'] > least_dangerous[1] and analysis['can_reverse']:
                linear = -0.3
                angular = 0.0
            else:
                linear = 0.3
                angular = 3.0 if target_angle > 0 else -3.0
            
        return linear, angular

# ============ فئة الروبوت مع دعم العربة الحقيقية ============
class Robot:
    def __init__(self, x, y, angle=0.0, real_robot_controller=None):
        self.x = x
        self.y = y
        self.angle = angle
        self.w = ROBOT_W
        self.h = ROBOT_H
        self.speed = ROBOT_SPEED_DEFAULT
        self.rot_speed = ROBOT_ROT_SPEED
        self.sensor_angles_rel = SENSOR_ANGLES
        self.sensors = [SENSOR_MAX_DIST] * len(self.sensor_angles_rel)
        self.history = deque(maxlen=400)
        self.ai_brain = SmartAIBrain(real_robot_controller)
        self.real_robot_controller = real_robot_controller

    def set_speed(self, speed):
        self.speed = max(ROBOT_SPEED_MIN, min(ROBOT_SPEED_MAX, speed))

    def world_to_local_angle(self, rel_angle):
        return (self.angle + rel_angle) % 360

    def sense(self, surface):
        readings = []
        endpoints = []
        hits = []
        for a in self.sensor_angles_rel:
            ang_world = self.world_to_local_angle(a)
            dist, (ex, ey), hit = cast_ray(surface, self.x, self.y, ang_world)
            readings.append(dist)
            endpoints.append((ex, ey))
            hits.append(hit)
        self.sensors = readings
        return readings, endpoints, hits

    def get_side_clearance(self):
        """حساب المساحة الواضحة للجوانب والخلف"""
        left = sum(self.sensors[0:3])
        front = sum(self.sensors[3:6])
        right = sum(self.sensors[6:9])
        rear = self.sensors[-1]
        return left, front, right, rear

    def step(self, surface):
        readings, ends, hits = self.sense(surface)
        
        # اختيار نوع التحكم
        if self.ai_brain.enabled:
            # استخدام الذكاء الاصطناعي المتطور مع DWA والحساس الخلفي
            linear, angular = self.ai_brain.make_smart_decision(self, readings)
        else:
            # التحكم التقليدي المُحسّن - ذكي أكثر مع الحساس الخلفي
            linear, angular = self.traditional_smart_control(readings)

        # تطبيق الحركة مع دعم الحركة الخلفية
        self.angle = (self.angle + angular * self.rot_speed/ROBOT_ROT_SPEED) % 360
        rad = math.radians(self.angle)
        nx = self.x + math.cos(rad) * linear
        ny = self.y + math.sin(rad) * linear

        if not is_obstacle_at(surface, nx, ny):
            self.x = nx
            self.y = ny
        else:
            # ارتداد ذكي عند التصادم
            if linear > 0:  # كان يتحرك للأمام
                # جرب الحركة الخلفية إذا كان الطريق خالي
                if readings[-1] > REAR_SAFE_DISTANCE:
                    self.x -= math.cos(rad) * 0.5
                    self.y -= math.sin(rad) * 0.5
                else:
                    self.angle = (self.angle + 30 * random.choice([-1,1])) % 360
            else:  # كان يتحرك للخلف
                self.angle = (self.angle + 45 * random.choice([-1,1])) % 360

        self.history.append((self.x, self.y))
        return readings, ends, hits
        
    def traditional_smart_control(self, readings):
        """التحكم التقليدي المحسن - أكثر ذكاءً مع الحساس الخلفي"""
        left_clear, front_clear, right_clear, rear_clear = self.get_side_clearance()
        
        # الحساسات الأمامية المهمة
        front_center = readings[4]  # المتوسط
        front_left = readings[3]    # يسار الوسط
        front_right = readings[5]   # يمين الوسط
        rear_sensor = readings[-1]  # الحساس الخلفي
        
        linear = 0.0
        angular = 0.0
        
        # التفكير المبكر - عدم انتظار الاقتراب كثيراً
        if front_center > EARLY_TURN_DISTANCE and front_left > SAFE_DISTANCE and front_right > SAFE_DISTANCE:
            # الطريق واضح تماماً - تحرك بثقة
            linear = self.speed
            
            # تفضيل الاتجاه الأوسع
            if right_clear > left_clear + 50:
                angular = 2.0  # انحياز طفيف لليمين
            elif left_clear > right_clear + 50:
                angular = -2.0  # انحياز طفيف لليسار
            else:
                angular = random.uniform(-0.8, 0.8)  # حركة طبيعية
                
        elif front_center > SAFE_DISTANCE:
            # حذر - الطريق ضيق نوعاً ما
            linear = self.speed * 0.7
            
            # اختيار الجانب الأوسع مبكراً
            if right_clear > left_clear + 30:
                angular = 3.5
            elif left_clear > right_clear + 30:
                angular = -3.5
            else:
                # البحث عن أفضل فجوة في الحساسات (بدون الخلفي)
                front_readings = readings[:-1]
                max_distance = max(front_readings)
                best_sensor = front_readings.index(max_distance)
                best_angle = SENSOR_ANGLES[best_sensor]
                angular = 2.5 if best_angle > 0 else -2.5
                
        else:
            # خطر قريب - تصرف فوراً
            linear = 0.2
            
            # فحص إمكانية استخدام الحساس الخلفي
            if rear_sensor > REAR_SAFE_DISTANCE * 1.5 and front_center < CRITICAL_DISTANCE:
                # استخدم الحركة الخلفية
                linear = -0.4
                angular = 0.0
            elif right_clear > left_clear + 20:
                angular = 5.0
            elif left_clear > right_clear + 20:
                angular = -5.0
            else:
                # البحث في جميع الحساسات عن أفضل مخرج (بدون الخلفي)
                sensor_scores = []
                front_readings = readings[:-1]
                for i, dist in enumerate(front_readings):
                    angle = SENSOR_ANGLES[i]
                    # تقييم الحساس بناءً على المسافة وسهولة الوصول
                    score = dist * (1.0 - abs(angle)/180.0)  # تفضيل الزوايا الأقرب للأمام
                    sensor_scores.append((score, angle))
                
                best_score, best_angle = max(sensor_scores)
                
                # مقارنة مع الحساس الخلفي
                rear_score = rear_sensor * 0.8  # تقليل قيمة الحساس الخلفي قليلاً
                
                if rear_score > best_score and rear_sensor > REAR_SAFE_DISTANCE:
                    # استخدم الحركة الخلفية
                    linear = -0.3
                    angular = 0.0
                elif best_score > CRITICAL_DISTANCE:
                    angular = 4.0 if best_angle > 0 else -4.0
                else:
                    # حالة طوارئ - دوران قوي
                    angular = random.choice([-6.0, 6.0])
        
        return linear, angular

# ============ وظائف مساعدة للخريطة ============
def load_map_image(path):
    try:
        img = pygame.image.load(path)
        img = pygame.transform.scale(img, (WINDOW_W, WINDOW_H))
        return img
    except Exception as e:
        print(f"خطأ في تحميل الخريطة من {path}: {e}")
        return None

def is_obstacle_at(surface, x, y):
    if x < 0 or x >= WINDOW_W or y < 0 or y >= WINDOW_H:
        return True
    col = surface.get_at((int(x), int(y)))
    threshold = 600
    return (col.r + col.g + col.b) < threshold

def cast_ray(surface, x0, y0, angle_deg, max_dist=SENSOR_MAX_DIST):
    angle = math.radians(angle_deg)
    dx = math.cos(angle) * SENSOR_STEP
    dy = math.sin(angle) * SENSOR_STEP
    x, y = x0, y0
    traveled = 0.0
    while traveled < max_dist:
        x += dx
        y += dy
        traveled += SENSOR_STEP
        if x < 0 or x >= WINDOW_W or y < 0 or y >= WINDOW_H:
            return traveled, (x, y), True
        if is_obstacle_at(surface, x, y):
            return traveled, (x, y), True
    return max_dist, (x, y), False

# ============ وظائف النصوص العربية ============
def render_arabic_text(font, text, color):
    english_translations = {
        "لوحة التحكم": "Control Panel",
        "السرعة": "Speed", 
        "وضع عادي": "Normal",
        "أضف عقبة": "Add Wall",
        "احذف عقبة": "Remove",
        "ارسم": "Draw",
        "مسح الكل": "Clear All",
        "انتهيت": "Finish",
        "عقبة عشوائية": "Random Wall",
        "الوضع": "Mode",
        "الموقع": "Position",
        "الزاوية": "Angle",
        "قراءات الحساسات": "Sensors",
        "يسار": "Left",
        "أمام": "Front", 
        "يمين": "Right",
        "خلف": "Rear",
        "تعليمات": "Instructions",
        "ذكاء اصطناعي": "Smart AI",
        "عربة حقيقية": "Real Robot",
        "اتصال": "Connect",
        "قطع اتصال": "Disconnect"
    }
    
    display_text = english_translations.get(text, text)
    return font.render(display_text, True, color)

# ============ فئة الأزرار والشرائح ============
class Button:
    def __init__(self, x, y, w, h, text, action=None, color_scheme=None):
        self.rect = pygame.Rect(x, y, w, h)
        self.text = text
        self.action = action
        self.hovered = False
        self.active = False
        self.color_scheme = color_scheme or {}
        
    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.rect.collidepoint(event.pos):
                if self.action:
                    self.action()
                return True
        elif event.type == pygame.MOUSEMOTION:
            self.hovered = self.rect.collidepoint(event.pos)
        return False
        
    def draw(self, surface, font):
        if self.active:
            color = self.color_scheme.get('active', BUTTON_ACTIVE)
        elif self.hovered:
            color = self.color_scheme.get('hover', BUTTON_HOVER)
        else:
            color = self.color_scheme.get('normal', BUTTON_COLOR)
            
        pygame.draw.rect(surface, color, self.rect)
        pygame.draw.rect(surface, (0,0,0), self.rect, 2)
        
        text_surf = render_arabic_text(font, self.text, (0,0,0))
        text_rect = text_surf.get_rect(center=self.rect.center)
        surface.blit(text_surf, text_rect)

class Slider:
    def __init__(self, x, y, w, h, min_val, max_val, initial_val, label):
        self.rect = pygame.Rect(x, y, w, h)
        self.min_val = min_val
        self.max_val = max_val
        self.val = initial_val
        self.label = label
        self.dragging = False
        self.handle_rect = pygame.Rect(0, 0, 20, h)
        self.update_handle_pos()
        
    def update_handle_pos(self):
        ratio = (self.val - self.min_val) / (self.max_val - self.min_val)
        handle_x = self.rect.x + ratio * (self.rect.w - self.handle_rect.w)
        self.handle_rect.x = handle_x
        self.handle_rect.y = self.rect.y
        
    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.handle_rect.collidepoint(event.pos):
                self.dragging = True
                return True
        elif event.type == pygame.MOUSEBUTTONUP:
            self.dragging = False
        elif event.type == pygame.MOUSEMOTION and self.dragging:
            rel_x = event.pos[0] - self.rect.x
            rel_x = max(0, min(self.rect.w - self.handle_rect.w, rel_x))
            ratio = rel_x / (self.rect.w - self.handle_rect.w)
            self.val = self.min_val + ratio * (self.max_val - self.min_val)
            self.update_handle_pos()
            return True
        return False
        
    def draw(self, surface, font):
        pygame.draw.rect(surface, SLIDER_COLOR, self.rect)
        pygame.draw.rect(surface, (0,0,0), self.rect, 1)
        pygame.draw.rect(surface, SLIDER_HANDLE, self.handle_rect)
        pygame.draw.rect(surface, (0,0,0), self.handle_rect, 2)
        
        text = f"{self.label}: {self.val:.1f}"
        text_surf = render_arabic_text(font, text, (0,0,0))
        surface.blit(text_surf, (self.rect.x, self.rect.y - 25))