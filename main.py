import cv2
import numpy as np
import pygetwindow as gw
import pyautogui
import time
import threading
import keyboard
import subprocess
import re
import random

def run_bot_rotation():
    from pymemuc import PyMemuc
    memu = PyMemuc()
    image_paths = [
        "images/exp.png", "images/food.png", "images/electric.png", "images/zent.png",
        "images/troops/raider.png", "images/troops/assulter.png", "images/troops/shooter.png" , "images/help.png", "images/wood.png",
    ]
    interval = 1
    instance_count = 3
    rotation_minutes = 5

    def wait_for_emulator_ready(device_id, timeout=300):
        start = time.time()
        while time.time() - start < timeout:
            result = subprocess.run(
                ["adb", "-s", device_id, "shell", "getprop", "sys.boot_completed"],
                capture_output=True, text=True, shell=True
            )
            if result.stdout.strip() == "1":
                print(f"Emulator {device_id} boot completed.")
                return True
            time.sleep(2)
        print(f"Timeout waiting for emulator {device_id} to boot.")
        return False

    while True:
        for idx in range(1, instance_count+1):
            print(f"\n=== Starting rotation for MEmu{idx} ===")
            memu.start_vm(idx)
            # Wait for emulator window to appear
            for _ in range(60):
                windows = gw.getAllWindows()
                found = any(f"MEmu{idx}" in w.title for w in windows)
                if found:
                    print(f"MEmu{idx} window detected.")
                    break
                time.sleep(2)
            else:
                print(f"MEmu{idx} did not start in time, skipping.")
                continue


            # Get the window and device id
            memu_windows = [w for w in gw.getAllWindows() if f"MEmu{idx}" in w.title]
            if not memu_windows:
                print(f"No window found for MEmu{idx}, skipping.")
                memu.stop_vm(idx)
                continue
            window = memu_windows[0]
            expected_port = 21513 + (idx-1)*10
            device_id = f"127.0.0.1:{expected_port}"

            # Wait for adb device to appear (up to 2 minutes)
            adb_timeout = 120
            adb_start = time.time()
            while time.time() - adb_start < adb_timeout:
                adb_devices = get_adb_devices()
                if device_id in adb_devices:
                    break
                print(f"Waiting for adb device {device_id}...")
                time.sleep(2)
            else:
                print(f"No adb device for {device_id} after waiting, skipping.")
                memu.stop_vm(idx)
                continue

            # Wait for emulator to be fully booted
            if not wait_for_emulator_ready(device_id, timeout=300):
                print(f"MEmu{idx} ({device_id}) did not boot in time, skipping.")
                memu.stop_vm(idx)
                continue

            # Launch the app: com.readygo.barrel.gp/com.im30.aps.debug.UnityPlayerActivityCustom
            launch_cmd = [
                "adb", "-s", device_id, "shell", "am", "start", "-n",
                "com.readygo.barrel.gp/com.im30.aps.debug.UnityPlayerActivityCustom"
            ]
            print(f"Launching Last Z: Survival Shooter on {device_id}...")
            subprocess.run(launch_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
            time.sleep(60)  # Give the app more time to start (was 30)

            # DPI scaling detection (Windows only)
            try:
                import ctypes
                user32 = ctypes.windll.user32
                user32.SetProcessDPIAware()
                screen_w = user32.GetSystemMetrics(0)
                screen_w_real = user32.GetSystemMetrics(78)  # SM_CXVIRTUALSCREEN
                dpi_scaling = screen_w_real / screen_w if screen_w > 0 else 1.0
                if dpi_scaling < 1.01:
                    # fallback: try Windows API for DPI
                    hdc = user32.GetDC(0)
                    dpi = ctypes.windll.gdi32.GetDeviceCaps(hdc, 88)  # LOGPIXELSX
                    dpi_scaling = dpi / 96.0
                print(f"[DPI] Detected scaling: {dpi_scaling:.2f}x")
            except Exception as e:
                print(f"[DPI] Could not determine DPI scaling, defaulting to 1.0x: {e}")
                dpi_scaling = 1.0

            bot = BotInstance(window, image_paths, interval, device_id, dpi_scaling=dpi_scaling)
            bot.start()
            print(f"Bot running on MEmu{idx} for {rotation_minutes} minutes...")
            # Let the bot run for the rotation period
            for _ in range(rotation_minutes*60):
                if not bot.running:
                    break
                time.sleep(1)
            bot.running = False
            bot.join()
            print(f"Stopping MEmu{idx}...")
            memu.stop_vm(idx)
            # Wait a bit before next instance
            time.sleep(10)
        print("All rotations complete.")


def get_adb_devices():
    """Returns a dict mapping IP:port strings to device serials from `adb devices` output."""
    result = subprocess.run(["adb", "devices"], capture_output=True, text=True, shell=True)
    devices = {}
    for line in result.stdout.splitlines():
        line = line.strip()
        if line and not line.startswith("List of devices"):
            parts = line.split()
            if len(parts) == 2 and parts[1] == "device":
                devices[parts[0]] = parts[0]  # key and value are the same here
    return devices

class BotInstance(threading.Thread):
    def __init__(self, window, image_paths, interval, device_id, dpi_scaling=1.0):
        super().__init__(daemon=True)
        self.window = window
        self.image_paths = image_paths
        self.interval = interval
        self.device_id = device_id
        self.running = True
        self.dpi_scaling = dpi_scaling

    def capture_window(self):
        x, y, w, h = self.window.left, self.window.top, self.window.width, self.window.height
        ignore_height = int(h * 0.1)
        y += ignore_height
        h -= ignore_height
        screenshot = pyautogui.screenshot(region=(x, y, w, h))
        return cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR), ignore_height

    def find_image(self, template_path, screenshot, threshold=0.8):
        template = cv2.imread(template_path, cv2.IMREAD_UNCHANGED)
        if template is None:
            return None
        # Scale template for DPI
        if self.dpi_scaling != 1.0:
            h, w = template.shape[:2]
            template = cv2.resize(template, (int(w * self.dpi_scaling), int(h * self.dpi_scaling)), interpolation=cv2.INTER_LINEAR)
        # If template has alpha channel, use it as mask
        if template.shape[2] == 4:
            template_rgb = template[:, :, :3]
            mask = template[:, :, 3]
            result = cv2.matchTemplate(screenshot, template_rgb, cv2.TM_CCOEFF_NORMED, mask=mask)
            _, max_val, _, max_loc = cv2.minMaxLoc(result)
            if max_val >= threshold:
                return max_loc
        result_color = cv2.matchTemplate(screenshot, template[:, :, :3] if template.shape[2] == 4 else template, cv2.TM_CCOEFF_NORMED)
        _, max_val_c, _, max_loc_c = cv2.minMaxLoc(result_color)
        screenshot_gray = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)
        template_gray = cv2.cvtColor(template[:, :, :3], cv2.COLOR_BGR2GRAY) if template.shape[2] == 4 else (cv2.cvtColor(template, cv2.COLOR_BGR2GRAY) if len(template.shape) == 3 else template)
        result_gray = cv2.matchTemplate(screenshot_gray, template_gray, cv2.TM_CCOEFF_NORMED)
        _, max_val_g, _, max_loc_g = cv2.minMaxLoc(result_gray)
        if max_val_c >= threshold:
            return max_loc_c
        elif max_val_g >= threshold:
            return max_loc_g
        return None

    def adb_click(self, x, y, offset_up=20):
        if not self.device_id:
            print(f"[{self.window.title}] ERROR: device_id not set.")
            return
        # Apply DPI scaling
        x = int(x * self.dpi_scaling)
        y = int((y - offset_up) * self.dpi_scaling)
        cmd = ["adb", "-s", self.device_id, "shell", "input", "tap", str(x), str(y)]
        try:
            subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=5, shell=True)
        except Exception as e:
            print(f"[{self.window.title}] ADB Exception: {e}")

    def run(self):
        last_action_time = None  # None ensures first run triggers the action immediately
        last_hourly_action_time = None  # Run hourly action immediately on startup
        first_actions = 0
        print(f"[{self.window.title}] Bot started.")
        while self.running:
            if first_actions == 0:
                # Click escape until a certain image appears
                escape_image = "images/home.png"  # Replace with your target image path
                max_tries = 30
                tries = 0
                while tries < max_tries:
                    screenshot, ignore_height = self.capture_window()
                    found = self.find_image(escape_image, screenshot, 0.8)
                    if found is not None:
                        print(f"[{self.window.title}] Escape target image found.")
                        break
                    # Send ESC key event via adb
                    cmd = ["adb", "-s", self.device_id, "shell", "input", "keyevent", "111"]
                    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=5, shell=True)
                    time.sleep(1)
                    tries += 1
                first_actions = 1
            now = time.time()
            # print(f"[{self.window.title}] Loop start. now={now}, last_action_time={last_action_time}, running={self.running}")
            # 2-min periodic action
            if last_action_time is None or now - last_action_time >= 120:
                print(f"[{self.window.title}] 2-min periodic action")
                self.do_2min_action()
                last_action_time = now

            # 1-hour periodic action placeholder
            if  last_hourly_action_time is None or now - last_hourly_action_time >= 3600:
                print(f"[{self.window.title}] 1-hour periodic action placeholder")
                self.do_hourly_action()
                last_hourly_action_time = time.time()

            screenshot, ignore_height = self.capture_window()
            for image_path in self.image_paths:
                tap_loc = self.find_image(image_path, screenshot, 0.8)
                # print(f"[{self.window.title}] Checking {image_path}, tap_loc={tap_loc}")
                if tap_loc is not None:
                    template = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
                    template_h, template_w = template.shape[:2]
                    tap_x = tap_loc[0] + (template_w // 2)
                    tap_y = ignore_height + tap_loc[1] + (template_h // 2)
                    # print(f"[{self.window.title}] Clicking at ({tap_x}, {tap_y})")
                    self.adb_click(tap_x, tap_y)
                    time.sleep(1)
                    if image_path == "images/troops/empty.png":
                        # print(f"[{self.window.title}] Empty troops logic triggered.")
                        self.adb_click(419, 955)
                        time.sleep(1)
                        self.adb_click(100, 100)
                        time.sleep(1)
                        cmd = ["adb", "-s", self.device_id, "shell", "input", "keyevent", "111"]; subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=5, shell=True)
            time.sleep(self.interval)

            screenshot, ignore_height = self.capture_window()
    def do_2min_action(self):
        # open world
        self.adb_click(500, 960)
        time.sleep(1)
        screenshot, ignore_height = self.capture_window()
        if self.find_image("images/gather/2_2.png", screenshot, 0.95) is not None:
            print(f"[{self.window.title}] 2/2 troops used. skipping.")
            self.adb_click(500, 960)
            time.sleep(1)
            return
        self.adb_click(35, 811)
        time.sleep(1)
        screenshot2, ignore_height2 = self.capture_window()
        resource = random.choice([ "Zent", "Wood", "Food" ])
        if resource == "Wood":
            tap_loc = self.find_image("images/lumberyard.png", screenshot2, 0.8)
            if tap_loc is not None:
                template = cv2.imread("images/lumberyard.png", cv2.IMREAD_UNCHANGED)
                template_h, template_w = template.shape[:2]
                tap_x = tap_loc[0] + (template_w // 2)
                tap_y = ignore_height2 + tap_loc[1] + (template_h // 2)
                self.adb_click(tap_x, tap_y)
                time.sleep(1)
        elif resource == "Food":
            tap_loc = self.find_image("images/farmland.png", screenshot2, 0.8)
            if tap_loc is not None:
                template = cv2.imread("images/farmland.png", cv2.IMREAD_UNCHANGED)
                template_h, template_w = template.shape[:2]
                tap_x = tap_loc[0] + (template_w // 2)
                tap_y = ignore_height2 + tap_loc[1] + (template_h // 2)
                self.adb_click(tap_x, tap_y)
                time.sleep(1)
        elif resource == "Zent":
            start_loc = self.find_image("images/farmland.png", screenshot2, 0.8)
            # print(f"[{self.window.title}] Zent start_loc: {start_loc}")
            if start_loc is not None:
                template = cv2.imread("images/farmland.png", cv2.IMREAD_UNCHANGED)
                template_h, template_w = template.shape[:2]
                start_x = start_loc[0] + (template_w // 2)
                start_y = ignore_height2 + start_loc[1] + (template_h // 2)
                end_x = 0
                end_y = start_y
                # print(f"[{self.window.title}] Swiping from ({start_x},{start_y}) to ({end_x},{end_y})")
                cmd = [
                    "adb", "-s", self.device_id, "shell", "input", "swipe",
                    str(start_x), str(start_y), str(end_x), str(end_y), "300"
                ]
                # print(f"[{self.window.title}] Running swipe command: {' '.join(cmd)}")
                try:
                    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=5, shell=True)
                except Exception as e:
                    print(f"[{self.window.title}] ADB Exception during drag: {e}")
                screenshot3, ignore_height3= self.capture_window()
                tap_loc = self.find_image("images/zentmining.png", screenshot3, 0.8)
                # print(f"[{self.window.title}] Food tap_loc: {tap_loc}")
                if tap_loc is not None:
                    template = cv2.imread("images/zentmining.png", cv2.IMREAD_UNCHANGED)
                    template_h, template_w = template.shape[:2]
                    tap_x = tap_loc[0] + (template_w // 2)
                    tap_y = ignore_height2 + tap_loc[1] + (template_h // 2)
                    self.adb_click(tap_x, tap_y)
                    time.sleep(5)
        if resource == "Zent":
            lvl = 5
        else:
            lvl = 6
        
        if lvl == 1:
            self.adb_click(167, 861, offset_up=0)
            time.sleep(1)
        elif lvl == 2:
            self.adb_click(192, 861, offset_up=0)
            time.sleep(1)
        elif lvl == 3:
            self.adb_click(237, 861, offset_up=0)
            time.sleep(1)
        elif lvl == 4:
            self.adb_click(285, 861, offset_up=0)
            time.sleep(1)
        elif lvl == 5:
            self.adb_click(342, 861, offset_up=0)
            time.sleep(1)
        elif lvl == 6:
            self.adb_click(367, 861, offset_up=0)
            time.sleep(1)
        self.adb_click(270, 915)
        time.sleep(1)
        self.adb_click(261, 505, offset_up=20)
        time.sleep(1)
        self.adb_click(270, 581, offset_up=20)
        time.sleep(1)
        self.adb_click(279, 735, offset_up=20)
        time.sleep(1)
        self.adb_click(500, 960)

    def do_hourly_action(self):
        print(f"[{self.window.title}] [DEBUG] Clicking at (510, 755) for hourly action")
        cmd = ["adb", "-s", self.device_id, "shell", "input", "tap", "510", "755"]
        print(f"[{self.window.title}] [DEBUG] Running command: {' '.join(cmd)}")
        self.adb_click(510, 755)
        time.sleep(1)
        self.adb_click(170, 630)
        time.sleep(1)
        screenshot2, ignore_height2 = self.capture_window()
        tap_loc = self.find_image("images/gather/research_recommended.png", screenshot2, 0.8)
        if tap_loc is not None:
            template = cv2.imread("images/gather/research_recommended.png", cv2.IMREAD_UNCHANGED)
            template_h, template_w = template.shape[:2]
            tap_x = tap_loc[0] + (template_w // 2)
            tap_y = ignore_height2 + tap_loc[1] + (template_h // 2)
            self.adb_click(tap_x, tap_y)
            time.sleep(1)
            for i in range(20):
                self.adb_click(360, 740)
                time.sleep(0.3)
            cmd = ["adb", "-s", self.device_id, "shell", "input", "keyevent", "111"]
            subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=5, shell=True)
            time.sleep(2)
            subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=5, shell=True)
            time.sleep(2)
            self.adb_click(400, 630)
            time.sleep(2)
            self.adb_click(300, 950)
            time.sleep(2)
            cmd = ["adb", "-s", self.device_id, "shell", "input", "keyevent", "111"]
            subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=5, shell=True)
            time.sleep(2)
            subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=5, shell=True)
            time.sleep(2)
            subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=5, shell=True)
            # 1-hour periodic action placeholder                              
            last_hourly_action_time = time.time()

    def stop(self):
        self.running = False

def find_memu_windows():
    print("Searching for MEmu windows...")
    memu_windows = []
    for window in gw.getAllWindows():
        if re.search(r'MEmu\s*[_\(]?(\d+)', window.title):
            memu_windows.append(window)
            print(f"Found MEmu window: '{window.title}'")
    return memu_windows

def get_device_id_for_window(window_title, adb_devices):
    match = re.search(r'MEmu\s*[_\(]?(\d+)', window_title)
    if not match:
        return None
    instance_num = int(match.group(1))
    # Create the expected port from instance_num:
    expected_port = 21513 + (instance_num - 1) * 10
    expected_device = f"127.0.0.1:{expected_port}"
    if expected_device in adb_devices:
        return expected_device
    return None

def main():
    run_bot_rotation()

if __name__ == "__main__":
    main()
