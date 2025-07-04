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
    def __init__(self, window, image_paths, interval, device_id):
        super().__init__(daemon=True)
        self.window = window
        self.image_paths = image_paths
        self.interval = interval
        self.device_id = device_id
        self.running = True

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
        # Reduced debug output
        y = y - offset_up  # Move click up by offset_up pixels
        cmd = ["adb", "-s", self.device_id, "shell", "input", "tap", str(x), str(y)]
        # print(f"[{self.window.title}] Running command: {' '.join(cmd)}")
        try:
            subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=5, shell=True)
        except Exception as e:
            print(f"[{self.window.title}] ADB Exception: {e}")

    def run(self):
        last_action_time = None  # None ensures first run triggers the action immediately
        print(f"[{self.window.title}] Bot started.")
        while self.running:
            now = time.time()
            # print(f"[{self.window.title}] Loop start. now={now}, last_action_time={last_action_time}, running={self.running}")
            if last_action_time is None or now - last_action_time >= 120:
                print(f"[{self.window.title}] 2-min periodic action")
                # TODO: Insert your periodic action here
                # open world
                self.adb_click(500, 960)
                time.sleep(5)
                screenshot, ignore_height = self.capture_window()
                # print(f"[{self.window.title}] Checking for 2/2 troops image...")
                if self.find_image("images/gather/2_2.png", screenshot, 0.95) is not None:
                    print(f"[{self.window.title}] 2/2 troops used. skipping.")
                    self.adb_click(500, 960)
                    time.sleep(5)
                else:
                    # print(f"[{self.window.title}] 2/2 troops NOT found, proceeding with resource click.")
                    self.adb_click(35, 811)
                    time.sleep(5)
                    screenshot2, ignore_height2 = self.capture_window()
                    # resource = random.choice(["Zent", "Wood", "Food"])
                    resource = random.choice([ "Food", "Wood"])
                    # print(f"[{self.window.title}] Random resource selected: {resource}")
                    if resource == "Wood":
                        tap_loc = self.find_image("images/lumberyard.png", screenshot2, 0.8)
                        # print(f"[{self.window.title}] Wood tap_loc: {tap_loc}")
                        if tap_loc is not None:
                            template = cv2.imread("images/lumberyard.png", cv2.IMREAD_UNCHANGED)
                            template_h, template_w = template.shape[:2]
                            tap_x = tap_loc[0] + (template_w // 2)
                            tap_y = ignore_height2 + tap_loc[1] + (template_h // 2)
                            self.adb_click(tap_x, tap_y)
                            time.sleep(5)
                    elif resource == "Food":
                        tap_loc = self.find_image("images/farmland.png", screenshot2, 0.8)
                        # print(f"[{self.window.title}] Food tap_loc: {tap_loc}")
                        if tap_loc is not None:
                            template = cv2.imread("images/farmland.png", cv2.IMREAD_UNCHANGED)
                            template_h, template_w = template.shape[:2]
                            tap_x = tap_loc[0] + (template_w // 2)
                            tap_y = ignore_height2 + tap_loc[1] + (template_h // 2)
                            self.adb_click(tap_x, tap_y)
                            time.sleep(5)
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
                            tap_loc = self.find_image("images/zentmining.png", screenshot2, 0.8)
                            # print(f"[{self.window.title}] Food tap_loc: {tap_loc}")
                            if tap_loc is not None:
                                template = cv2.imread("images/zentmining.png", cv2.IMREAD_UNCHANGED)
                                template_h, template_w = template.shape[:2]
                                tap_x = tap_loc[0] + (template_w // 2)
                                tap_y = ignore_height2 + tap_loc[1] + (template_h // 2)
                                self.adb_click(tap_x, tap_y)
                                time.sleep(5)
                    lvl = 6
                    # print(f"[{self.window.title}] Setting level: {lvl}")
                    if lvl == 1:
                        self.adb_click(167, 861, offset_up=0)
                        time.sleep(5)
                    elif lvl == 2:
                        self.adb_click(192, 861, offset_up=0)
                        time.sleep(5)
                    elif lvl == 3:
                        self.adb_click(237, 861, offset_up=0)
                        time.sleep(5)
                    elif lvl == 4:
                        self.adb_click(285, 861, offset_up=0)
                        time.sleep(5)
                    elif lvl == 5:
                        self.adb_click(342, 861, offset_up=0)
                        time.sleep(5)
                    elif lvl == 6:
                        self.adb_click(367, 861, offset_up=0)
                        time.sleep(5)
                    # print(f"[{self.window.title}] Clicking on world to finish periodic action.")
                    self.adb_click(270, 915, offset_up=0)
                    time.sleep(5)
                    self.adb_click(261, 505, offset_up=20)
                    time.sleep(5)
                    self.adb_click(270, 581, offset_up=20)
                    time.sleep(5)
                    self.adb_click(279, 735, offset_up=20)
                    time.sleep(5)
                    self.adb_click(500, 960)
                last_action_time = now
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
                    time.sleep(5)
                    if image_path == "images/troops/empty.png":
                        # print(f"[{self.window.title}] Empty troops logic triggered.")
                        self.adb_click(419, 955)
                        time.sleep(5)
                        self.adb_click(100, 100)
                        time.sleep(1)
                        cmd = ["adb", "-s", self.device_id, "shell", "input", "keyevent", "111"]; subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=5, shell=True)
            time.sleep(self.interval)

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
    image_paths = [
        "images/exp.png", "images/food.png", "images/electric.png", "images/zent.png",
        "images/troops/raider.png", "images/troops/assulter.png", "images/troops/shooter.png" , "images/help.png", "images/wood.png"
    ]
    interval = 1
    adb_devices = get_adb_devices()
    if not adb_devices:
        print("No adb devices found.")
        return
    memu_windows = find_memu_windows()
    if not memu_windows:
        print("No MEmu windows found.")
        return

    # Prompt user for which instance(s) to run
    print("\nAvailable MEmu instances:")
    for idx, window in enumerate(memu_windows):
        print(f"{idx+1}: {window.title}")
    print("A: All instances")

    selection = input("Select instance number (e.g. 1), or 'A' for all: ").strip().lower()
    selected_windows = []
    if selection == 'a':
        selected_windows = memu_windows
    else:
        try:
            idx = int(selection) - 1
            if 0 <= idx < len(memu_windows):
                selected_windows = [memu_windows[idx]]
            else:
                print("Invalid selection.")
                return
        except ValueError:
            print("Invalid input.")
            return

    # Move all selected windows side by side before starting bots
    screen_x = 0
    screen_y = 0
    max_height = 0
    for window in selected_windows:
        try:
            window.restore()  # Restore if minimized
            window.moveTo(screen_x, screen_y)
            screen_x += window.width
            if window.height > max_height:
                max_height = window.height
        except Exception as e:
            print(f"Could not move window '{window.title}': {e}")

    bots = []
    for window in selected_windows:
        device_id = get_device_id_for_window(window.title, adb_devices)
        if device_id is None:
            print(f"No adb device matched for window '{window.title}'")
            continue
        bot = BotInstance(window, image_paths, interval, device_id)
        bots.append(bot)

    if not bots:
        print("No bots started.")
        return

    keyboard.add_hotkey('ctrl+shift+q', lambda: [setattr(bot, 'running', False) for bot in bots])
    print("Press Ctrl+Shift+Q to stop all bots.")

    # Only start the bot threads, do NOT duplicate clicking logic here.
    for i, bot in enumerate(bots):
        bot.start()
        if i < len(bots) - 1:
            time.sleep(20)  # 20 second grace period between starting each bot

    try:
        while any(bot.running for bot in bots):
            time.sleep(1)
    except KeyboardInterrupt:
        print("KeyboardInterrupt received. Stopping all bots...")
        for bot in bots:
            bot.running = False
    print("All bots stopped.")

if __name__ == "__main__":
    main()
    print("All bots stopped.")
    print("All bots stopped.")

if __name__ == "__main__":
    main()
    print("All bots stopped.")
