# --- SCRCPY/ADB BOT FOR PHYSICAL DEVICE (ONEPLUS A5010) ---
import cv2
import numpy as np
import time
import subprocess
import random
import os
import sys
import psutil
import datetime
import logging
import argparse

# --- CONFIGURATION ---
CONFIG = {
    "device_model": "ONEPLUS A5010",
    "scrcpy_path": os.path.join(os.path.dirname(os.path.abspath(__file__)), 'scrcpy', 'scrcpy.exe'),
    "debug_dir": "debug",
    "debug_level": 1, # 0=Quiet, 1=Info, 2=Debug (saves screenshots)
    "image_paths": {
        "home": "images/home.png",
        "hq": "images/hq.png",
        "quit_game": "images/quit_game.png",
        "exp": "images/exp.png",
        "food": "images/food.png",
        "electric": "images/electric.png",
        "zent": "images/zent.png",
        "raider": "images/troops/raider.png",
        "assulter": "images/troops/assulter.png",
        "shooter": "images/troops/shooter.png",
        "help": "images/help.png",
        "wood": "images/wood.png",
        "empty_troops": "images/troops/empty.png",
        "fuel": "images/fuel.png",
        "lumberyard": "images/lumberyard.png",
        "farmyard": "images/farmyard.png",
        "zentyard": "images/zentyard.png",
        "research_recommended": "images/research_recommended.png",
        "research_recommended2": "images/research_recommended2.png",
        "troops_2_of_2": "images/gather/3_3.png",
        "collect_8hrs": "images/collect_8hours.png",
        "research_free": "images/research_free.png",
        "research_confirm": "images/research_confirm.png",
    },
    "game_coords": {
        "game1": (126, 1191),
        "game2": (331, 1191),
        "game3": (549, 1191),
        "game4": (745,1191),
        "game5": (957,1191)
    },
    "game_settings": {
        "game1": {"train_troops": True},
        "game2": {"train_troops": True},
        "game3": {"train_troops": True},
        "game4": {"train_troops": True},
        "game5": {"train_troops": True}
    },
    "action_coords": {
        "open_world": (972, 2070),
        "search_button": (64, 1746),
        "alliance": (1036, 1634),
        "alliance_techs": (349, 1350),
        "research_help": (745, 1594),
        "alliance_gifts": (856, 1335),
        "alliance_gifts_claimall": (561, 2070),
        "gather_search": (565, 2062),
        "gather_confirm_march": (540, 1569),
        "train_troops_confirm": (820, 2034),
        "train_troops_select": (10, 10),
        "gather_button": (536, 1252),
        "middle_screen": (540, 1072),
        "8hrs_chest_collect": (558, 1594)
    },
    "resource_levels": {
        1: (316, 1933),
        2: (432, 1933),
        3: (478, 1933),
        4: (600, 1933),
        5: (712, 1933),
        6: (774, 1933),
    },
    "timing": {
        "interval": 1,
        "rotation_minutes": 10,
        "action_delay": 2,
        "long_delay": 5,
        "swipe_duration": 300,
        "fast_click_delay": 0.3,
        "periodic_action_interval_2min": 120,
        "periodic_action_interval_hourly": 3600,
    },
    "click_offset": 1 # Max random pixels to add to each click coordinate
}

# --- UTILITY FUNCTIONS ---

def get_device_id():
    """Returns the device ID for the connected ONEPLUS A5010 via adb."""
    try:
        result = subprocess.run(["adb", "devices"], capture_output=True, text=True, check=True)
        for line in result.stdout.splitlines():
            if line.strip().endswith("device"):
                device_id = line.split()[0]
                name_result = subprocess.run(
                    ["adb", "-s", device_id, "shell", "getprop", "ro.product.model"],
                    capture_output=True, text=True, check=True
                )
                if CONFIG["device_model"] in name_result.stdout:
                    logging.info(f"Found device: {device_id} ({CONFIG['device_model']})")
                    return device_id
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        logging.error(f"Checking for device: {e}")
        return None
    return None

def is_scrcpy_running():
    """Check if any process named 'scrcpy' is running."""
    for proc in psutil.process_iter(['name', 'exe', 'cmdline']):
        try:
            if 'scrcpy' in proc.info['name'].lower():
                return True
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
    return False

def start_scrcpy():
    """Starts the scrcpy executable from the configured path."""
    scrcpy_path = CONFIG["scrcpy_path"]
    if not os.path.exists(scrcpy_path):
        logging.error(f"scrcpy.exe not found at {scrcpy_path}")
        sys.exit(1)
    logging.info(f"Starting scrcpy from {scrcpy_path} ...")
    try:
        subprocess.Popen([scrcpy_path])
        time.sleep(2) # Give scrcpy time to start
    except Exception as e:
        logging.error(f"Failed to launch scrcpy: {e}")
        sys.exit(1)

# --- BOT CLASS ---

class ScrcpyBot:
    def __init__(self, device_id):
        self.device_id = device_id
        self.running = True
        self.templates = self._load_templates()
        self.last_2min_action_time = 0
        self.last_hourly_action_time = 0
        self.all_troops_busy = False # New state variable

    def _load_templates(self):
        """Loads all template images from paths in CONFIG into memory."""
        templates = {}
        for name, path in CONFIG["image_paths"].items():
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if img is None:
                logging.warning(f"Could not load template image: {path}")
            else:
                templates[name] = img
                logging.info(f"Loaded template: {path}")
        return templates

    # --- ADB Helper Methods ---
    def adb_command(self, *args):
        """Runs an ADB command for the current device."""
        cmd = ["adb", "-s", self.device_id] + list(args)
        try:
            return subprocess.run(cmd, capture_output=True, text=True, timeout=10, check=False)
        except FileNotFoundError:
            logging.error("`adb` command not found. Is Android SDK Platform-Tools in your PATH?")
            sys.exit(1)
        except Exception as e:
            logging.error(f"ADB command failed: {' '.join(cmd)} -> {e}")
            return None

    def adb_tap(self, x, y):
        offset = CONFIG.get("click_offset", 0)
        rand_x = x + random.randint(-offset, offset)
        rand_y = y + random.randint(-offset, offset)
        self.adb_command("shell", "input", "tap", str(int(rand_x)), str(int(rand_y)))

    def adb_swipe(self, x1, y1, x2, y2, duration_ms):
        self.adb_command("shell", "input", "swipe", str(x1), str(y1), str(x2), str(y2), str(duration_ms))

    def adb_keyevent(self, keycode):
        self.adb_command("shell", "input", "keyevent", str(keycode))

    def adb_screenshot(self):
        """Takes a screenshot and returns it as a CV2 image."""
        path = "/sdcard/screen.png"
        self.adb_command("shell", "screencap", "-p", path)
        # Use a unique local path to avoid conflicts if running multiple bots
        local_path = f"screen_{self.device_id}.png"
        self.adb_command("pull", path, local_path)
        img = cv2.imread(local_path, cv2.IMREAD_UNCHANGED)
        os.remove(local_path) # Clean up

        if CONFIG["debug_level"] >= 2 and img is not None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            debug_path = os.path.join(CONFIG["debug_dir"], f"screenshot_{timestamp}.png")
            cv2.imwrite(debug_path, img)
        return img

    # --- Image Recognition ---
    def find_template(self, template_name, screenshot, threshold=0.8, click=False):
        """Finds a template on screen and optionally clicks its center."""
        template = self.templates.get(template_name)
        if template is None or screenshot is None:
            return None

        # Ensure images are in BGR format for matching
        template_bgr = template
        if len(template.shape) == 3 and template.shape[2] == 4:
            template_bgr = cv2.cvtColor(template, cv2.COLOR_BGRA2BGR)

        screenshot_bgr = screenshot
        if len(screenshot.shape) == 3 and screenshot.shape[2] == 4:
            screenshot_bgr = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2BGR)

        mask = None
        if len(template.shape) == 3 and template.shape[2] == 4 and np.any(template[:, :, 3] < 255):
            mask = template[:, :, 3]

        result = cv2.matchTemplate(screenshot_bgr, template_bgr, cv2.TM_CCOEFF_NORMED, mask=mask)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)

        logging.debug(f"Matching {template_name}: max_val={max_val:.3f}, threshold={threshold}")
        if max_val >= threshold:
            logging.info(f"Found '{template_name}' with confidence {max_val:.2f} at {max_loc}")
            if click:
                h, w = template.shape[:2]
                tap_x = max_loc[0] + w // 2
                tap_y = max_loc[1] + h // 2
                self.adb_tap(tap_x, tap_y)
                logging.info(f"Clicked '{template_name}' at ({tap_x}, {tap_y}) with randomization")
                time.sleep(CONFIG["timing"]["action_delay"])
                if CONFIG["debug_level"] >= 2:
                    # Save annotated debug image
                    annotated = screenshot.copy()
                    cv2.rectangle(annotated, max_loc, (max_loc[0] + w, max_loc[1] + h), (0, 255, 0), 2)
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    debug_path = os.path.join(CONFIG["debug_dir"], f"match_{template_name}_{timestamp}.png")
                    cv2.imwrite(debug_path, annotated)
            return max_loc
        return None

    # --- Game Actions ---
    def go_home(self):
        self.adb_keyevent(3) # KEYCODE_HOME
        time.sleep(CONFIG["timing"]["action_delay"])

    def launch_game(self, game_name):
        self.go_home()
        coord = CONFIG["game_coords"][game_name]
        logging.info(f"Launching {game_name} at {coord}")
        self.adb_tap(coord[0], coord[1])
        time.sleep(CONFIG["timing"]["long_delay"])

    def ensure_base_view(self):
        """
        Presses ESC multiple times to attempt to return to the main base view.
        Includes a timeout to prevent infinite loops.
        """
        logging.info("Attempting to return to base view...")
        start_time = time.time()
        max_wait_time = 15 # seconds

        while time.time() - start_time < max_wait_time:
            screenshot = self.adb_screenshot()
            if screenshot is None:
                logging.warning("Failed to get screenshot for base view check.")
                time.sleep(1)
                continue

            # If there's a quit game dialog, press ESC
            elif self.find_template("quit_game", screenshot, click=False):
                logging.info("Quit game dialog found, pressing ESC.")
                self.adb_keyevent(111) # ESC key
                time.sleep(CONFIG["timing"]["action_delay"])

            # If we are already at the home screen, we're done.
            elif self.find_template("home", screenshot, click=False):
                logging.info("Base view confirmed.")
                time.sleep(CONFIG["timing"]["action_delay"])
                return

            # If we are in the world view, click the home button
            elif self.find_template("hq", screenshot, click=True):
                logging.info("In world view, clicking HQ to return to base.")
                # The click action in find_template already has a delay
                continue # Restart loop to confirm we are home
            
            # Default action is to press ESC to close any other menu
            else:
                logging.info("No specific view found, pressing ESC.")
                self.adb_keyevent(111) # ESC key
                time.sleep(CONFIG["timing"]["action_delay"])

            time.sleep(CONFIG["timing"]["action_delay"]) # Wait for UI to settle

        logging.warning("Failed to return to base view within the time limit.")

    def select_resource_level(self, level=random.randint(1, 6)):
        """Clicks the button for the specified resource level."""
        logging.info("Chose resource level: %d", level)
        coords = CONFIG["resource_levels"].get(level)
        if coords:
            self.adb_tap(coords[0], coords[1])
            time.sleep(CONFIG["timing"]["action_delay"])
            return True
        return False

    def do_2min_action(self):
        """
        Performs one attempt to send troops to gather.
        Returns True if all troops are busy, False otherwise.
        """
        logging.info("Performing gather resource action.")
        self.ensure_base_view()
        self.adb_tap(*CONFIG["action_coords"]["open_world"])
        time.sleep(CONFIG["timing"]["action_delay"])

        screenshot = self.adb_screenshot()
        if self.find_template("troops_2_of_2", screenshot, 0.95):
            logging.info("All troops are busy. Gather action complete.")
            self.adb_tap(*CONFIG["action_coords"]["open_world"]) # Close world view
            return True # Goal met

        # If troops are available, proceed to send one
        self.adb_tap(*CONFIG["action_coords"]["search_button"])
        time.sleep(CONFIG["timing"]["action_delay"])
        screenshot = self.adb_screenshot()

        # resource = random.choice(["Zent", "Wood", "Food"])
        resource = random.choice(["Food"])
        logging.info(f"Searching for {resource}.")

        if resource == "Wood":
            self.find_template("lumberyard", screenshot, click=True)
            self.select_resource_level()
        elif resource == "Food":
            start_loc = self.find_template("lumberyard", screenshot)
            found_farmyard = False
            if start_loc:
                # Try swiping left up to 3 times to find Zent
                h, w = self.templates["lumberyard"].shape[:2]
                start_x, start_y = start_loc[0] + w // 2, start_loc[1] + h // 2
                for _ in range(3):
                    self.adb_swipe(start_x, start_y, start_x - 300, start_y, CONFIG["timing"]["swipe_duration"])
                    time.sleep(CONFIG["timing"]["action_delay"])
                    screenshot = self.adb_screenshot()
                    if self.find_template("farmyard", screenshot, click=True):
                        self.select_resource_level(5)
                        found_farmyard = True
                        break
            if not found_farmyard:
                logging.warning("Could not find found_farmyard after swiping.")
                self.select_resource_level() # Attempt to select anyway
        elif resource == "Zent":
            start_loc = self.find_template("lumberyard", screenshot)
            found_zentyard = False
            if start_loc:
                # Try swiping left up to 3 times to find Zent
                h, w = self.templates["lumberyard"].shape[:2]
                start_x, start_y = start_loc[0] + w // 2, start_loc[1] + h // 2
                for _ in range(3):
                    self.adb_swipe(start_x, start_y, start_x - 300, start_y, CONFIG["timing"]["swipe_duration"])
                    time.sleep(CONFIG["timing"]["action_delay"])
                    screenshot = self.adb_screenshot()
                    if self.find_template("zentyard", screenshot, click=True):
                        self.select_resource_level(5)
                        found_zentyard = True
                        break
            if not found_zentyard:
                logging.warning("Could not find zentyard after swiping.")
                self.select_resource_level() # Attempt to select anyway

        self.adb_tap(*CONFIG["action_coords"]["gather_search"])
        time.sleep(CONFIG["timing"]["action_delay"])
        self.adb_tap(*CONFIG["action_coords"]["middle_screen"])
        time.sleep(CONFIG["timing"]["action_delay"])
        self.adb_tap(*CONFIG["action_coords"]["gather_button"])
        time.sleep(CONFIG["timing"]["action_delay"])
        self.adb_tap(*CONFIG["action_coords"]["gather_confirm_march"])
        time.sleep(CONFIG["timing"]["long_delay"]) # Wait for march to start
        self.ensure_base_view()
        return False # Troops were sent, but maybe more are available

    def do_hourly_action(self):
        logging.info("Performing hourly action: Research and alliance help.")
        self.ensure_base_view()
        self.adb_tap(*CONFIG["action_coords"]["alliance"])
        time.sleep(CONFIG["timing"]["action_delay"])
        self.adb_tap(*CONFIG["action_coords"]["alliance_techs"])
        time.sleep(CONFIG["timing"]["action_delay"])

        screenshot = self.adb_screenshot()
        if self.find_template("research_recommended", screenshot, click=True):
            time.sleep(CONFIG["timing"]["action_delay"])
            for _ in range(20):
                self.adb_tap(*CONFIG["action_coords"]["research_help"])
                time.sleep(CONFIG["timing"]["fast_click_delay"])
            self.adb_keyevent(111) # ESC
            time.sleep(2)
            self.adb_keyevent(111) # ESC
            time.sleep(2)

        self.adb_tap(*CONFIG["action_coords"]["alliance_gifts"])
        time.sleep(2)
        self.adb_tap(*CONFIG["action_coords"]["alliance_gifts_claimall"])
        time.sleep(2)
        for _ in range(3):
            self.adb_keyevent(111) # ESC
            time.sleep(1)

    def run_one_cycle(self, game_name):
        """Performs one cycle of passive actions and checks for periodic tasks."""
        logging.info(f"--- New Cycle for {game_name} ---")
        self.ensure_base_view() # Return to a known state
        now = time.time()
        screenshot = self.adb_screenshot()
        if screenshot is None:
            logging.warning("Failed to get screenshot. Skipping cycle.")
            return

        # --- Passive Actions (run every cycle) ---
        clicked_something = False
        passive_checks = ["exp", "food", "electric", "zent", "wood", "fuel", "help", "raider", "assulter", "shooter"]
        for name in passive_checks:
            if self.find_template(name, screenshot, click=True):
                clicked_something = True
                time.sleep(CONFIG["timing"]["action_delay"])
                # Get a fresh screenshot if we clicked something
                screenshot = self.adb_screenshot()
                if screenshot is None: break
                
        # do research 
        if self.find_template("research_free", screenshot, click=True, threshold=0.9):
            time.sleep(CONFIG["timing"]["long_delay"])
            self.find_template("research_recommended2", self.adb_screenshot(), click=True, threshold=0.95)
            time.sleep(CONFIG["timing"]["action_delay"])
            self.find_template("research_confirm", self.adb_screenshot(), click=True)
            self.ensure_base_view()

            

        # Train troops if a bay is empty and the setting is enabled for this game
        if CONFIG["game_settings"].get(game_name, {}).get("train_troops", False):
            if self.find_template("empty_troops", screenshot, click=True):
                logging.info(f"Training troops for {game_name}.")
                self.adb_tap(*CONFIG["action_coords"]["train_troops_confirm"])
                time.sleep(CONFIG["timing"]["action_delay"])
                self.adb_tap(*CONFIG["action_coords"]["train_troops_select"])
                time.sleep(CONFIG["timing"]["action_delay"])
                self.adb_keyevent(111) # ESC
        
        if self.find_template("collect_8hrs", screenshot, click=True):
            logging.info("Collecting 8hrs chest.")
            self.adb_tap(*CONFIG["action_coords"]["8hrs_chest_collect"])
            time.sleep(CONFIG["timing"]["action_delay"])
            self.ensure_base_view()


        # --- Periodic Actions ---
        if now - self.last_2min_action_time >= CONFIG["timing"]["periodic_action_interval_2min"]:
            # Reset the busy flag on a timer, so we check for returning troops
            self.all_troops_busy = False
            self.last_2min_action_time = now 
        
        # If it's time to act and we know troops aren't busy, try to send them.
        if not self.all_troops_busy:
            logging.info("Troops are not all busy, attempting to send more.")
            self.all_troops_busy = self.do_2min_action()

        if now - self.last_hourly_action_time >= CONFIG["timing"]["periodic_action_interval_hourly"]:
            self.do_hourly_action()
            self.last_hourly_action_time = now

        logging.info(f"Cycle finished. Sleeping for {CONFIG['timing']['interval']}s.")
        time.sleep(CONFIG["timing"]["interval"])

# --- MAIN EXECUTION ---

def main():
    parser = argparse.ArgumentParser(description="A bot for LastZ using scrcpy.")
    parser.add_argument(
        "-d", "--debug",
        type=int,
        choices=[0, 1, 2],
        default=CONFIG["debug_level"],
        help="Set debug level: 0=Quiet, 1=Info, 2=Debug (saves screenshots)."
    )
    args = parser.parse_args()
    CONFIG["debug_level"] = args.debug

    # Setup logging
    log_level = logging.WARNING
    if CONFIG["debug_level"] == 1:
        log_level = logging.INFO
    elif CONFIG["debug_level"] >= 2:
        log_level = logging.DEBUG
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    os.makedirs(CONFIG["debug_dir"], exist_ok=True)

    if not is_scrcpy_running():
        start_scrcpy()
    else:
        logging.info("scrcpy is already running.")

    device_id = get_device_id()
    if not device_id:
        logging.error("No suitable Android device found. Exiting.")
        sys.exit(1)

    bot = ScrcpyBot(device_id)
    game_names = list(CONFIG["game_coords"].keys())
    rotation_seconds = CONFIG["timing"]["rotation_minutes"] * 60

    try:
        while True:
            for game in game_names:
                logging.info(f"--- Rotating to {game} for {CONFIG['timing']['rotation_minutes']} minutes ---")
                bot.launch_game(game)
                
                start_time = time.time()
                while time.time() - start_time < rotation_seconds:
                    bot.run_one_cycle(game)
                
                logging.info(f"Finished with {game}. Rotating to next app.")

    except KeyboardInterrupt:
        logging.info("Bot stopped by user. Exiting.")
    except Exception as e:
        logging.critical(f"An unexpected error occurred: {e}", exc_info=True)
    finally:
        sys.exit(0)

if __name__ == "__main__":
    main()


