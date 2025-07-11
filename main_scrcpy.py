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
import json
from logging.handlers import RotatingFileHandler

# --- CONFIGURATION ---
CONFIG = {
    "device_model": "ONEPLUS A5010",
    "scrcpy_path": os.path.join(os.path.dirname(os.path.abspath(__file__)), 'scrcpy', 'scrcpy.exe'),
    "debug_dir": "debug",
    "logs_dir": "logs",
    "state_file": "bot_state.json",
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
        "game4": {"train_troops": True}
        #"game5": {"train_troops": True}
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
        "8hrs_chest_collect": (558, 1594),
        "clear_all": (532, 1933) # NOTE: Adjust these coordinates for your device's "Clear All" button
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
        # New action intervals for state tracking
        "train_troops_check_interval": 300,  # Check every 5 minutes
        "research_check_interval": 1800,     # Check every 30 minutes
        "collect_8hrs_check_interval": 600,  # Check every 10 minutes
        "passive_checks_interval": 60,       # Check passive items every minute
    },
    "click_offset": 1 # Max random pixels to add to each click coordinate
}

# --- STATE MANAGEMENT ---

class StateManager:
    def __init__(self, state_file):
        self.state_file = state_file
        self.logger = logging.getLogger('state')
        self.state = self._load_state()
        
    def _load_state(self):
        """Load state from file or create default state."""
        try:
            if os.path.exists(self.state_file):
                with open(self.state_file, 'r') as f:
                    state = json.load(f)
                    self.logger.info(f"Loaded state from {self.state_file}")
                    # Clean up any inconsistent keys from old versions
                    self._cleanup_state_keys(state)
                    return state
        except (json.JSONDecodeError, IOError) as e:
            self.logger.warning(f"Failed to load state from {self.state_file}: {e}")
        
        # Return default state
        default_state = {
            "last_train_troops_check": 0,
            "last_research_check": 0,
            "last_collect_8hrs_check": 0,
            "last_passive_checks_check": 0,
            "last_2min_action_time": 0,
            "last_hourly_action_time": 0,
            "troops_training": False,
            "research_in_progress": False,
            "all_troops_busy": False,
            "current_game": None,
            "current_rotation_start": 0,
            "bot_start_time": time.time(),
            "total_cycles": 0,
            "total_rotations": 0,
            "actions_performed": {
                "passive_actions": 0,
                "research_started": 0,
                "research_completed": 0,
                "troops_trained": 0,
                "chests_collected": 0,
                "gather_attempts": 0,
                "hourly_actions": 0
            }
        }
        self.logger.info("Created default state")
        return default_state
    
    def _cleanup_state_keys(self, state):
        """Clean up inconsistent state keys from older versions."""
        # Fix the passive checks key inconsistency
        if "last_passive_checks" in state and "last_passive_checks_check" not in state:
            state["last_passive_checks_check"] = state.pop("last_passive_checks")
        elif "last_passive_checks" in state and "last_passive_checks_check" in state:
            # Remove the old inconsistent key
            state.pop("last_passive_checks")
        
        # Ensure all required keys exist
        if "current_rotation_start" not in state:
            state["current_rotation_start"] = 0
        if "total_rotations" not in state:
            state["total_rotations"] = 0

    def save_state(self):
        """Save current state to file."""
        try:
            # Create backup of existing state
            if os.path.exists(self.state_file):
                backup_file = f"{self.state_file}.backup"
                # Remove existing backup if it exists
                if os.path.exists(backup_file):
                    os.remove(backup_file)
                os.rename(self.state_file, backup_file)
            
            with open(self.state_file, 'w') as f:
                json.dump(self.state, f, indent=2)
            
            self.logger.debug(f"Saved state to {self.state_file}")
            
        except IOError as e:
            self.logger.error(f"Failed to save state: {e}")
    
    def get(self, key, default=None):
        """Get a value from state."""
        return self.state.get(key, default)
    
    def set(self, key, value):
        """Set a value in state."""
        self.state[key] = value
        self.save_state()
    
    def update(self, updates):
        """Update multiple values in state."""
        self.state.update(updates)
        self.save_state()
    
    def increment_counter(self, key):
        """Increment a counter in the actions_performed section."""
        if "actions_performed" not in self.state:
            self.state["actions_performed"] = {}
        
        current_value = self.state["actions_performed"].get(key, 0)
        self.state["actions_performed"][key] = current_value + 1
        self.save_state()
    
    def should_check_action(self, action_name, interval_key):
        """Check if enough time has passed since the last action check."""
        now = time.time()
        last_check = self.get(f"last_{action_name}_check", 0)
        interval = CONFIG["timing"][interval_key]
        return now - last_check >= interval
    
    def update_action_state(self, action_name, **kwargs):
        """Update the state for a specific action."""
        updates = {f"last_{action_name}_check": time.time()}
        updates.update(kwargs)
        self.update(updates)
    
    def start_game_rotation(self, game_name):
        """Mark the start of a new game rotation."""
        self.update({
            "current_game": game_name,
            "current_rotation_start": time.time()
        })
        self.logger.info(f"Started rotation for {game_name}")
    
    def end_game_rotation(self, game_name):
        """Mark the end of a game rotation."""
        rotation_duration = time.time() - self.get("current_rotation_start", time.time())
        self.update({
            "current_game": None,
            "current_rotation_start": 0,
            "total_rotations": self.get("total_rotations", 0) + 1
        })
        self.logger.info(f"Ended rotation for {game_name} after {rotation_duration:.1f}s")
    
    def get_stats(self):
        """Get bot statistics."""
        now = time.time()
        start_time = self.get("bot_start_time", now)
        uptime = now - start_time
        current_rotation_start = self.get("current_rotation_start", 0)
        current_rotation_duration = now - current_rotation_start if current_rotation_start > 0 else 0
        
        return {
            "uptime_seconds": uptime,
            "uptime_formatted": str(datetime.timedelta(seconds=int(uptime))),
            "total_cycles": self.get("total_cycles", 0),
            "total_rotations": self.get("total_rotations", 0),
            "current_game": self.get("current_game", "None"),
            "current_rotation_duration": current_rotation_duration,
            "actions_performed": self.get("actions_performed", {}),
            "troops_busy": self.get("all_troops_busy", False),
            "research_in_progress": self.get("research_in_progress", False)
        }

# --- UTILITY FUNCTIONS ---

def setup_logging(debug_level):
    """Set up comprehensive logging with file rotation and console output."""
    # Create logs directory
    os.makedirs(CONFIG["logs_dir"], exist_ok=True)
    
    # Configure log levels
    log_levels = {
        0: logging.WARNING,
        1: logging.INFO,
        2: logging.DEBUG
    }
    log_level = log_levels.get(debug_level, logging.INFO)
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    console_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)-8s - %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # Set up root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    # File handler - rotating log files
    log_file = os.path.join(CONFIG["logs_dir"], "lastz_bot.log")
    file_handler = RotatingFileHandler(
        log_file, 
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    file_handler.setLevel(logging.DEBUG)  # Always log everything to file
    file_handler.setFormatter(detailed_formatter)
    root_logger.addHandler(file_handler)
    
    # Create specialized loggers
    loggers = {
        'bot': logging.getLogger('bot'),
        'image': logging.getLogger('image'),
        'adb': logging.getLogger('adb'),
        'game': logging.getLogger('game'),
        'state': logging.getLogger('state')
    }
    
    return loggers

def get_device_id():
    """Returns the device ID for the connected ONEPLUS A5010 via adb."""
    logger = logging.getLogger('adb')
    try:
        result = subprocess.run(["adb", "devices"], capture_output=True, text=True, check=True)
        logger.debug(f"ADB devices output: {result.stdout}")
        
        for line in result.stdout.splitlines():
            if line.strip().endswith("device"):
                device_id = line.split()[0]
                logger.debug(f"Found device ID: {device_id}")
                
                name_result = subprocess.run(
                    ["adb", "-s", device_id, "shell", "getprop", "ro.product.model"],
                    capture_output=True, text=True, check=True
                )
                device_model = name_result.stdout.strip()
                logger.debug(f"Device model: {device_model}")
                
                if CONFIG["device_model"] in device_model:
                    logger.info(f"Connected to device: {device_id} ({device_model})")
                    return device_id
                else:
                    logger.warning(f"Device {device_id} model '{device_model}' does not match expected '{CONFIG['device_model']}'")
                    
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        logger.error(f"Failed to check for device: {e}")
        return None
    
    logger.warning("No suitable Android device found")
    return None

def is_scrcpy_running():
    """Check if any process named 'scrcpy' is running."""
    logger = logging.getLogger('bot')
    for proc in psutil.process_iter(['name', 'exe', 'cmdline']):
        try:
            if 'scrcpy' in proc.info['name'].lower():
                logger.debug(f"Found scrcpy process: {proc.info}")
                return True
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
    return False

def start_scrcpy():
    """Starts the scrcpy executable from the configured path."""
    logger = logging.getLogger('bot')
    scrcpy_path = CONFIG["scrcpy_path"]
    
    if not os.path.exists(scrcpy_path):
        logger.critical(f"scrcpy.exe not found at {scrcpy_path}")
        sys.exit(1)
        
    logger.info(f"Starting scrcpy from {scrcpy_path}")
    try:
        subprocess.Popen([scrcpy_path])
        logger.debug("scrcpy process started, waiting for initialization...")
        time.sleep(2)
        logger.info("scrcpy startup complete")
    except Exception as e:
        logger.critical(f"Failed to launch scrcpy: {e}")
        sys.exit(1)

# --- BOT CLASS ---

class ScrcpyBot:
    def __init__(self, device_id, state_manager):
        self.device_id = device_id
        self.running = True
        self.state = state_manager
        
        # Initialize loggers first
        self.logger = logging.getLogger('bot')
        self.image_logger = logging.getLogger('image')
        self.adb_logger = logging.getLogger('adb')
        self.game_logger = logging.getLogger('game')
        self.state_logger = logging.getLogger('state')
        
        # Now load templates (which uses image_logger)
        self.templates = self._load_templates()
        
        self.logger.info(f"Bot initialized for device: {device_id}")
        
        # Log current state
        stats = self.state.get_stats()
        self.state_logger.info(f"Bot statistics: {stats}")

    def _load_templates(self):
        """Loads all template images from paths in CONFIG into memory."""
        templates = {}
        failed_templates = []
        
        for name, path in CONFIG["image_paths"].items():
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if img is None:
                self.image_logger.warning(f"Could not load template image: {path}")
                failed_templates.append(name)
            else:
                templates[name] = img
                self.image_logger.debug(f"Loaded template: {name} from {path} (shape: {img.shape})")
        
        self.image_logger.info(f"Loaded {len(templates)} templates successfully")
        if failed_templates:
            self.image_logger.warning(f"Failed to load {len(failed_templates)} templates: {failed_templates}")
            
        return templates

    # --- ADB Helper METHODS ---
    def adb_command(self, *args):
        """Runs an ADB command for the current device."""
        cmd = ["adb", "-s", self.device_id] + list(args)
        cmd_str = ' '.join(cmd)
        
        try:
            self.adb_logger.debug(f"Executing: {cmd_str}")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10, check=False)
            
            if result.returncode != 0:
                self.adb_logger.warning(f"Command failed with code {result.returncode}: {cmd_str}")
                if result.stderr:
                    self.adb_logger.warning(f"stderr: {result.stderr}")
            else:
                self.adb_logger.debug(f"Command successful: {cmd_str}")
                
            return result
            
        except FileNotFoundError:
            self.adb_logger.critical("`adb` command not found. Is Android SDK Platform-Tools in your PATH?")
            sys.exit(1)
        except subprocess.TimeoutExpired:
            self.adb_logger.error(f"Command timed out: {cmd_str}")
            return None
        except Exception as e:
            self.adb_logger.error(f"ADB command failed: {cmd_str} -> {e}")
            return None

    def adb_tap(self, x, y):
        offset = CONFIG.get("click_offset", 0)
        rand_x = x + random.randint(-offset, offset)
        rand_y = y + random.randint(-offset, offset)
        self.adb_logger.debug(f"Tapping at ({rand_x}, {rand_y}) [original: ({x}, {y})]")
        self.adb_command("shell", "input", "tap", str(int(rand_x)), str(int(rand_y)))

    def adb_swipe(self, x1, y1, x2, y2, duration_ms):
        self.adb_logger.debug(f"Swiping from ({x1}, {y1}) to ({x2}, {y2}) in {duration_ms}ms")
        self.adb_command("shell", "input", "swipe", str(x1), str(y1), str(x2), str(y2), str(duration_ms))

    def adb_keyevent(self, keycode):
        self.adb_logger.debug(f"Sending keyevent: {keycode}")
        self.adb_command("shell", "input", "keyevent", str(keycode))

    def adb_screenshot(self):
        """Takes a screenshot and returns it as a CV2 image."""
        path = "/sdcard/screen.png"
        self.adb_command("shell", "screencap", "-p", path)
        # Use a unique local path to avoid conflicts if running multiple bots
        local_path = f"screen_{self.device_id}.png"
        self.adb_command("pull", path, local_path)
        img = cv2.imread(local_path, cv2.IMREAD_UNCHANGED)
        
        try:
            os.remove(local_path) # Clean up
        except FileNotFoundError:
            pass

        if CONFIG["debug_level"] >= 2 and img is not None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            debug_path = os.path.join(CONFIG["debug_dir"], f"screenshot_{timestamp}.png")
            cv2.imwrite(debug_path, img)
            self.adb_logger.debug(f"Saved debug screenshot: {debug_path}")
        return img

    # --- Image Recognition ---
    def find_template(self, template_name, screenshot, threshold=0.05, click=False):
        """
        Finds a template on screen and optionally clicks its center.
        Uses TM_SQDIFF_NORMED, where 0 is a perfect match.
        """
        template = self.templates.get(template_name)
        if template is None:
            self.image_logger.warning(f"Template '{template_name}' not found in loaded templates")
            return None
        if screenshot is None:
            self.image_logger.warning(f"Screenshot is None for template '{template_name}'")
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
            self.image_logger.debug(f"Using mask for template '{template_name}'")

        # Use TM_SQDIFF_NORMED: 0 = perfect match, 1 = worst match
        result = cv2.matchTemplate(screenshot_bgr, template_bgr, cv2.TM_SQDIFF_NORMED, mask=mask)
        min_val, _, min_loc, _ = cv2.minMaxLoc(result)

        self.image_logger.debug(f"Template matching '{template_name}': confidence={min_val:.4f}, threshold={threshold}")
        
        if min_val <= threshold:
            self.image_logger.info(f"Found '{template_name}' with confidence {min_val:.4f} at {min_loc}")
            if click:
                h, w = template.shape[:2]
                tap_x = min_loc[0] + w // 2
                tap_y = min_loc[1] + h // 2
                self.adb_tap(tap_x, tap_y)
                self.game_logger.info(f"Clicked '{template_name}' at ({tap_x}, {tap_y})")
                time.sleep(CONFIG["timing"]["action_delay"])
                
                if CONFIG["debug_level"] >= 2:
                    # Save annotated debug image
                    annotated = screenshot.copy()
                    cv2.rectangle(annotated, min_loc, (min_loc[0] + w, min_loc[1] + h), (0, 255, 0), 2)
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    debug_path = os.path.join(CONFIG["debug_dir"], f"match_{template_name}_{timestamp}.png")
                    cv2.imwrite(debug_path, annotated)
                    self.image_logger.debug(f"Saved annotated match image: {debug_path}")
                    
            return min_loc
        else:
            self.image_logger.debug(f"Template '{template_name}' not found (confidence {min_val:.4f} > threshold {threshold})")
            return None

    # --- Game Actions ---
    def go_home(self):
        self.adb_keyevent(3) # KEYCODE_HOME
        time.sleep(CONFIG["timing"]["action_delay"])

    def launch_game(self, game_name):
        """Launch a specific game and wait for it to load."""
        self.go_home()
        coord = CONFIG["game_coords"][game_name]
        self.game_logger.info(f"Launching {game_name} at {coord}")
        self.adb_tap(coord[0], coord[1])
        time.sleep(CONFIG["timing"]["long_delay"])
        
        # Update state to track game rotation
        self.state.start_game_rotation(game_name)

    def ensure_base_view(self):
        """
        Presses ESC multiple times to attempt to return to the main base view.
        Includes a timeout to prevent infinite loops.
        """
        self.game_logger.info("Attempting to return to base view...")
        start_time = time.time()
        max_wait_time = 15 # seconds

        while time.time() - start_time < max_wait_time:
            screenshot = self.adb_screenshot()
            if screenshot is None:
                self.game_logger.warning("Failed to get screenshot for base view check.")
                time.sleep(1)
                continue

            # If there's a quit game dialog, press ESC
            if self.find_template("quit_game", screenshot, click=False):
                self.game_logger.info("Quit game dialog found, pressing ESC.")
                self.adb_keyevent(111) # ESC key
                time.sleep(CONFIG["timing"]["action_delay"])

            # If we are already at the home screen, we're done.
            elif self.find_template("home", screenshot, click=False):
                self.game_logger.info("Base view confirmed.")
                time.sleep(CONFIG["timing"]["action_delay"])
                return

            # If we are in the world view, click the home button
            elif self.find_template("hq", screenshot, click=True):
                self.game_logger.info("In world view, clicking HQ to return to base.")
                # The click action in find_template already has a delay
                continue # Restart loop to confirm we are home
            
            # Default action is to press ESC to close any other menu
            else:
                self.game_logger.debug("No specific view found, pressing ESC.")
                self.adb_keyevent(111) # ESC key
                time.sleep(CONFIG["timing"]["action_delay"])

            time.sleep(CONFIG["timing"]["action_delay"]) # Wait for UI to settle

        self.game_logger.warning("Failed to return to base view within the time limit.")

    def close_all_apps(self):
        """Closes all recent applications using hardcoded coordinates with multiple attempts."""
        self.game_logger.info("Closing all recent applications.")
        max_attempts = 3
        
        for attempt in range(max_attempts):
            try:
                self.game_logger.debug(f"Attempt {attempt + 1} to close all apps...")
                
                # Press the recent apps button
                self.adb_keyevent(187)  # KEYCODE_APP_SWITCH
                time.sleep(CONFIG["timing"]["action_delay"] + 1)  # Extra time for animation
                
                # Tap the 'Clear All' button using hardcoded coordinates
                self.adb_tap(*CONFIG["action_coords"]["clear_all"])
                time.sleep(CONFIG["timing"]["action_delay"] + 1)  # Extra time for clearing
                
                # Press home to ensure we're back to launcher
                self.adb_keyevent(3)  # KEYCODE_HOME
                time.sleep(CONFIG["timing"]["action_delay"])
                
                # Verify we're at home by pressing home again
                self.adb_keyevent(3)  # KEYCODE_HOME
                time.sleep(CONFIG["timing"]["action_delay"])
                
                self.game_logger.info(f"Successfully closed all apps on attempt {attempt + 1}.")
                return True
                
            except Exception as e:
                self.game_logger.warning(f"Attempt {attempt + 1} failed to close apps: {e}")
                if attempt < max_attempts - 1:
                    time.sleep(2)  # Wait before retry
                else:
                    self.game_logger.error("All attempts to close apps failed.")
                    
        # Fallback: just go home
        self.game_logger.info("Falling back to home screen.")
        self.go_home()
        return False

    def select_resource_level(self, level=None):
        """Clicks the button for the specified resource level."""
        if level is None:
            level = random.randint(1, 6)
        
        self.game_logger.debug(f"Selecting resource level: {level}")
        coords = CONFIG["resource_levels"].get(level)
        if coords:
            self.adb_tap(coords[0], coords[1])
            time.sleep(CONFIG["timing"]["action_delay"])
            return True
        else:
            self.game_logger.warning(f"Invalid resource level: {level}")
            return False

    def search_and_gather_resource(self, resource_type):
        """Search for and gather a specific resource type."""
        self.game_logger.info(f"Searching for {resource_type}")
        
        if resource_type == "Wood":
            if self.find_template("lumberyard", self.adb_screenshot(), click=True):
                self.select_resource_level()
                return True
                
        elif resource_type in ["Food", "Zent"]:
            target_template = "farmyard" if resource_type == "Food" else "zentyard"
            start_loc = self.find_template("lumberyard", self.adb_screenshot())
            
            if start_loc:
                h, w = self.templates["lumberyard"].shape[:2]
                start_x, start_y = start_loc[0] + w // 2, start_loc[1] + h // 2
                
                for attempt in range(3):
                    self.adb_swipe(start_x, start_y, start_x - 300, start_y, CONFIG["timing"]["swipe_duration"])
                    time.sleep(CONFIG["timing"]["action_delay"])
                    
                    if self.find_template(target_template, self.adb_screenshot(), click=True):
                        self.select_resource_level()  # Use random level instead of hardcoded 5
                        return True
                        
                self.game_logger.warning(f"Could not find {target_template} after swiping.")
                self.select_resource_level() # Attempt to select anyway
                return False
        
        return False

    def do_2min_action(self):
        """
        Performs one attempt to send troops to gather.
        Returns True if all troops are busy, False otherwise.
        """
        self.game_logger.info("Performing gather resource action.")
        self.ensure_base_view()
        self.adb_tap(*CONFIG["action_coords"]["open_world"])
        time.sleep(CONFIG["timing"]["action_delay"])

        screenshot = self.adb_screenshot()
        if self.find_template("troops_2_of_2", screenshot, threshold=0.05):
            self.game_logger.info("All troops are busy. Gather action complete.")
            self.adb_tap(*CONFIG["action_coords"]["open_world"]) # Close world view
            self.state.set("all_troops_busy", True)
            return True # Goal met

        # If troops are available, proceed to send one
        self.adb_tap(*CONFIG["action_coords"]["search_button"])
        time.sleep(CONFIG["timing"]["action_delay"])

        # Search for resources
        resource = random.choice(["Food"])  # Can be expanded later
        if self.search_and_gather_resource(resource):
            self.game_logger.info(f"Successfully found and selected {resource}")
        else:
            self.game_logger.warning(f"Failed to find {resource}, continuing anyway")

        # Complete the gathering process
        self.adb_tap(*CONFIG["action_coords"]["gather_search"])
        time.sleep(CONFIG["timing"]["action_delay"])
        self.adb_tap(*CONFIG["action_coords"]["middle_screen"])
        time.sleep(CONFIG["timing"]["action_delay"])
        self.adb_tap(*CONFIG["action_coords"]["gather_button"])
        time.sleep(CONFIG["timing"]["action_delay"])
        self.adb_tap(*CONFIG["action_coords"]["gather_confirm_march"])
        time.sleep(CONFIG["timing"]["long_delay"]) # Wait for march to start
        self.ensure_base_view()
        
        # Update state
        self.state.increment_counter("gather_attempts")
        self.state.set("all_troops_busy", False)
        return False # Troops were sent, but maybe more are available

    def do_hourly_action(self):
        """Perform alliance help and gift collection."""
        self.game_logger.info("Performing hourly action: Research and alliance help.")
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
        
        # Close all menus
        for _ in range(3):
            self.adb_keyevent(111) # ESC
            time.sleep(1)
        
        # Update state
        self.state.increment_counter("hourly_actions")
        self.state.set("last_hourly_action_time", time.time())

    def run_one_cycle(self, game_name):
        """Performs one cycle of passive actions and checks for periodic tasks."""
        cycle_start = time.time()
        self.game_logger.info(f"Starting cycle for {game_name}")
        
        self.ensure_base_view()
        now = time.time()
        screenshot = self.adb_screenshot()
        
        if screenshot is None:
            self.logger.warning("Failed to get screenshot. Skipping cycle.")
            return

        actions_performed = []

        # --- Passive Actions ---
        if self.state.should_check_action("passive_checks", "passive_checks_interval"):
            self.state_logger.info("Checking passive actions...")
            passive_checks = ["exp", "food", "electric", "zent", "wood", "fuel", "help", "raider", "assulter", "shooter"]
            
            for name in passive_checks:
                if self.find_template(name, screenshot, click=True):
                    actions_performed.append(f"passive_{name}")
                    self.state.increment_counter("passive_actions")
                    time.sleep(CONFIG["timing"]["action_delay"])
                    screenshot = self.adb_screenshot()
                    if screenshot is None: break
            
            self.state.update_action_state("passive_checks")
            
        # --- Research Actions ---
        if self.state.should_check_action("research", "research_check_interval"):
            self.state_logger.info("Checking research actions...")
            if self.find_template("research_free", screenshot, click=True, threshold=0.05):
                time.sleep(CONFIG["timing"]["long_delay"])
                if self.find_template("research_complete", self.adb_screenshot(), click=True, threshold=0.05):
                    self.game_logger.info("Research completed and collected")
                    self.state.update_action_state("research", research_in_progress=False)
                    self.state.increment_counter("research_completed")
                    actions_performed.append("research_complete")
                else:
                    self.find_template("research_recommended2", self.adb_screenshot(), click=True, threshold=0.05)
                    time.sleep(CONFIG["timing"]["action_delay"])
                    self.find_template("research_confirm", self.adb_screenshot(), click=True, threshold=0.05)
                    self.state.update_action_state("research", research_in_progress=True)
                    self.state.increment_counter("research_started")
                    self.ensure_base_view()
                    actions_performed.append("research_started")
            else:
                self.state.update_action_state("research")

        # --- Train Troops ---
        if (CONFIG["game_settings"].get(game_name, {}).get("train_troops", False) and 
            self.state.should_check_action("train_troops", "train_troops_check_interval")):
            self.state_logger.info("Checking troop training...")
            if self.find_template("empty_troops", screenshot, click=True):
                self.game_logger.info(f"Training troops for {game_name}")
                self.adb_tap(*CONFIG["action_coords"]["train_troops_confirm"])
                time.sleep(CONFIG["timing"]["action_delay"])
                self.adb_tap(*CONFIG["action_coords"]["train_troops_select"])
                time.sleep(CONFIG["timing"]["action_delay"])
                self.adb_keyevent(111)
                self.state.update_action_state("train_troops", troops_training=True)
                self.state.increment_counter("troops_trained")
                actions_performed.append("train_troops")
            else:
                self.state.update_action_state("train_troops")
        
        # --- Collect 8hrs chest ---
        if self.state.should_check_action("collect_8hrs", "collect_8hrs_check_interval"):
            self.state_logger.info("Checking 8hrs chest collection...")
            if self.find_template("collect_8hrs", screenshot, click=True):
                self.game_logger.info("Collecting 8hrs chest")
                self.adb_tap(*CONFIG["action_coords"]["8hrs_chest_collect"])
                time.sleep(CONFIG["timing"]["action_delay"])
                self.ensure_base_view()
                self.state.increment_counter("chests_collected")
                actions_performed.append("collect_8hrs")
            self.state.update_action_state("collect_8hrs")

        # --- Periodic Actions ---
        if now - self.state.get("last_2min_action_time", 0) >= CONFIG["timing"]["periodic_action_interval_2min"]:
            self.state.update({"all_troops_busy": False, "last_2min_action_time": now})
            self.state_logger.info("Reset troop busy flag")
        
        if not self.state.get("all_troops_busy", False):
            self.state_logger.info("Attempting to send troops to gather")
            all_troops_busy = self.do_2min_action()
            if all_troops_busy:
                actions_performed.append("gather_troops_busy")
            else:
                actions_performed.append("gather_troops_sent")

        if now - self.state.get("last_hourly_action_time", 0) >= CONFIG["timing"]["periodic_action_interval_hourly"]:
            self.do_hourly_action()
            actions_performed.append("hourly_action")

        # Update cycle counter
        self.state.set("total_cycles", self.state.get("total_cycles", 0) + 1)
        
        cycle_duration = time.time() - cycle_start
        self.game_logger.info(f"Cycle completed in {cycle_duration:.2f}s. Actions: {actions_performed}")
        
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
    parser.add_argument(
        "--reset-state",
        action="store_true",
        help="Reset bot state and start fresh."
    )
    args = parser.parse_args()
    CONFIG["debug_level"] = args.debug

    # Setup enhanced logging
    loggers = setup_logging(CONFIG["debug_level"])
    main_logger = logging.getLogger('bot')
    
    # Create directories
    os.makedirs(CONFIG["debug_dir"], exist_ok=True)
    
    # Handle state reset
    if args.reset_state:
        state_file = CONFIG["state_file"]
        if os.path.exists(state_file):
            os.remove(state_file)
            main_logger.info("Bot state reset")
        else:
            main_logger.info("No state file found to reset")
    
    # Initialize state manager
    state_manager = StateManager(CONFIG["state_file"])
    
    main_logger.info("=" * 60)
    main_logger.info(f"LastZ Bot starting - Debug Level: {CONFIG['debug_level']}")
    main_logger.info("=" * 60)
    
    # Log bot statistics
    stats = state_manager.get_stats()
    main_logger.info(f"Bot uptime: {stats['uptime_formatted']}")
    main_logger.info(f"Total cycles: {stats['total_cycles']}")
    main_logger.info(f"Actions performed: {stats['actions_performed']}")

    if not is_scrcpy_running():
        start_scrcpy()
    else:
        main_logger.info("scrcpy is already running")

    device_id = get_device_id()
    if not device_id:
        main_logger.critical("No suitable Android device found. Exiting.")
        sys.exit(1)

    bot = ScrcpyBot(device_id, state_manager)
    
    # Close all apps at startup
    main_logger.info("Performing startup cleanup...")
    bot.close_all_apps()
    time.sleep(2)
    
    game_names = list(CONFIG["game_coords"].keys())
    rotation_seconds = CONFIG["timing"]["rotation_minutes"] * 60
    
    main_logger.info(f"Starting game rotation: {game_names}")
    main_logger.info(f"Rotation interval: {CONFIG['timing']['rotation_minutes']} minutes per game")

    try:
        while True:
            for game in game_names:
                main_logger.info(f"{'='*20} Starting {game} {'='*20}")
                bot.launch_game(game)
                
                start_time = time.time()
                cycle_count = 0
                
                while time.time() - start_time < rotation_seconds:
                    cycle_count += 1
                    main_logger.debug(f"Cycle {cycle_count} for {game}")
                    bot.run_one_cycle(game)
                
                elapsed = time.time() - start_time
                main_logger.info(f"Finished {game} after {elapsed:.1f}s ({cycle_count} cycles)")
                
                # End the rotation and clear current game
                state_manager.end_game_rotation(game)
                
                main_logger.info("Closing apps before rotation...")
                bot.close_all_apps()

    except KeyboardInterrupt:
        main_logger.info("Bot stopped by user (Ctrl+C)")
    except Exception as e:
        main_logger.critical(f"Unexpected error occurred: {e}", exc_info=True)
    finally:
        # Log final statistics
        stats = state_manager.get_stats()
        main_logger.info(f"Final statistics: {stats}")
        main_logger.info("Bot shutdown complete")
        sys.exit(0)

if __name__ == "__main__":
    main()


