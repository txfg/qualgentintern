import os
import time
import base64
import json
import re
import xml.etree.ElementTree as ET
from typing import Optional, Tuple
from dotenv import load_dotenv
from ppadb.client import Client as AdbClient
import google.generativeai as genai
from PIL import Image, ImageDraw, ImageFont

# Load environment variables
load_dotenv()

# --- CONFIGURATION ---
ADB_HOST = "127.0.0.1"
ADB_PORT = 5037
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)


def create_grid_overlay(image_path: str, output_path: str = None, grid_size: int = 50) -> str:
    """
    Create a copy of the image with a coordinate grid overlay.
    Grid lines every `grid_size` pixels with coordinate labels.
    Returns path to the grid image.
    """
    if output_path is None:
        output_path = image_path.replace(".png", "_grid.png")
    
    with Image.open(image_path) as img:
        draw = ImageDraw.Draw(img)
        width, height = img.size
        
        # Draw vertical lines - thicker lines every 100px, thin lines every 50px
        for x in range(0, width, grid_size):
            line_width = 2 if x % 100 == 0 else 1
            color = (255, 0, 0) if x % 100 == 0 else (255, 100, 100)
            draw.line([(x, 0), (x, height)], fill=color, width=line_width)
            # Label every 100px
            if x % 100 == 0:
                draw.text((x + 2, 2), str(x), fill=(255, 255, 0))
        
        # Draw horizontal lines - thicker lines every 100px, thin lines every 50px  
        for y in range(0, height, grid_size):
            line_width = 2 if y % 100 == 0 else 1
            color = (255, 0, 0) if y % 100 == 0 else (255, 100, 100)
            draw.line([(0, y), (width, y)], fill=color, width=line_width)
            # Label every 100px
            if y % 100 == 0:
                draw.text((2, y + 2), str(y), fill=(255, 255, 0))
        
        # Add coordinate markers at major intersections (every 200px)
        for x in range(0, width, 200):
            for y in range(0, height, 200):
                if x > 0 and y > 0:  # Skip origin labels
                    draw.text((x + 2, y + 2), f"({x},{y})", fill=(0, 255, 255))
        
        # Also add markers at 100px intervals in the top area where icons typically are
        for x in range(100, min(width, 1000), 100):
            for y in range(100, 300, 100):
                draw.text((x + 2, y + 2), f"({x},{y})", fill=(0, 200, 200))
        
        img.save(output_path)
    
    return output_path


class AgentMemory:
    """
    Persistent memory for the QA agent to remember learned patterns and locations.
    Saves to JSON so it persists across runs.
    """
    def __init__(self, memory_file: str = "agent_memory.json"):
        self.memory_file = memory_file
        self.data = {
            "element_locations": {},  # element_name -> {x, y, app, screen_context}
            "successful_actions": [],  # List of action patterns that worked
            "failed_actions": [],      # List of action patterns that failed
            "app_knowledge": {},       # app_name -> {structure, patterns}
            "session_context": {}      # Current session state
        }
        self.load()
    
    def load(self):
        """Load memory from file if it exists."""
        if os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, 'r') as f:
                    saved = json.load(f)
                    self.data.update(saved)
                print(f"📚 Loaded agent memory from {self.memory_file}")
            except Exception as e:
                print(f"⚠ Could not load memory: {e}")
    
    def save(self):
        """Save memory to file."""
        try:
            with open(self.memory_file, 'w') as f:
                json.dump(self.data, f, indent=2)
        except Exception as e:
            print(f"⚠ Could not save memory: {e}")
    
    def remember_element_location(self, element_name: str, x: int, y: int, context: str = ""):
        """Remember where an element was found."""
        key = element_name.lower().strip()
        self.data["element_locations"][key] = {
            "x": x, "y": y, 
            "context": context,
            "found_at": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        self.save()
        print(f"  💾 Memorized: '{element_name}' at ({x}, {y})")
    
    def recall_element_location(self, element_name: str) -> Optional[Tuple[int, int]]:
        """Try to recall where an element was found before."""
        key = element_name.lower().strip()
        # Also try partial matches
        for stored_key, loc in self.data["element_locations"].items():
            if key in stored_key or stored_key in key:
                print(f"  🧠 Recalled: '{stored_key}' was at ({loc['x']}, {loc['y']})")
                return (loc['x'], loc['y'])
        return None
    
    def remember_successful_action(self, action: str, screen_context: str):
        """Remember an action that worked."""
        self.data["successful_actions"].append({
            "action": action,
            "context": screen_context,
            "time": time.strftime("%Y-%m-%d %H:%M:%S")
        })
        # Keep only last 100 actions
        self.data["successful_actions"] = self.data["successful_actions"][-100:]
        self.save()
    
    def remember_failed_action(self, action: str, screen_context: str, reason: str = ""):
        """Remember an action that failed."""
        self.data["failed_actions"].append({
            "action": action,
            "context": screen_context,
            "reason": reason,
            "time": time.strftime("%Y-%m-%d %H:%M:%S")
        })
        # Keep only last 50 failures
        self.data["failed_actions"] = self.data["failed_actions"][-50:]
        
        # If a gear/settings action failed, clear that memory since it was wrong
        if "gear" in action.lower() or "settings" in action.lower():
            if "gear" in self.data["element_locations"]:
                print(f"  🧹 Clearing bad memory for 'gear' since the action failed")
                del self.data["element_locations"]["gear"]
            if "settings" in self.data["element_locations"]:
                del self.data["element_locations"]["settings"]
        
        self.save()
    
    def forget_element(self, element_name: str):
        """Forget a memorized element location."""
        key = element_name.lower().strip()
        if key in self.data["element_locations"]:
            del self.data["element_locations"][key]
            self.save()
            print(f"  🧹 Forgot location for '{element_name}'")
    
    def set_session_context(self, key: str, value):
        """Store session-specific context."""
        self.data["session_context"][key] = value
    
    def get_session_context(self, key: str, default=None):
        """Get session-specific context."""
        return self.data["session_context"].get(key, default)
    
    def get_memory_summary(self) -> str:
        """Get a summary of remembered knowledge for the LLM."""
        summary_parts = []
        
        # Element locations
        if self.data["element_locations"]:
            locs = []
            for name, loc in self.data["element_locations"].items():
                locs.append(f"'{name}' at ({loc['x']}, {loc['y']})")
            summary_parts.append(f"Known element locations: {'; '.join(locs)}")
        
        # Recent failures to avoid
        recent_failures = self.data["failed_actions"][-5:]
        if recent_failures:
            fails = [f["action"] for f in recent_failures]
            summary_parts.append(f"Recent failed actions to avoid: {fails}")
        
        return "\n".join(summary_parts) if summary_parts else ""
    
    def clear_session(self):
        """Clear session-specific data but keep learned patterns."""
        self.data["session_context"] = {}


# Global memory instance
agent_memory = AgentMemory()


class ADBTools:
    def __init__(self):
        try:
            client = AdbClient(host=ADB_HOST, port=ADB_PORT)
            devices = client.devices()
            if not devices:
                raise Exception("No device connected. Is the emulator running?")
            self.device = devices[0]
            print(f"Connected to device: {self.device.serial}")
        except Exception as e:
            print(f"ADB Connection Failed: {e}")
            exit(1)

    def take_screenshot(self, filename="state.png"):
        result = self.device.screencap()
        with open(filename, "wb") as fp:
            fp.write(result)
        return filename

    def dump_ui_xml(self) -> Optional[str]:
        """Dump the current UI hierarchy and return it as text."""
        try:
            # uiautomator dump writes to /sdcard/window_dump.xml by default
            self.device.shell("uiautomator dump")
            raw = self.device.shell("cat /sdcard/window_dump.xml")
            return raw if raw else None
        except Exception:
            return None

    def find_bounds_by_text(self, needle: str) -> Optional[Tuple[int, int, int, int]]:
        """Find the first node whose text or content-desc contains `needle` (case-insensitive)."""
        xml_text = self.dump_ui_xml()
        if not xml_text:
            return None
        try:
            root = ET.fromstring(xml_text)
        except Exception:
            return None

        def matches(node):
            text = (node.attrib.get("text") or "") + " " + (node.attrib.get("content-desc") or "")
            return needle.lower() in text.lower()

        for node in root.iter():
            if matches(node):
                bounds_str = node.attrib.get("bounds")
                if not bounds_str:
                    continue
                m = re.match(r"\[(\d+),(\d+)\]\[(\d+),(\d+)\]", bounds_str)
                if not m:
                    continue
                x1, y1, x2, y2 = map(int, m.groups())
                return x1, y1, x2, y2
        return None

    def find_bounds_by_keywords(self, *keywords) -> Optional[Tuple[int, int, int, int]]:
        """Find the first element matching any of the keywords (for flexible button matching)."""
        # Try each keyword in order: exact text match first, then contains, then content-desc
        xml_text = self.dump_ui_xml()
        if not xml_text:
            return None
        try:
            root = ET.fromstring(xml_text)
        except Exception:
            return None

        # Exact text match
        for keyword in keywords:
            for node in root.iter():
                text = (node.attrib.get("text", "") or "").strip()
                if text.lower() == keyword.lower():
                    bounds_str = node.attrib.get("bounds")
                    if bounds_str:
                        m = re.match(r"\[(\d+),(\d+)\]\[(\d+),(\d+)\]", bounds_str)
                        if m:
                            return tuple(map(int, m.groups()))

        # Exact content-desc match (important for navigation buttons like "Navigate up", "Back")
        for keyword in keywords:
            for node in root.iter():
                content_desc = (node.attrib.get("content-desc", "") or "").strip()
                if content_desc.lower() == keyword.lower():
                    bounds_str = node.attrib.get("bounds")
                    if bounds_str:
                        m = re.match(r"\[(\d+),(\d+)\]\[(\d+),(\d+)\]", bounds_str)
                        if m:
                            return tuple(map(int, m.groups()))

        # Contains text match
        for keyword in keywords:
            for node in root.iter():
                text = (node.attrib.get("text", "") or "").strip()
                if keyword.lower() in text.lower():
                    bounds_str = node.attrib.get("bounds")
                    if bounds_str:
                        m = re.match(r"\[(\d+),(\d+)\]\[(\d+),(\d+)\]", bounds_str)
                        if m:
                            return tuple(map(int, m.groups()))

        # Contains content-desc match
        for keyword in keywords:
            for node in root.iter():
                content_desc = (node.attrib.get("content-desc", "") or "").lower()
                if keyword.lower() in content_desc:
                    bounds_str = node.attrib.get("bounds")
                    if bounds_str:
                        m = re.match(r"\[(\d+),(\d+)\]\[(\d+),(\d+)\]", bounds_str)
                        if m:
                            return tuple(map(int, m.groups()))

        return None

    def dump_visible_text(self) -> str:
        """Return all visible text in the UI hierarchy for debugging."""
        xml_text = self.dump_ui_xml()
        if not xml_text:
            return "[UI dump failed]"
        try:
            root = ET.fromstring(xml_text)
        except Exception as e:
            return f"[XML parse failed: {e}]"
        
        texts = []
        for node in root.iter():
            text = node.attrib.get("text", "").strip()
            if text:
                texts.append(text)
        return "; ".join(texts[:20])  # First 20 visible texts

    def get_all_ui_text_and_bounds(self) -> dict:
        """Return dict of all visible text -> bounds for debugging."""
        xml_text = self.dump_ui_xml()
        if not xml_text:
            return {}
        try:
            root = ET.fromstring(xml_text)
        except Exception:
            return {}
        
        result = {}
        for node in root.iter():
            text = node.attrib.get("text", "").strip()
            bounds_str = node.attrib.get("bounds", "")
            if text and bounds_str:
                m = re.match(r"\[(\d+),(\d+)\]\[(\d+),(\d+)\]", bounds_str)
                if m:
                    x1, y1, x2, y2 = map(int, m.groups())
                    result[text] = (x1, y1, x2, y2)
        return result

    def find_input_field_bounds(self, label_text: str) -> Optional[Tuple[int, int, int, int]]:
        """Find an input field (EditText/TextInput) located below a given label."""
        xml_text = self.dump_ui_xml()
        if not xml_text:
            return None
        try:
            root = ET.fromstring(xml_text)
        except Exception:
            return None

        # Locate the label node by text
        label_node = None
        for node in root.iter():
            text = node.attrib.get("text", "").lower()
            if label_text.lower() in text:
                label_node = node
                break

        if not label_node:
            return None

        # Parse label bounds
        label_bounds_str = label_node.attrib.get("bounds")
        if not label_bounds_str:
            return None
        m = re.match(r"\[(\d+),(\d+)\]\[(\d+),(\d+)\]", label_bounds_str)
        if not m:
            return None
        _, label_y1, _, label_y2 = map(int, m.groups())

        # Search for input-like widgets below the label
        for node in root.iter():
            class_attr = node.attrib.get("class", "")
            bounds_str = node.attrib.get("bounds", "")
            if "EditText" in class_attr or "TextInput" in class_attr or node.attrib.get("resource-id", "").endswith("input"):
                if bounds_str:
                    m = re.match(r"\[(\d+),(\d+)\]\[(\d+),(\d+)\]", bounds_str)
                    if m:
                        x1, y1, x2, y2 = map(int, m.groups())
                        if y1 >= label_y2 - 10:  # ensure below or aligned with label
                            return (x1, y1, x2, y2)

        return None

    def find_placeholder_bounds(self, *placeholders) -> Optional[Tuple[int, int, int, int]]:
        """Find an input field by its placeholder text (e.g., 'My vault'). Only matches EditText widgets."""
        xml_text = self.dump_ui_xml()
        if not xml_text:
            return None
        try:
            root = ET.fromstring(xml_text)
        except Exception:
            return None

        for node in root.iter():
            text = (node.attrib.get("text", "") or "").lower()
            content_desc = (node.attrib.get("content-desc", "") or "").lower()
            bounds_str = node.attrib.get("bounds", "")
            class_attr = node.attrib.get("class", "")
            if not bounds_str:
                continue
            # ONLY match EditText widgets, not labels or headers
            if "EditText" not in class_attr:
                continue
            for ph in placeholders:
                ph_l = ph.lower()
                if ph_l in text or ph_l in content_desc:
                    m = re.match(r"\[(\d+),(\d+)\]\[(\d+),(\d+)\]", bounds_str)
                    if m:
                        return tuple(map(int, m.groups()))
        return None

    def find_first_edit_text(self) -> Optional[Tuple[int, int, int, int]]:
        """Find the first EditText widget on screen (useful fallback for empty input fields)."""
        xml_text = self.dump_ui_xml()
        if not xml_text:
            return None
        try:
            root = ET.fromstring(xml_text)
        except Exception:
            return None

        for node in root.iter():
            class_attr = node.attrib.get("class", "")
            bounds_str = node.attrib.get("bounds", "")
            if "EditText" in class_attr and bounds_str:
                m = re.match(r"\[(\d+),(\d+)\]\[(\d+),(\d+)\]", bounds_str)
                if m:
                    return tuple(map(int, m.groups()))
        return None

    def find_toggle_or_switch(self, label_text: str = None) -> Optional[Tuple[int, int, int, int]]:
        """Find a toggle/switch widget, optionally near a label."""
        xml_text = self.dump_ui_xml()
        if not xml_text:
            return None
        try:
            root = ET.fromstring(xml_text)
        except Exception:
            return None

        for node in root.iter():
            class_attr = node.attrib.get("class", "")
            checkable = node.attrib.get("checkable", "")
            bounds_str = node.attrib.get("bounds", "")
            text = node.attrib.get("text", "")
            
            # Look for toggle/switch/checkbox widgets
            is_toggle = ("Switch" in class_attr or "Toggle" in class_attr or 
                        "Check" in class_attr or checkable == "true")
            
            if is_toggle and bounds_str:
                m = re.match(r"\[(\d+),(\d+)\]\[(\d+),(\d+)\]", bounds_str)
                if m:
                    bounds = tuple(map(int, m.groups()))
                    # If no label specified, return first toggle found
                    if not label_text:
                        return bounds
                    # If label specified, check if text matches
                    if label_text.lower() in text.lower():
                        return bounds
        
        # If label specified but no matching toggle found, find toggle near label
        if label_text:
            label_bounds = self.find_bounds_by_text(label_text)
            if label_bounds:
                label_y = label_bounds[1]
                # Find any toggle within 200px vertically of the label
                for node in root.iter():
                    checkable = node.attrib.get("checkable", "")
                    class_attr = node.attrib.get("class", "")
                    bounds_str = node.attrib.get("bounds", "")
                    is_toggle = ("Switch" in class_attr or "Toggle" in class_attr or 
                                "Check" in class_attr or checkable == "true")
                    if is_toggle and bounds_str:
                        m = re.match(r"\[(\d+),(\d+)\]\[(\d+),(\d+)\]", bounds_str)
                        if m:
                            x1, y1, x2, y2 = map(int, m.groups())
                            if abs(y1 - label_y) < 200:
                                return (x1, y1, x2, y2)
        return None

    def find_button_by_text(self, button_text: str) -> Optional[Tuple[int, int, int, int]]:
        """
        Find a clickable Button widget by its text.
        This is more precise than find_bounds_by_text because it only matches actual buttons,
        not labels or other text elements.
        """
        xml_text = self.dump_ui_xml()
        if not xml_text:
            return None
        try:
            root = ET.fromstring(xml_text)
        except Exception:
            return None

        button_text_lower = button_text.lower()
        
        # First pass: Look for actual Button widgets with matching text
        for node in root.iter():
            class_attr = node.attrib.get("class", "")
            text = (node.attrib.get("text", "") or "").strip()
            bounds_str = node.attrib.get("bounds", "")
            clickable = node.attrib.get("clickable", "")
            
            # Match Button class or clickable elements
            is_button = ("Button" in class_attr or clickable == "true")
            
            if is_button and bounds_str and text.lower() == button_text_lower:
                m = re.match(r"\[(\d+),(\d+)\]\[(\d+),(\d+)\]", bounds_str)
                if m:
                    return tuple(map(int, m.groups()))
        
        # Second pass: Look for clickable elements containing the text
        for node in root.iter():
            class_attr = node.attrib.get("class", "")
            text = (node.attrib.get("text", "") or "").strip()
            bounds_str = node.attrib.get("bounds", "")
            clickable = node.attrib.get("clickable", "")
            
            is_button = ("Button" in class_attr or clickable == "true")
            
            if is_button and bounds_str and button_text_lower in text.lower():
                m = re.match(r"\[(\d+),(\d+)\]\[(\d+),(\d+)\]", bounds_str)
                if m:
                    return tuple(map(int, m.groups()))
        
        return None

    def find_settings_icon(self) -> Optional[Tuple[int, int, int, int]]:
        """Find the settings/gear icon by looking for common content-desc patterns."""
        xml_text = self.dump_ui_xml()
        if not xml_text:
            return None
        try:
            root = ET.fromstring(xml_text)
        except Exception:
            return None

        # Common content-desc patterns for settings icons
        settings_patterns = ["settings", "gear", "cog", "preferences", "options", "config", "open settings"]
        
        # First pass: look for clickable elements with settings-related content-desc
        for node in root.iter():
            content_desc = (node.attrib.get("content-desc", "") or "").lower()
            bounds_str = node.attrib.get("bounds", "")
            clickable = node.attrib.get("clickable", "")
            class_attr = node.attrib.get("class", "")
            
            is_clickable = (clickable == "true" or "Button" in class_attr or "Image" in class_attr)
            
            if bounds_str and is_clickable:
                for pattern in settings_patterns:
                    if pattern in content_desc:
                        m = re.match(r"\[(\d+),(\d+)\]\[(\d+),(\d+)\]", bounds_str)
                        if m:
                            print(f"  → Found settings icon via content-desc '{content_desc}'")
                            return tuple(map(int, m.groups()))
        
        # Second pass: also check text attribute for "Settings"
        for node in root.iter():
            text = (node.attrib.get("text", "") or "").lower()
            bounds_str = node.attrib.get("bounds", "")
            clickable = node.attrib.get("clickable", "")
            
            if bounds_str and (clickable == "true" or text):
                if "settings" in text:
                    m = re.match(r"\[(\d+),(\d+)\]\[(\d+),(\d+)\]", bounds_str)
                    if m:
                        print(f"  → Found settings via text '{text}'")
                        return tuple(map(int, m.groups()))
        
        return None

    def dump_all_content_desc(self) -> list:
        """Return all content-desc values in UI for debugging."""
        xml_text = self.dump_ui_xml()
        if not xml_text:
            return []
        try:
            root = ET.fromstring(xml_text)
        except Exception:
            return []
        
        result = []
        for node in root.iter():
            content_desc = node.attrib.get("content-desc", "").strip()
            bounds_str = node.attrib.get("bounds", "")
            if content_desc and bounds_str:
                result.append(f"{content_desc} @ {bounds_str}")
        return result

    def dump_all_clickable_elements(self) -> list:
        """Return all clickable elements with their class, text, content-desc for debugging."""
        xml_text = self.dump_ui_xml()
        if not xml_text:
            return []
        try:
            root = ET.fromstring(xml_text)
        except Exception:
            return []
        
        result = []
        for node in root.iter():
            clickable = node.attrib.get("clickable", "")
            class_attr = node.attrib.get("class", "")
            bounds_str = node.attrib.get("bounds", "")
            
            if clickable == "true" and bounds_str:
                text = node.attrib.get("text", "").strip()
                content_desc = node.attrib.get("content-desc", "").strip()
                resource_id = node.attrib.get("resource-id", "").strip()
                result.append({
                    "class": class_attr,
                    "text": text,
                    "content_desc": content_desc,
                    "resource_id": resource_id,
                    "bounds": bounds_str
                })
        return result

    def find_bottom_left_icon(self, max_x=300, min_y=1800, max_y=2200) -> Optional[Tuple[int, int, int, int]]:
        """Find clickable icon in bottom-left area of sidebar (NOT the navigation bar at very bottom)."""
        xml_text = self.dump_ui_xml()
        if not xml_text:
            return None
        try:
            root = ET.fromstring(xml_text)
        except Exception:
            return None
        
        candidates = []
        for node in root.iter():
            clickable = node.attrib.get("clickable", "")
            bounds_str = node.attrib.get("bounds", "")
            class_attr = node.attrib.get("class", "")
            text = node.attrib.get("text", "").strip()
            
            # Look for clickable elements (especially ImageView/ImageButton) without text (icons)
            if bounds_str and (clickable == "true" or "Image" in class_attr):
                m = re.match(r"\[(\d+),(\d+)\]\[(\d+),(\d+)\]", bounds_str)
                if m:
                    x1, y1, x2, y2 = map(int, m.groups())
                    # Check if in sidebar bottom region (NOT the nav bar at very bottom)
                    # Skip elements with navigation text
                    if x2 <= max_x and y1 >= min_y and y2 <= max_y:
                        if text.lower() not in ["navigate back", "navigate forward", ""]:
                            candidates.append((x1, y1, x2, y2, class_attr, text))
                        elif not text:  # Empty text could be icon
                            candidates.append((x1, y1, x2, y2, class_attr, text))
        
        # Filter out navigation buttons explicitly
        candidates = [c for c in candidates if c[5].lower() not in ["navigate back", "navigate forward"]]
        
        # Return the bottommost one
        if candidates:
            candidates.sort(key=lambda c: c[1], reverse=True)  # Sort by y1 descending
            best = candidates[0]
            print(f"  → Found bottom-left icon: {best[4]} text='{best[5]}' at bounds [{best[0]},{best[1]}][{best[2]},{best[3]}]")
            return (best[0], best[1], best[2], best[3])
        return None

    def get_screen_size(self):
        """Return (width, height) parsed from `wm size`, fallback to screenshot size."""
        try:
            raw = self.device.shell("wm size")
            match = re.search(r"Physical size:\s*(\d+)x(\d+)", raw)
            if match:
                return int(match.group(1)), int(match.group(2))
        except Exception:
            pass

        # Fallback: use a quick screenshot to infer dimensions
        tmp_path = "_tmp_screen.png"
        self.take_screenshot(tmp_path)
        try:
            with Image.open(tmp_path) as im:
                return im.size  # (width, height)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def tap(self, x, y):
        print(f"Executing: TAP ({x}, {y})")
        self.device.shell(f"input tap {x} {y}")

    def clear_text_field(self):
        """Clear text in currently focused field using select-all + delete."""
        # CTRL+A to select all (keycode 29 = A, with meta ctrl)
        self.device.shell("input keyevent --longpress 67")  # Long press delete
        time.sleep(0.1)
        # Move to end and delete backwards (more reliable)
        self.device.shell("input keyevent 123")  # KEYCODE_MOVE_END
        time.sleep(0.1)
        # Send multiple deletes to clear any existing text
        for _ in range(30):
            self.device.shell("input keyevent 67")  # KEYCODE_DEL
        time.sleep(0.2)

    def type_text(self, text, clear_first=False):
        print(f"Executing: TYPE '{text}'")
        
        if clear_first:
            print("  → Clearing field first...")
            self.clear_text_field()
        
        # Add small delay before typing to let UI settle
        time.sleep(0.3)
        
        # ADB input text: use proper shell escaping
        # Replace spaces with %s (ADB convention)
        escaped = text.replace(" ", "%s")
        # Use printf-style escaping for special chars
        escaped = escaped.replace("'", "\\'")
        escaped = escaped.replace('"', '\\"')
        escaped = escaped.replace("&", "\\&")
        escaped = escaped.replace("|", "\\|")
        escaped = escaped.replace(";", "\\;")
        
        # Send command directly without outer quotes
        cmd = f"input text {escaped}"
        print(f"  → ADB command: {cmd}")
        result = self.device.shell(cmd)
        if result:
            print(f"  → Result: {result}")
    
    def key_event(self, key_code):
        self.device.shell(f"input keyevent {key_code}")

    def swipe(self, start_x, start_y, end_x, end_y, duration_ms=300):
        """Perform a swipe gesture."""
        cmd = f"input swipe {start_x} {start_y} {end_x} {end_y} {duration_ms}"
        self.device.shell(cmd)

class LLMService:
    """Modular wrapper to easily swap models later."""
    def __init__(self):
        # Allow model override via env; default to current public image-capable model
        model_name = os.getenv("GEMINI_MODEL", "models/gemini-2.5-flash")
        self.model = genai.GenerativeModel(model_name)

    def analyze_image(self, prompt, image_path):
        img = Image.open(image_path)
        response = self.model.generate_content([prompt, img])
        return response.text.strip()

# --- AGENT ROLES ---

def extract_target_text(step_description: str) -> list:
    """
    Extract potential target text from a step description.
    Returns a list of candidate texts to search for in the UI.
    """
    candidates = []
    
    # Extract quoted strings (e.g., "Tap the 'Create a vault' button")
    quoted = re.findall(r"['\"]([^'\"]+)['\"]", step_description)
    candidates.extend(quoted)
    
    # Extract text after "the" and before common suffixes (e.g., "Tap the Create button")
    match = re.search(r"tap\s+(?:the\s+)?(.+?)(?:\s+button|\s+icon|\s+link|\s+option|\s*$)", step_description, re.IGNORECASE)
    if match:
        text = match.group(1).strip().strip("'\"")
        if text and text not in candidates:
            candidates.append(text)
    
    # Also try the full step if it's short (like "Continue" or "Create")
    words = step_description.split()
    for word in words:
        clean = word.strip("'\".,!?")
        if len(clean) > 3 and clean.lower() not in ["tap", "the", "button", "icon", "click", "press", "labeled"]:
            if clean not in candidates:
                candidates.append(clean)
    
    return candidates

class Planner:
    def __init__(self, llm, memory: AgentMemory = None):
        self.llm = llm
        self.memory = memory or agent_memory

    def get_next_step(self, objective, history, screenshot_path, visible_ui_text=""):
        # Get memory summary to help with decisions
        memory_hint = self.memory.get_memory_summary()
        
        prompt = f"""
You are an expert Android QA Agent. Analyze the screenshot and decide the SINGLE next action.

OBJECTIVE: {objective}

ACTIONS ALREADY COMPLETED: {history if history else "None yet"}

VISIBLE UI TEXT ON SCREEN: {visible_ui_text}

{f"AGENT MEMORY (learned from past runs): {memory_hint}" if memory_hint else ""}

CRITICAL RULES:
1. NEVER repeat an action you already did. Check the history above.
2. If you just typed something, DON'T tap on that same field again - move to the NEXT thing.
3. Use EXACT text labels from the screen - check "VISIBLE UI TEXT" above.
4. For app icons, use the app NAME shown below the icon (e.g., "Tap the 'Obsidian' app icon")
5. If you see typed text in VISIBLE UI TEXT (like 'Meeting Notes'), that typing is DONE - don't redo it!

APP BEHAVIOR - IMPORTANT:
- The app title bar (at very top) may show 'Untitled' until you leave the field
- Look at the ACTUAL CONTENT in the editor, not just the title bar
- If your typed text appears in VISIBLE UI TEXT, it was successfully typed
- After typing title, the title bar updates when you tap elsewhere or dismiss keyboard

NAVIGATION - FINDING SETTINGS:
- Settings is usually accessed via: gear icon (⚙), "Settings" text, or menu
- In Obsidian mobile app:
  1. First: Tap the "Expand" button (≡) on the left to open the sidebar
  2. Then: Look for a gear icon (⚙) in the sidebar - it may be near the top next to other icons
  3. Just say "Tap the gear icon" - the vision system will find it
- Settings screens have tabs/sections like "Appearance", "Editor", "Files", etc.
- "More options" (⋮) is NOT Settings - avoid clicking that

IMPORTANT: If you see an icon but it doesn't have text, just describe what to tap (e.g., "Tap the gear icon"). The vision system will locate it.

UI ELEMENT WARNINGS - AVOID CONFUSION:
- "Navigate back" and "Navigate forward" at the BOTTOM of the screen are UNDO/REDO buttons for the note editor, NOT navigation buttons!
- Do NOT tap these for going back in Settings - they are unrelated to Settings navigation
- If you need to go back in Settings, look for a back arrow at the TOP-LEFT of the Settings screen
- When in Settings/Appearance, IGNORE UI elements from the note editor behind the overlay

VERIFYING COLORS:
- If asked to verify an icon color, LOOK at the icon in the screenshot
- Describe what color you actually SEE (e.g., gray, purple, blue, red, etc.)
- If the color does NOT match what's expected, output: FAIL: The icon is [actual color], not [expected color]
- If the color matches, output: DONE

SPATIAL AWARENESS:
- Title fields are at the TOP of the note editor (below the app bar)
- Body/content areas are BELOW the title - they're the large empty space
- To move from title to body: "Press down arrow" or tap the empty area BELOW the title
- NEVER tap on text you already typed - that will select/overwrite it!

DECISION LOGIC:
1. What does the objective ask for that ISN'T done yet?
2. Look at the screenshot - what's the current state?
3. What's the SINGLE next step to make progress?
4. If EVERYTHING in the objective is visible/done, output: DONE

OUTPUT FORMAT - Use EXACT text from the screen:
- "Tap the 'Obsidian' app icon" (use exact app name, not description like "purple icon")
- "Tap the 'Create a vault' button" (use exact button text)
- "Tap the text input field" (for empty input fields)
- "Tap the gear icon" (for icons without text labels - vision will find it)
- "Type '<text>'" (to enter text - cursor must already be in a field)
- "Press down arrow" (to move from title to body)
- "Tap the body area below the title" (to focus the body/content area)
- "DONE" (if objective is fully complete)
- "FAIL: <reason>" (ONLY if truly impossible - e.g., app crashed, element doesn't exist at all)

CRITICAL OUTPUT RULE: Output ONLY the action itself. NO explanations, NO reasoning, NO paragraphs. Just the single action line like "Tap the gear icon" - nothing else!

IMPORTANT FOR ICONS: If you see an icon (gear, settings, menu) but it has no text label, just describe what to tap like "Tap the gear icon in the sidebar". The vision system will locate it visually.

If you just typed the title and now need to type in the body, you MUST first move to the body (press down arrow or tap body area) BEFORE typing body content.
"""
        response = self.llm.analyze_image(prompt, screenshot_path)
        # Extract just the action line if Planner gave explanations
        lines = response.strip().split('\n')
        for line in reversed(lines):
            line = line.strip()
            if line and (line.startswith('Tap') or line.startswith('Type') or line.startswith('Press') or 
                        line.startswith('DONE') or line.startswith('FAIL')):
                return line
        # Fallback to last non-empty line
        for line in reversed(lines):
            if line.strip():
                return line.strip()
        return response

class Executor:
    def __init__(self, llm, screen_size, memory: AgentMemory = None):
        self.llm = llm
        self.screen_size = screen_size
        self.memory = memory or agent_memory

    def execute_step(self, step_description, screenshot_path, target_hint=None):
        step_lower = step_description.lower()
        
        # Check memory for known element locations (for icons without text)
        if target_hint is None and "tap" in step_lower:
            # Try to recall from memory
            for keyword in ["gear", "settings", "menu", "back", "expand"]:
                if keyword in step_lower:
                    remembered = self.memory.recall_element_location(keyword)
                    if remembered:
                        print(f"  → Using memorized location for '{keyword}'")
                        return {"action": "tap", "x": remembered[0], "y": remembered[1]}
        
        # FAST PATH: Key press actions (arrow keys, enter, etc.)
        if "press" in step_lower and "arrow" in step_lower:
            if "down" in step_lower:
                return {"action": "key", "keycode": 20}  # KEYCODE_DPAD_DOWN
            if "up" in step_lower:
                return {"action": "key", "keycode": 19}  # KEYCODE_DPAD_UP
            if "left" in step_lower:
                return {"action": "key", "keycode": 21}  # KEYCODE_DPAD_LEFT
            if "right" in step_lower:
                return {"action": "key", "keycode": 22}  # KEYCODE_DPAD_RIGHT
        if "press enter" in step_lower or "press return" in step_lower:
            return {"action": "key", "keycode": 66}  # KEYCODE_ENTER
        
        # FAST PATH: Swipe actions
        if "swipe" in step_lower or "scroll" in step_lower:
            # Determine direction and area
            screen_w, screen_h = self.screen_size
            # Default: scroll in center of left side (sidebar area)
            center_x = 250  # Sidebar center
            if "up" in step_lower or "down" in step_lower:
                # Swipe up = scroll down (reveal bottom content)
                if "up" in step_lower:
                    return {"action": "swipe", "start_x": center_x, "start_y": screen_h * 0.7, 
                            "end_x": center_x, "end_y": screen_h * 0.3}
                else:  # Swipe down = scroll up
                    return {"action": "swipe", "start_x": center_x, "start_y": screen_h * 0.3, 
                            "end_x": center_x, "end_y": screen_h * 0.7}
        
        # FAST PATH: If we have precise bounds from UI dump, use them directly (no LLM call needed)
        if target_hint and "tap" in step_lower:
            cx = (target_hint[0] + target_hint[2]) // 2
            cy = (target_hint[1] + target_hint[3]) // 2
            print(f"  → Using UI dump bounds directly: center ({cx}, {cy})")
            return {"action": "tap", "x": cx, "y": cy}
        
        # FAST PATH: For type actions, extract text directly (no LLM needed)
        if "type" in step_lower:
            quoted = re.findall(r"['\"]([^'\"]+)['\"]", step_description)
            if quoted:
                return {"action": "type", "text": quoted[0]}
        
        # SLOW PATH: Need LLM to figure out coordinates (no UI dump match found)
        # Create a grid overlay to help the model with coordinates
        grid_image_path = create_grid_overlay(screenshot_path, grid_size=50)
        print(f"  → Using vision with grid overlay to find element")
        
        prompt = f"""
You are an expert Android Automation Executor. Your job is to translate a human-readable action into precise screen coordinates.

ACTION TO PERFORM: {step_description}

SCREEN SIZE: {self.screen_size[0]}x{self.screen_size[1]} pixels

The screenshot has a RED GRID overlay with coordinate labels:
- Major red lines every 100 pixels, minor lines every 50 pixels
- Yellow numbers along edges show X (horizontal) and Y (vertical) coordinates  
- Cyan labels at intersections show (x,y) coordinates
- Use these grid lines to precisely identify the location of UI elements

INSTRUCTIONS:
1. Look at the screenshot with the grid overlay.
2. Find the UI element that matches the action description.
3. Use the grid lines and labels to determine PRECISE coordinates.
4. For TAP actions: return the CENTER coordinates of the target element.
5. Return ONLY a valid JSON object.

FINDING ICONS IN THE SIDEBAR HEADER:
- When the sidebar is open, look at the TOP ROW of the sidebar panel (y around 150-250)
- There are typically several icons in a row: menu/hamburger, vault name, and GEAR ICON
- The GEAR icon (⚙) is usually on the RIGHT side of the sidebar header, NOT the left
- Look for a circular icon with notches/teeth - this is the settings gear
- The gear is typically between x=800-900 when sidebar is open
- Be careful not to tap vault name or other icons to the LEFT of the gear

COORDINATE TIPS:
- Grid has major lines every 100px, minor lines every 50px
- Find which grid cell contains your target element
- Look at the exact pixel position, not just the grid cell
- Double-check: the gear icon should be the RIGHTMOST icon in the sidebar header row

SUPPORTED ACTIONS:
- Tap: {{"action": "tap", "x": <center_x>, "y": <center_y>}}
- Type: {{"action": "type", "text": "<text_to_type>"}}
- Wait: {{"action": "wait", "seconds": <1-3>}}

OUTPUT: Valid JSON only, no explanation.
"""
        response = self.llm.analyze_image(prompt, grid_image_path)
        # Clean up code block formatting if present
        clean_json = response.replace("```json", "").replace("```", "").strip()

        try:
            action = json.loads(clean_json)
        except Exception as e:
            print(f"  ! JSON parse failed: {e}")
            print(f"  ! Raw response: {repr(response)[:200]}")
            # Fallback: infer action from step description
            quoted = re.findall(r"'([^']+)'", step_description)
            if "type" in step_lower and quoted:
                return {"action": "type", "text": quoted[0]}
            return {"action": "wait", "seconds": 1}

        if isinstance(action, list):
            action = action[0] if action else {"action": "wait", "seconds": 1}
        if not isinstance(action, dict) or "action" not in action:
            return {"action": "wait", "seconds": 1}

        return action

class Supervisor:
    def __init__(self, llm):
        self.llm = llm
        
    def verify_state(self, objective, screenshot_path, step_count=0):
        prompt = f"""
You are a QA Supervisor checking if a test objective is complete.

OBJECTIVE: {objective}
STEPS COMPLETED: {step_count}

Look at the screenshot and determine the status:

PASS - Output this ONLY if:
- The FULL objective has been achieved
- You can SEE the content that was requested (title text, body text) in the editor area
- NOTE: The app title bar may still show 'Untitled' - look at the ACTUAL TEXT in the note content, not the title bar
- If you see both the title text AND body text in the editor, that's a PASS

FAIL - Output this ONLY if:
- There's an actual ERROR message or crash on screen
- The app is stuck or broken
- Something went clearly WRONG (not just "not done yet")
- The text is clearly WRONG (misspelled, in wrong place)

CONTINUE - Output this if:
- The app is showing intermediate screens (setup, permissions, sync options, etc.)
- The task is still in progress
- You're on a screen that's part of the normal flow but not the final result
- The objective isn't complete YET but nothing is wrong

IMPORTANT: Intermediate screens like "sync setup", "permissions", "vault configuration" are NORMAL - output CONTINUE, not FAIL.
Only output FAIL for actual errors or crashes.
"""
        return self.llm.analyze_image(prompt, screenshot_path)

# --- MAIN LOOP ---

def run_test_case(objective):
    os.makedirs("debug_taps", exist_ok=True)
    adb = ADBTools()
    llm = LLMService()
    screen_size = adb.get_screen_size()
    print(f"Detected screen size: {screen_size[0]}x{screen_size[1]}")
    
    planner = Planner(llm)
    executor = Executor(llm, screen_size)
    supervisor = Supervisor(llm)
    
    history = []
    tap_log = []
    step_count = 0
    max_steps = 15 # Safety limit - allow enough steps for multi-screen flows
    
    print(f"--- STARTING TEST: {objective} ---")
    
    while step_count < max_steps:
        # 1. Observe
        screenshot = adb.take_screenshot("current_state.png")
        visible_ui = adb.dump_visible_text()
        print(f"Visible UI text: {visible_ui}")
        
        # Debug: show all available text-bounds for diagnostics
        all_text_bounds = adb.get_all_ui_text_and_bounds()
        if all_text_bounds and step_count == 0:  # Log once at start
            print(f"DEBUG - Available UI elements: {list(all_text_bounds.keys())}")
        
        # Verify pending gear/settings location - if we now see settings content, memorize it
        pending_gear = agent_memory.get_session_context("pending_gear_location")
        if pending_gear:
            settings_indicators = ["appearance", "editor", "files & links", "about", "base color", "accent color", "theme"]
            if any(ind in visible_ui.lower() for ind in settings_indicators):
                print(f"  ✓ Verified: gear tap worked! Memorizing location ({pending_gear['x']}, {pending_gear['y']})")
                agent_memory.remember_element_location("gear", pending_gear['x'], pending_gear['y'], context="Settings screen")
            agent_memory.set_session_context("pending_gear_location", None)
        
        # 2. Check Status (Supervisor)
        # Only check after we've done at least 3 steps (let app load and navigate)
        if step_count >= 3:
            status = supervisor.verify_state(objective, screenshot, step_count)
            print(f"Supervisor Status: {status}")
            if "PASS" in status:
                print("Test Passed!")
                return True
            if "FAIL" in status:
                print(f"Supervisor reports failure, but continuing...")
                # Only fail after many steps - give the agent time to complete the task
                if step_count > 7:
                    print("Test Failed after retries.")
                    return False

        # 3. Plan - pass visible UI text to help Planner use exact labels
        next_step = planner.get_next_step(objective, history, screenshot, visible_ui)
        print(f"Planner suggests: {next_step}")
        
        if "DONE" in next_step:
            print("Planner decided task is finished.")
            agent_memory.remember_successful_action(f"Completed: {objective}", visible_ui[:100])
            break
        if "FAIL" in next_step:
            print(f"Planner reported failure: {next_step}")
            agent_memory.remember_failed_action(next_step, visible_ui[:100], reason=next_step)
            return False
            
        history.append(next_step)
        
        # --- ELEMENT DETECTION: Use UI dump for precise coordinates ---
        step_lower = next_step.lower()
        target_bounds = None
        target_label = "Target"
        
        # Get all UI elements with their bounds
        all_bounds_dict = adb.get_all_ui_text_and_bounds()
        
        # Detect if this action is about the BODY AREA (must check FIRST, before input field detection)
        is_body_action = "body" in step_lower or "content area" in step_lower
        
        # Detect if this action is about SETTINGS/GEAR icon
        is_settings_action = any(kw in step_lower for kw in ["settings", "gear", "cog", "preferences"])
        
        # Debug: when looking for settings, show all clickable elements
        if is_settings_action:
            all_content_desc = adb.dump_all_content_desc()
            all_clickable = adb.dump_all_clickable_elements()
            print(f"  DEBUG - All content-desc values: {all_content_desc}")
            print(f"  DEBUG - Clickable elements in header area (y < 300):")
            potential_gear = None
            for el in all_clickable:
                # Parse bounds to get coordinates
                m = re.match(r"\[(\d+),(\d+)\]\[(\d+),(\d+)\]", el['bounds'])
                if m:
                    x1, y1, x2, y2 = map(int, m.groups())
                    if y2 < 300:  # Show header elements
                        print(f"    → {el['class']}: text='{el['text']}' @ {el['bounds']}")
                        # Look for unlabeled clickable element on right side of header (likely gear icon)
                        if not el['text'] and x1 > 700 and y1 < 260:
                            potential_gear = (x1, y1, x2, y2)
                            print(f"    *** POTENTIAL GEAR ICON (unlabeled, right side of header)")
            
            # If we found a potential gear, use it
            if potential_gear and not target_bounds:
                target_bounds = potential_gear
                target_label = "PotentialGearIcon"
                print(f"  → Using unlabeled header icon as gear: {target_bounds}")
        
        # Detect if this action is about an INPUT FIELD (not a button/label)
        # But NOT if it's a body action or settings action
        is_input_action = not is_body_action and not is_settings_action and any(kw in step_lower for kw in [
            "input", "field", "textbox", "text box", "text field", 
            "type in", "enter text", "vault name"
        ])
        
        # Detect if this action is about a PERMISSION BUTTON (Allow, Deny, OK, Cancel, etc.)
        is_permission_action = any(kw in step_lower for kw in [
            "allow", "deny", "permit", "grant", "ok", "cancel", "accept", "decline"
        ])
        
        # A. SETTINGS/GEAR ICON - find by content-desc, or let LLM vision find it
        if is_settings_action and "tap" in step_lower:
            settings_bounds = adb.find_settings_icon()
            if settings_bounds:
                target_bounds = settings_bounds
                target_label = "SettingsIcon"
                print(f"  → Found Settings/Gear icon via content-desc at: {target_bounds}")
            else:
                # No UI dump match - let LLM vision find the gear icon
                print(f"  → Settings icon not in UI dump, will use LLM vision to locate gear icon")
        
        # B. BODY AREA - tap below the title (check before input field detection)
        if not target_bounds and is_body_action and "tap" in step_lower:
            # Tap below where title would be - use 40% down the screen
            screen_w, screen_h = screen_size
            body_x = screen_w // 2
            body_y = int(screen_h * 0.4)  # 40% down the screen (below title, above keyboard area)
            target_bounds = (body_x - 50, body_y - 50, body_x + 50, body_y + 50)
            target_label = "BodyArea"
            print(f"  → BODY AREA action: tapping at ({body_x}, {body_y})")
        
        # C. INPUT FIELD ACTIONS - Use EditText detection
        if not target_bounds and is_input_action:
            print(f"  → Detected INPUT FIELD action, looking for EditText widgets...")
            edit_bounds = adb.find_first_edit_text()
            if edit_bounds:
                target_bounds = edit_bounds
                target_label = "EditText"
                print(f"  ✓ Found EditText at: {target_bounds}")
        
        # D. PERMISSION BUTTON ACTIONS - Find actual clickable buttons, not question text!
        if not target_bounds and is_permission_action:
            print(f"  → Detected PERMISSION action, looking for clickable buttons...")
            # Try common permission button texts
            for btn_text in ["Allow", "ALLOW", "OK", "Deny", "Cancel", "Accept"]:
                if btn_text.lower() in step_lower:
                    bounds = adb.find_button_by_text(btn_text)
                    if bounds:
                        target_bounds = bounds
                        target_label = f"Button:{btn_text}"
                        print(f"  ✓ Found button '{btn_text}' at: {target_bounds}")
                        break
        
        # E. Toggle/Switch detection
        if not target_bounds and ("toggle" in step_lower or "switch" in step_lower or "enable" in step_lower):
            toggle_bounds = adb.find_toggle_or_switch()
            if toggle_bounds:
                target_bounds = toggle_bounds
                target_label = "Toggle"
                print(f"  → Detected toggle at: {target_bounds}")
        
        # F. BUTTON/LABEL ACTIONS - Use text matching to find buttons
        if not target_bounds and "tap" in step_lower:
            candidates = extract_target_text(next_step)
            print(f"  → Looking for UI elements matching: {candidates}")
            
            # First try to find actual clickable buttons with this text
            for candidate in candidates:
                bounds = adb.find_button_by_text(candidate)
                if bounds:
                    target_bounds = bounds
                    target_label = f"Button:{candidate}"
                    print(f"  ✓ Found button '{candidate}' at bounds: {bounds}")
                    break
            
            # Fallback: find any element with matching text
            if not target_bounds:
                for candidate in candidates:
                    bounds = adb.find_bounds_by_text(candidate)
                    if bounds:
                        target_bounds = bounds
                        target_label = candidate
                        print(f"  ✓ Found '{candidate}' at bounds: {bounds}")
                        break
            
            # If no match by text, try by keywords (partial matching)
            if not target_bounds and candidates:
                bounds = adb.find_bounds_by_keywords(*candidates)
                if bounds:
                    target_bounds = bounds
                    target_label = "keyword-match"
                    print(f"  ✓ Found element by keyword at bounds: {bounds}")
        
        # E. TYPE actions also need EditText (but NOT if we already have body area bounds)
        if not target_bounds and "type" in step_lower and not is_body_action:
            edit_bounds = adb.find_first_edit_text()
            if edit_bounds:
                target_bounds = edit_bounds
                target_label = "EditText"
                print(f"  → Detected EditText for typing: {target_bounds}")
        
        # Log if we couldn't find the element
        if "tap" in step_lower and not target_bounds:
            print(f"  ⚠ Could not find element in UI dump. Available elements: {list(all_bounds_dict.keys()) if all_bounds_dict else 'none'}")
        
        # 4. Execute
        try:
            action_data = executor.execute_step(next_step, screenshot, target_hint=target_bounds)
            print(f"Executor Action: {action_data}")
            
            if action_data["action"] == "tap":
                x, y = action_data["x"], action_data["y"]
                adb.tap(x, y)
                tap_log.append({"x": x, "y": y, "step": step_count + 1, "desc": next_step, "screenshot": screenshot, "target_bounds": target_bounds})
                save_tap_overlay(screenshot, x, y, step_count + 1, target_bounds=target_bounds, target_label=target_label, all_bounds=all_bounds_dict)
                
                # Remember element location for future use (only for elements found via UI dump, not vision)
                # Vision-found elements will be verified by checking if screen changed
                if target_bounds is not None:  # Only memorize if we had UI dump bounds
                    for keyword in ["expand", "menu", "back"]:
                        if keyword in next_step.lower():
                            agent_memory.remember_element_location(keyword, x, y, context=visible_ui[:100])
                            break
                
                # For gear/settings, we'll verify on next iteration if we actually got to settings
                if "gear" in next_step.lower() or "settings" in next_step.lower():
                    agent_memory.set_session_context("pending_gear_location", {"x": x, "y": y})
                
                # Give app time to respond (longer for app launches, shorter for UI interactions)
                if "obsidian" in next_step.lower() or "app" in next_step.lower() or "open" in next_step.lower():
                    print("Waiting 3s for app to load...")
                    time.sleep(3)
                else:
                    time.sleep(1)
            elif action_data["action"] == "type":
                text_to_type = action_data.get("text", "")
                print(f"Typing text: '{text_to_type}'")
                
                # Check if previous action was tapping body area - if so, don't re-tap, cursor is already positioned
                prev_action = history[-2] if len(history) >= 2 else ""
                was_body_tap = "body" in prev_action.lower() if prev_action else False
                
                if was_body_tap:
                    # Cursor should already be in body area from previous tap - just type
                    print(f"  → Typing in body (cursor already positioned from body tap)")
                elif target_bounds and target_label != "BodyArea":
                    # Tap to focus the field (for title/input fields, not after body tap)
                    x1, y1, x2, y2 = target_bounds
                    focus_x, focus_y = (x1 + x2) // 2, (y1 + y2) // 2
                    print(f"  1. Tapping field at ({focus_x}, {focus_y}) to focus...")
                    adb.tap(focus_x, focus_y)
                    time.sleep(0.5)  # Wait for field focus
                
                # Now type
                print(f"  2. Typing...")
                adb.type_text(text_to_type)
                print(f"  ✓ Done typing")
                time.sleep(0.3)
                
                # Dismiss keyboard by pressing Back (keycode 4) - this hides keyboard without submitting
                print(f"  3. Dismissing keyboard...")
                adb.key_event(4)  # KEYCODE_BACK - dismisses keyboard
                time.sleep(0.5)
            elif action_data["action"] == "wait":
                time.sleep(action_data["seconds"])
            elif action_data["action"] == "key":
                keycode = action_data.get("keycode", 0)
                print(f"Pressing key: {keycode}")
                adb.key_event(keycode)
                time.sleep(0.5)
            elif action_data["action"] == "swipe":
                start_x = int(action_data["start_x"])
                start_y = int(action_data["start_y"])
                end_x = int(action_data["end_x"])
                end_y = int(action_data["end_y"])
                print(f"Executing: SWIPE ({start_x},{start_y}) → ({end_x},{end_y})")
                adb.swipe(start_x, start_y, end_x, end_y)
                time.sleep(0.5)
                
            time.sleep(1)  # UI settle time
            step_count += 1
            
        except Exception as e:
            print(f"Execution Error: {e}")
            break

def save_tap_overlay(image_path, x, y, idx, radius=24, target_bounds=None, target_label="target", all_bounds=None):
    """Save an annotated copy showing taps (red), target (lime), and all bounds (gray) on the screenshot."""
    try:
        os.makedirs("debug_taps", exist_ok=True)
        with Image.open(image_path) as im:
            draw = ImageDraw.Draw(im)
            
            # Draw all detected bounds in gray (for context)
            if all_bounds:
                for label, (x1, y1, x2, y2) in all_bounds.items():
                    draw.rectangle([(x1, y1), (x2, y2)], outline="gray", width=1)
            
            # Draw target bounds in lime (highlighted)
            if target_bounds:
                x1, y1, x2, y2 = target_bounds
                draw.rectangle([(x1, y1), (x2, y2)], outline="lime", width=4)
                draw.text((x2 + 6, y1), target_label, fill="lime")
            
            # Draw tap location in red with number
            bbox = [(x - radius, y - radius), (x + radius, y + radius)]
            draw.ellipse(bbox, outline="red", width=4)
            draw.text((x + radius + 6, y - radius), f"#{idx}", fill="red")
            
            out_path = os.path.join("debug_taps", f"tap_{idx}.png")
            im.save(out_path)
            print(f"  ✓ Saved overlay: {out_path}")
    except Exception as e:
        print(f"  ! Failed to save tap overlay: {e}")

def setup_fresh_state(adb: ADBTools, package_name: str = "md.obsidian"):
    """
    Fast wipe: Clear app data and delete vaults without reinstalling.
    This makes the app forget everything (settings, vaults, permissions) instantly.
    """
    print("=" * 50)
    print("⚡ FAST WIPE: Resetting app to fresh state...")
    print("=" * 50)
    
    # 1. Force stop the app (if running)
    print("  → Force stopping app...")
    adb.device.shell(f"am force-stop {package_name}")
    time.sleep(0.5)
    
    # 2. Clear app data (this resets settings, permissions, internal storage)
    print("  → Clearing app data...")
    result = adb.device.shell(f"pm clear {package_name}")
    print(f"    Result: {result.strip()}")
    
    # 3. Delete any vaults in common locations
    vault_locations = [
        "/sdcard/Documents/",
        "/sdcard/Obsidian/",
        "/sdcard/Download/",
    ]
    
    for location in vault_locations:
        # List and remove any vault-like directories
        print(f"  → Cleaning vault location: {location}")
        try:
            # Remove entire Obsidian-related folders
            adb.device.shell(f"rm -rf {location}*Vault* 2>/dev/null")
            adb.device.shell(f"rm -rf {location}*vault* 2>/dev/null")
            adb.device.shell(f"rm -rf {location}.obsidian 2>/dev/null")
        except Exception:
            pass
    
    # 4. Also clean the app's internal vault storage location
    print("  → Cleaning internal app storage...")
    adb.device.shell(f"rm -rf /data/data/{package_name}/files/* 2>/dev/null")
    
    # 5. Wait for system to settle
    time.sleep(1)
    
    print("  ✓ Fresh state ready! App will behave like first launch.")
    print("=" * 50)

if __name__ == "__main__":
    import sys
    
    # Parse command line options
    skip_wipe = "--no-wipe" in sys.argv or "-n" in sys.argv
    clear_memory = "--clear-memory" in sys.argv or "-m" in sys.argv
    start_test = 1
    for arg in sys.argv:
        if arg.startswith("--test="):
            start_test = int(arg.split("=")[1])
        elif arg == "-t" and sys.argv.index(arg) + 1 < len(sys.argv):
            start_test = int(sys.argv[sys.argv.index(arg) + 1])
    
    # Clear memory if requested
    if clear_memory:
        print("\n🧹 Clearing agent memory...")
        if os.path.exists("agent_memory.json"):
            os.remove("agent_memory.json")
            print("  ✓ Memory cleared")
        # Reload fresh memory
        agent_memory.data = AgentMemory().data
    
    # Initialize ADB
    adb = ADBTools()
    
    # Fast wipe: Reset Obsidian to fresh state before testing (unless skipped)
    if not skip_wipe:
        setup_fresh_state(adb, package_name="md.obsidian")
    else:
        print("\n⚡ Skipping wipe (--no-wipe flag used)")
    
    # Test Case 1: Create a new vault
    if start_test <= 1:
        print("\n" + "=" * 60)
        print("TEST 1: Create a new vault")
        print("=" * 60)
        run_test_case("Open Obsidian, create a new vault named 'internVault', and enter the editor.")
    
    # Test Case 2: Create a note with content
    if start_test <= 2:
        print("\n" + "=" * 60)
        print("TEST 2: Create a new note with content")
        print("=" * 60)
        run_test_case("Create a new note titled 'Meeting Notes' and type the text 'Daily Standup' into the body.")
    
    # Test Case 3: Verify icon color
    if start_test <= 3:
        print("\n" + "=" * 60)
        print("TEST 3: Verify Appearance icon color")
        print("=" * 60)
        run_test_case("Go to Settings and tap on 'Appearance'. Look at the current screen and check if there is any RED colored icon visible. If you see 'Appearance' settings content (like 'Base color scheme', 'Accent color', 'Font'), verify: is there a red icon anywhere? If no red icon is visible, report FAIL: No red icon found.")
    
    # Test Case 4: Verify agent reports missing element (SHOULD FAIL)
    if start_test <= 4:
        print("\n" + "=" * 60)
        print("TEST 4: Find non-existent 'Print to PDF' button (SHOULD FAIL)")
        print("=" * 60)
        result = run_test_case("Find and click the 'Print to PDF' button in the main file menu. Search thoroughly in all menus and options. If after checking all available menus you cannot find this button, report FAIL: Element not found.")
        
        # For this test, we actually WANT it to fail (report element not found)
        if result is False:
            print("\n✓ TEST 4 PASSED: Agent correctly reported that 'Print to PDF' button does not exist")
        else:
            print("\n✗ TEST 4 FAILED: Agent should have reported element not found, but didn't")