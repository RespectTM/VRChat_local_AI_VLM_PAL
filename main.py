import cv2
import asyncio
from pythonosc import udp_client
import warnings
import mss
import numpy as np
import json
import math
import pyautogui
import time
import random
import datetime
import os
import torch
try:
    import torch_directml as dml
    _HAS_DML = True
except Exception:
    dml = None
    _HAS_DML = False
from transformers import (
    BlipProcessor,
    BlipForConditionalGeneration,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    AutoModelForCausalLM,
    AutoTokenizer,
)
from PIL import Image
import pygetwindow as gw
import re

# Suppress warnings
warnings.filterwarnings("ignore")

# --- CONFIG ---
VRC_IP = "127.0.0.1"
VRC_PORT = 9000
VISION_INTERVAL = 0.333  # ~3fps
THINKING_MODEL_NAME = "gpt2"
CHAT_MODEL_NAME = "gpt2"

class VRChatAI:
    def __init__(self):
        # Select device: CUDA -> DirectML (AMD on Windows) -> CPU
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            device_desc = torch.cuda.get_device_name(0)
        elif _HAS_DML:
            self.device = dml.device()
            device_desc = "DirectML"
        else:
            self.device = torch.device("cpu")
            device_desc = "cpu"
        print(f"Using device: {self.device} ({device_desc})")
        
        # OSC client
        self.client = udp_client.SimpleUDPClient(VRC_IP, VRC_PORT)
        
        # Models
        self.vision_processor = None
        self.vision_model = None
        self.thinking_model = None
        self.thinking_tokenizer = None
        self.chat_model = None
        self.chat_tokenizer = None
        
        # Load logs early so load failures can be recorded
        # Logging
        self.vision_log = open("vision.log", "a", encoding='utf-8')
        self.thinking_log = open("thinking.log", "a", encoding='utf-8')
        self.chat_log = open("chat.log", "a", encoding='utf-8')
        self.events_log = open("events.log", "a", encoding='utf-8')
        self.errors_log = open("errors.log", "a", encoding='utf-8')
        self.osc_log = open("osc.log", "a", encoding='utf-8')
        self.surroundings_log = open("surroundings.log", "a", encoding='utf-8')

        # Load models
        self.load_models()

        # State
        self.latest_vision = ""
        self.chat_history = []
        self.memory = []
        
        # Logging
        self.vision_log = open("vision.log", "a", encoding='utf-8')
        self.thinking_log = open("thinking.log", "a", encoding='utf-8')
        self.chat_log = open("chat.log", "a", encoding='utf-8')
        self.events_log = open("events.log", "a", encoding='utf-8')
        self.errors_log = open("errors.log", "a", encoding='utf-8')
        self.osc_log = open("osc.log", "a", encoding='utf-8')
        self.surroundings_log = open("surroundings.log", "a", encoding='utf-8')
        
        # VRChat window
        self.vrchat_window = None
        
        self.log_event("VRChatAI initialized with BLIP vision and GPT-2 models.")

    def load_models(self):
        try:
            # Try Vision: Moondream2 (local) first
            try:
                vision_path = "models/moondream2"
                self.vision_model = AutoModelForCausalLM.from_pretrained(vision_path, trust_remote_code=True).to(self.device)
                self.vision_tokenizer = AutoTokenizer.from_pretrained(vision_path, trust_remote_code=True)
                self.vision_mode = "moondream2"
                self.log_event("Vision model (moondream2) loaded successfully.")
            except Exception:
                # Fallback to BLIP if Moondream isn't available or fails to load
                self.vision_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
                self.vision_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(self.device)
                self.vision_mode = "blip"
                self.log_event("Vision model (BLIP) loaded as fallback.")
            
            # Thinking: GPT-2
            self.thinking_model = GPT2LMHeadModel.from_pretrained(THINKING_MODEL_NAME).to(self.device)
            self.thinking_tokenizer = GPT2Tokenizer.from_pretrained(THINKING_MODEL_NAME)
            self.thinking_tokenizer.pad_token = self.thinking_tokenizer.eos_token
            self.log_event("Thinking model (GPT-2) loaded successfully.")
            
            # Chat: GPT-2
            self.chat_model = GPT2LMHeadModel.from_pretrained(CHAT_MODEL_NAME).to(self.device)
            self.chat_tokenizer = GPT2Tokenizer.from_pretrained(CHAT_MODEL_NAME)
            self.chat_tokenizer.pad_token = self.chat_tokenizer.eos_token
            self.log_event("Chat model (GPT-2) loaded successfully.")
            
        except Exception as e:
            self.log_error(f"Error loading models: {e}")
            raise

    def log_event(self, message):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.events_log.write(f"[{timestamp}] {message}\n")
        self.events_log.flush()

    def log_error(self, message):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.errors_log.write(f"[{timestamp}] {message}\n")
        self.errors_log.flush()

    def log_vision(self, message):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.vision_log.write(f"[{timestamp}] {message}\n")
        self.vision_log.flush()

    def log_thinking(self, message):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.thinking_log.write(f"[{timestamp}] {message}\n")
        self.thinking_log.flush()

    def log_chat(self, message):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.chat_log.write(f"[{timestamp}] {message}\n")
        self.chat_log.flush()

    def log_osc(self, message):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.osc_log.write(f"[{timestamp}] {message}\n")
        self.osc_log.flush()

    def log_surroundings(self, message):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.surroundings_log.write(f"[{timestamp}] {message}\n")
        self.surroundings_log.flush()

    def find_vrchat_window(self):
        try:
            windows = gw.getWindowsWithTitle("VRChat")
            if windows:
                self.vrchat_window = windows[0]
                self.log_event(f"VRChat window found: {self.vrchat_window}")
                return True
            else:
                self.log_error("VRChat window not found.")
                return False
        except Exception as e:
            self.log_error(f"Error finding VRChat window: {e}")
            return False

    def capture_screen(self):
        with mss.mss() as sct:
            if self.vrchat_window:
                bbox = (self.vrchat_window.left, self.vrchat_window.top, self.vrchat_window.right, self.vrchat_window.bottom)
                screenshot = sct.grab(bbox)
            else:
                screenshot = sct.grab(sct.monitors[1])  # Primary monitor
            img = np.array(screenshot)
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            return img

    def process_vision(self, image):
        try:
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
            # If moondream2 is loaded, use its vision QA API
            if getattr(self, "vision_mode", None) == "moondream2":
                prompt = "Describe the VRChat scene in detail, including avatars, chat messages, actions, and environment."
                with torch.no_grad():
                    try:
                        response = self.vision_model.answer_question(pil_image, prompt, self.vision_tokenizer, max_new_tokens=120, num_beams=1, temperature=0.7)
                        caption = response.strip() if isinstance(response, str) else str(response)
                    except Exception as e:
                        # Some custom model implementations may require a different call
                        self.log_error(f"Moondream answer_question failed: {e}")
                        caption = ""
            else:
                inputs = self.vision_processor(pil_image, return_tensors="pt")
                # Move tensors to device
                for k, v in inputs.items():
                    inputs[k] = v.to(self.device)
                with torch.no_grad():
                    out = self.vision_model.generate(**inputs, max_new_tokens=50)
                # decoding depends on processor
                caption = self.vision_processor.decode(out[0], skip_special_tokens=True)
            
            # VRChat-specific post-processing
            caption = self.post_process_vision(caption)
            self.latest_vision = caption
            self.log_vision(caption)
            return caption
        except Exception as e:
            self.log_error(f"Vision processing error: {e}")
            return "Vision processing failed."

    def post_process_vision(self, text):
        # Enhance for VRChat context
        text = re.sub(r'\b(avatar|player|user)\b', r'\1 (VRChat)', text, flags=re.IGNORECASE)
        text = re.sub(r'\b(chat|message)\b', r'\1 (in-game)', text, flags=re.IGNORECASE)
        return text

    def think_and_decide(self, vision_text):
        try:
            prompt = f"Based on this VRChat scene: {vision_text}\nWhat should the AI do next? Think step-by-step and decide on an action."
            
            inputs = self.thinking_tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
            for k, v in inputs.items():
                inputs[k] = v.to(self.device)
            with torch.no_grad():
                outputs = self.thinking_model.generate(
                    **inputs, max_new_tokens=50, num_beams=2, temperature=0.8, do_sample=True,
                    pad_token_id=self.thinking_tokenizer.eos_token_id
                )
            thought = self.thinking_tokenizer.decode(outputs[0], skip_special_tokens=True)
            self.log_thinking(thought)
            
            # Simple decision: if "chat" in thought, generate chat
            if "chat" in thought.lower() or "talk" in thought.lower():
                return "chat"
            else:
                return "observe"
        except Exception as e:
            self.log_error(f"Thinking error: {e}")
            return "observe"

    def generate_chat(self, vision_text):
        try:
            prompt = f"In VRChat, seeing: {vision_text}\nGenerate a short, friendly chat message as the AI."
            
            inputs = self.chat_tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
            for k, v in inputs.items():
                inputs[k] = v.to(self.device)
            with torch.no_grad():
                outputs = self.chat_model.generate(
                    **inputs, max_new_tokens=30, num_beams=2, temperature=0.9, do_sample=True,
                    pad_token_id=self.chat_tokenizer.eos_token_id
                )
            message = self.chat_tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Analyze worthiness
            if self.analyze_message_worthiness(message):
                self.send_chat(message)
                self.log_chat(f"Sent: {message}")
            else:
                self.log_chat(f"Skipped low-quality message: {message}")
        except Exception as e:
            self.log_error(f"Chat generation error: {e}")

    def analyze_message_worthiness(self, message):
        # Simple quality check
        if len(message.strip()) < 5:
            return False
        if any(word in message.lower() for word in ["spam", "repeat", "nonsense"]):
            return False
        # Check for repetition in history
        if message in [c['message'] for c in self.chat_history[-5:]]:
            return False
        return True

    def send_chat(self, message):
        try:
            # Send via OSC (adjust address as needed)
            self.client.send_message("/chatbox/input", [message, True])
            self.log_osc(f"Chat sent: {message}")
            self.chat_history.append({'message': message, 'timestamp': time.time()})
            if len(self.chat_history) > 20:
                self.chat_history.pop(0)
        except Exception as e:
            self.log_error(f"OSC send error: {e}")

    async def run(self):
        self.find_vrchat_window()
        last_vision_time = 0
        
        while True:
            current_time = time.time()
            if current_time - last_vision_time >= VISION_INTERVAL:
                try:
                    image = self.capture_screen()
                    vision_text = self.process_vision(image)
                    self.log_surroundings(vision_text)
                    
                    action = self.think_and_decide(vision_text)
                    if action == "chat":
                        self.generate_chat(vision_text)
                    
                    last_vision_time = current_time
                except Exception as e:
                    self.log_error(f"Main loop error: {e}")
            
            await asyncio.sleep(0.1)  # Small sleep to prevent busy loop

if __name__ == "__main__":
    ai = VRChatAI()
    asyncio.run(ai.run())
