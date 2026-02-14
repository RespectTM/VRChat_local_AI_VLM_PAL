import asyncio
from pythonosc import udp_client
import time
import cv2
import mss
import numpy as np
import torch
import torch_directml as dml
from transformers import ViTImageProcessor, VisionEncoderDecoderModel, AutoTokenizer
import warnings

# VRChat OSC Configuration
VRC_IP = "127.0.0.1"
VRC_PORT = 9000

class VRChatAI:
    def __init__(self):
        self.client = udp_client.SimpleUDPClient(VRC_IP, VRC_PORT)
        print(f"[INFO] Connected to VRChat OSC at {VRC_IP}:{VRC_PORT}")

        # Device
        self.device = dml.device()
        print(f"[INFO] Using device: {self.device}")

        # Models
        self.vision_processor = None
        self.vision_model = None
        self.vision_tokenizer = None
        
        # Thinking model for decision making
        self.thinking_tokenizer = None
        self.thinking_model = None

        # State
        self.latest_vision_data = "No vision data yet."
        self.latest_thought = "No thoughts yet."
        self.message_counter = 0  # Track when to send messages

    def send_message(self, address, value):
        """Send OSC message to VRChat."""
        try:
            self.client.send_message(address, value)
            print(f"[OSC] Sent {address}: {value}")
        except Exception as e:
            print(f"[ERROR] Failed to send OSC: {e}")

    def is_vrchat_running(self):
        """Check if VRChat game is currently running."""
        try:
            import pygetwindow as gw
            # Try different possible window titles
            possible_titles = ['VRChat', 'VRChat ', 'VRChat.exe']
            for title in possible_titles:
                windows = gw.getWindowsWithTitle(title)
                # Filter out non-game windows
                game_windows = []
                for w in windows:
                    skip_keywords = ['visual studio', 'file explorer', 'vscode', 'code', 'explorer']
                    should_skip = any(keyword in w.title.lower() for keyword in skip_keywords)
                    if not should_skip and w.visible and w.width > 200 and w.height > 200:
                        game_windows.append(w)
                if game_windows:
                    return True
            return False
        except ImportError:
            return False

    def load_models(self):
        """Load vision model and thinking model."""
        # Load vision model
        try:
            print("[INFO] Loading vision model on GPU...")
            self.vision_processor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
            self.vision_model = VisionEncoderDecoderModel.from_pretrained(
                "nlpconnect/vit-gpt2-image-captioning",
                use_safetensors=True,
                torch_dtype=torch.float32
            ).to(self.device)
            self.vision_tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

            print("[INFO] Vision model loaded successfully on GPU.")
        except Exception as e:
            print(f"[ERROR] Failed to load vision model on GPU: {e}")
            print("[INFO] Falling back to CPU...")
            try:
                self.device = "cpu"
                self.vision_processor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
                self.vision_model = VisionEncoderDecoderModel.from_pretrained(
                    "nlpconnect/vit-gpt2-image-captioning",
                    use_safetensors=True,
                    torch_dtype=torch.float32
                ).to(self.device)
                self.vision_tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
                print("[INFO] Vision model loaded successfully on CPU.")
            except Exception as e2:
                print(f"[ERROR] Failed to load vision model on CPU too: {e2}")
                print("[INFO] Trying alternative model...")
                try:
                    # Try a different, potentially more compatible model
                    self.vision_processor = ViTImageProcessor.from_pretrained("microsoft/DialoGPT-small")
                    self.vision_model = VisionEncoderDecoderModel.from_pretrained("microsoft/DialoGPT-small").to(self.device)
                    self.vision_tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
                    print("[INFO] Alternative vision model loaded.")
                except Exception as e3:
                    print(f"[ERROR] All vision model loading attempts failed: {e3}")
                    self.vision_model = None
                    self.vision_processor = None
                    self.vision_tokenizer = None
        
        # Load thinking model
        try:
            print("[INFO] Loading thinking model on GPU...")
            from transformers import AutoModelForCausalLM
            self.thinking_tokenizer = AutoTokenizer.from_pretrained("gpt2")
            self.thinking_model = AutoModelForCausalLM.from_pretrained("gpt2").to(self.device)
            
            # Set pad token
            if self.thinking_tokenizer.pad_token is None:
                self.thinking_tokenizer.pad_token = self.thinking_tokenizer.eos_token
            
            print("[INFO] Thinking model loaded successfully on GPU.")
        except Exception as e:
            print(f"[ERROR] Failed to load thinking model on GPU: {e}")
            print("[INFO] Falling back to CPU for thinking model...")
            try:
                self.device = "cpu"
                from transformers import AutoModelForCausalLM
                self.thinking_tokenizer = AutoTokenizer.from_pretrained("gpt2")
                self.thinking_model = AutoModelForCausalLM.from_pretrained("gpt2").to(self.device)
                
                if self.thinking_tokenizer.pad_token is None:
                    self.thinking_tokenizer.pad_token = self.thinking_tokenizer.eos_token
                
                print("[INFO] Thinking model loaded successfully on CPU.")
            except Exception as e2:
                print(f"[ERROR] Failed to load thinking model: {e2}")
                self.thinking_model = None
                self.thinking_tokenizer = None

    def capture_screenshot(self):
        """Capture ONLY the VRChat window (no fallback to monitor)."""
        try:
            with mss.mss() as sct:
                # Find VRChat window
                try:
                    import pygetwindow as gw
                    # Try different possible window titles
                    possible_titles = ['VRChat', 'VRChat ', 'VRChat.exe']
                    vrchat_window = None

                    for title in possible_titles:
                        windows = gw.getWindowsWithTitle(title)
                        if windows:
                            # Filter out non-game windows (like VS Code, File Explorer)
                            game_windows = []
                            for w in windows:
                                # Skip windows that contain development tool keywords
                                skip_keywords = ['visual studio', 'file explorer', 'vscode', 'code', 'explorer']
                                should_skip = any(keyword in w.title.lower() for keyword in skip_keywords)
                                if not should_skip and w.visible and w.width > 200 and w.height > 200:
                                    game_windows.append(w)

                            if game_windows:
                                # Select the window with the most typical game aspect ratio (16:9 or similar)
                                # Games are usually wider than tall, unlike development tools
                                vrchat_window = max(game_windows, key=lambda w: w.width / max(w.height, 1))
                                break

                    if vrchat_window:
                        # Check if window is reasonable size and visible
                        if vrchat_window.width > 200 and vrchat_window.height > 200:
                            bbox = (vrchat_window.left, vrchat_window.top, vrchat_window.right, vrchat_window.bottom)
                            screenshot = np.array(sct.grab(bbox))
                            print(f"[VISION] Captured VRChat GAME window: {bbox} ({vrchat_window.width}x{vrchat_window.height})")
                            frame = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2BGR)
                            return frame
                        else:
                            print(f"[WARNING] VRChat window too small: {vrchat_window.width}x{vrchat_window.height}")
                            return None
                    else:
                        # Debug: list all windows to help identify VRChat
                        all_windows = gw.getAllTitles()
                        vrchat_related = [title for title in all_windows if 'vrchat' in title.lower() or 'vrc' in title.lower()]
                        print(f"[DEBUG] Found VRChat-related windows: {vrchat_related}")
                        print("[WARNING] No suitable VRChat game window found. Make sure VRChat is running and visible.")
                        return None
                except ImportError:
                    print("[ERROR] pygetwindow not available. Cannot detect VRChat window.")
                    return None
        except Exception as e:
            print(f"[ERROR] Screenshot capture failed: {e}")
            return None

    def process_vision(self, frame):
        """Process frame with vision model to understand surroundings."""
        if self.vision_model is None:
            print("[ERROR] Vision model is not loaded")
            return "Vision model not loaded."

        try:
            small_frame = cv2.resize(frame, (1280, 720))
            _, buffer = cv2.imencode('.jpg', small_frame)

            inputs = self.vision_processor(frame, return_tensors="pt").to(self.device)
            output = self.vision_model.generate(**inputs, max_new_tokens=50)
            description = self.vision_tokenizer.decode(output[0], skip_special_tokens=True)

            self.latest_vision_data = description
            return description
        except Exception as e:
            print(f"[ERROR] Vision processing failed: {e}")
            return "Error processing vision."

    def think(self, vision_description):
        """Process vision data with thinking model to make decisions."""
        if self.thinking_model is None:
            print("[ERROR] Thinking model is not loaded")
            return "Thinking model not loaded."

        try:
            # Create a prompt for the thinking model
            prompt = f"I am an AI in VRChat. I see: {vision_description}. What should I do next? I should"
            
            # Tokenize input
            inputs = self.thinking_tokenizer.encode(prompt, return_tensors="pt").to(self.device)
            
            # Generate response
            with torch.no_grad():
                outputs = self.thinking_model.generate(
                    inputs,
                    max_length=inputs.shape[1] + 50,
                    num_return_sequences=1,
                    temperature=0.8,
                    do_sample=True,
                    pad_token_id=self.thinking_tokenizer.pad_token_id,
                    eos_token_id=self.thinking_tokenizer.eos_token_id,
                    attention_mask=torch.ones_like(inputs)
                )
            
            # Decode response
            full_response = self.thinking_tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract just the generated part (remove the prompt)
            thought = full_response[len(prompt):].strip()
            
            # Clean up the response
            if thought:
                # Take first sentence or reasonable chunk
                thought = thought.split('.')[0].split('!')[0].split('?')[0] + '.'
            
            self.latest_thought = thought
            return thought
            
        except Exception as e:
            print(f"[ERROR] Thinking failed: {e}")
            return "Error thinking."

    def write_message(self, vision_description, thought):
        """Generate a chat message based on vision and thoughts."""
        if self.thinking_model is None:
            print("[ERROR] Thinking model is not loaded")
            return "Model not loaded."

        try:
            # Create a prompt for message generation
            prompt = f"I am an AI in VRChat. I see: {vision_description}. I'm thinking: {thought}. What should I say in chat? \""
            
            # Tokenize input
            inputs = self.thinking_tokenizer.encode(prompt, return_tensors="pt").to(self.device)
            
            # Generate response
            with torch.no_grad():
                outputs = self.thinking_model.generate(
                    inputs,
                    max_length=inputs.shape[1] + 30,  # Shorter for chat messages
                    num_return_sequences=1,
                    temperature=0.9,  # Slightly more creative for chat
                    do_sample=True,
                    pad_token_id=self.thinking_tokenizer.pad_token_id,
                    eos_token_id=self.thinking_tokenizer.eos_token_id,
                    attention_mask=torch.ones_like(inputs)
                )
            
            # Decode response
            full_response = self.thinking_tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract just the message part (between quotes)
            message = full_response[len(prompt):].strip()
            
            # Clean up - take until closing quote or reasonable length
            if '"' in message:
                message = message.split('"')[0]
            else:
                message = message.split('.')[0].split('!')[0].split('?')[0]
            
            # Limit message length for chat
            if len(message) > 100:
                message = message[:97] + "..."
            
            return message.strip()
            
        except Exception as e:
            print(f"[ERROR] Message generation failed: {e}")
            return "Hello everyone!"

    async def run_ai_loop(self):
        """Main AI loop: capture surroundings, understand them, and think about decisions."""
        print("[INFO] Starting vision + thinking AI loop...")
        while True:
            try:
                # Check if VRChat is running
                if not self.is_vrchat_running():
                    print("[WARNING] VRChat not detected. Waiting for VRChat to start...")
                    await asyncio.sleep(10.0)
                    continue

                # Capture and process vision
                frame = self.capture_screenshot()
                if frame is not None:
                    description = self.process_vision(frame)
                    print(f"[VISION] {description}")

                    # Think about what to do based on vision
                    thought = self.think(description)
                    print(f"[THINKING] {thought}")

                    # Generate a chat message based on vision and thoughts
                    message = self.write_message(description, thought)
                    print(f"[MESSAGE] {message}")

                    # Occasionally send message to VRChat chat
                    self.message_counter += 1
                    should_send = False
                    
                    # Send every 5th message, or randomly 10% of the time
                    import random
                    if self.message_counter % 5 == 0 or random.random() < 0.1:
                        should_send = True
                        self.message_counter = 0  # Reset counter after sending
                    
                    if should_send and message and len(message.strip()) > 0:
                        # Send to VRChat chat
                        self.send_message("/chatbox/input", [message, True, False])
                        print(f"[CHAT] Sent to VRChat: {message}")

                    # Log vision, thoughts, and message
                    with open("surroundings.log", "a", encoding='utf-8') as f:
                        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - VISION: {description} | THOUGHT: {thought} | MESSAGE: {message} | SENT: {should_send}\n")
                else:
                    print("[WARNING] Failed to capture VRChat window")
                    await asyncio.sleep(5.0)
                    continue

                # Wait before next perception
                await asyncio.sleep(5.0)  # Slower for development
            except Exception as e:
                print(f"[ERROR] AI loop error: {e}")
                await asyncio.sleep(10.0)

async def main():
    ai = VRChatAI()
    ai.load_models()

    # Check if VRChat is running
    if not ai.is_vrchat_running():
        print("[WARNING] VRChat not detected. Please start VRChat before running the AI.")
        print("[INFO] The AI will wait for VRChat to start...")

    # Test connection
    ai.send_message("/chatbox/input", ["AI Vision + Thinking + Chat System Online - Seeing, thinking, and chatting about VRChat surroundings...", True, False])

    # Start perception loop
    await ai.run_ai_loop()

if __name__ == "__main__":
    asyncio.run(main())