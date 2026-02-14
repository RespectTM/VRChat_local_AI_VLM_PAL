# VRChat Local AI Vision-Language Model PAL

A sophisticated AI system that enables VRChat avatars to perceive their surroundings, think about situations, and communicate naturally - all running locally on AMD GPUs.

## 🤖 System Overview

This AI system consists of **3 specialized models** working together:

### 🏗️ Architecture
```
VRChat Window → Vision Model → Scene Description
    ↓
Scene Description → Thinking Model → Decision/Thought
    ↓
Description + Thought → Chat Model → VRChat Message
```

### 🎯 Key Features
- **Local Processing**: No external APIs or cloud services required
- **AMD GPU Acceleration**: Optimized for AMD GPUs using DirectML
- **Game-Only Capture**: Intelligently captures only VRChat window (filters out development tools)
- **Contextual AI**: Messages and thoughts based on real-time VRChat surroundings
- **OSC Integration**: Seamless communication with VRChat
- **Comprehensive Logging**: All AI activities logged for analysis

## 🚀 Quick Start

### Prerequisites
- **VRChat** running and OSC enabled (port 9000)
- **Python 3.10+**
- **AMD GPU** (RDNA architecture recommended)
- **Windows 10/11**

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/VRChat_local_AI_VLM_PAL.git
   cd VRChat_local_AI_VLM_PAL
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the AI**
   ```bash
   python main.py
   ```

## 📋 System Components

### 👁️ Vision Model (`nlpconnect/vit-gpt2-image-captioning`)
- **Purpose**: Understands VRChat surroundings by analyzing screenshots
- **Input**: VRChat game window captures
- **Output**: Natural language descriptions ("a room with furniture and people")
- **GPU**: AMD GPU accelerated

### 🧠 Thinking Model (`gpt2`)
- **Purpose**: Makes decisions and plans actions based on vision
- **Input**: Vision descriptions
- **Output**: Thoughts and decisions ("I should explore this area")
- **GPU**: AMD GPU accelerated

### 💬 Chat Model (`gpt2`)
- **Purpose**: Generates natural conversation messages
- **Input**: Vision descriptions + thoughts
- **Output**: Contextual chat messages sent to VRChat
- **Examples**:
  - "That looks interesting! What's happening?"
  - "I see some people over there, should we go say hi?"
  - "This place is amazing!"

## 🔧 Technical Details

### Hardware Requirements
- **GPU**: AMD Radeon RX 5000/6000/7000 series or newer
- **RAM**: 8GB+ recommended
- **Storage**: ~2GB for models
- **OS**: Windows 10/11

### Dependencies
- `torch-directml`: AMD GPU acceleration
- `transformers`: Hugging Face models
- `opencv-python`: Image processing
- `mss`: Screen capture
- `pygetwindow`: Window detection
- `python-osc`: VRChat communication

### Model Specifications
| Model | Size | Purpose | Memory Usage |
|-------|------|---------|--------------|
| ViT-GPT2 | ~444MB | Vision | ~1GB VRAM |
| GPT-2 | ~148MB | Thinking/Chat | ~500MB VRAM |
| **Total** | **~592MB** | **All** | **~1.5GB VRAM** |

## 🎮 VRChat Integration

### OSC Configuration
- **IP**: 127.0.0.1 (localhost)
- **Port**: 9000 (VRChat default)
- **Messages**: `/chatbox/input` for chat

### Window Detection
- Automatically finds VRChat game window
- Filters out development tools (VS Code, File Explorer, etc.)
- Captures only game content (1294x703 resolution typical)

### Message Frequency
- Sends messages every 4-5 cycles OR randomly (15% chance)
- Messages are contextual to current surroundings
- Natural conversation flow

## 📊 Usage Examples

### Example 1: Social Interaction
```
[VISION] several people standing in a lobby area
[THINKING] approach the group and introduce myself
[MESSAGE] Hey everyone! Mind if I join the conversation?
[CHAT] ✅ SENT TO VRCHAT
```

### Example 2: Environmental Awareness
```
[VISION] a beautiful garden with flowers and fountains
[THINKING] this looks like a peaceful place to relax
[MESSAGE] This garden is so beautiful! I love the flowers.
[CHAT] ✅ SENT TO VRCHAT
```

### Example 3: Curiosity
```
[VISION] someone showing off a custom avatar
[THINKING] ask about their avatar creation process
[MESSAGE] That avatar looks amazing! How did you make it?
[CHAT] ✅ SENT TO VRCHAT
```

## 📁 Project Structure
```
VRChat_local_AI_VLM_PAL/
├── main.py                 # Main AI system
├── requirements.txt        # Python dependencies
├── surroundings.log        # AI activity log
├── README.md              # This documentation
├── vrchat_old_project_example/  # Original project reference
└── venv/                  # Virtual environment
```

## 🔍 Troubleshooting

### Common Issues

**"VRChat window not found"**
- Ensure VRChat is running and visible
- Check if VRChat is minimized (bring to foreground)

**"Model loading failed"**
- Verify AMD GPU drivers are up to date
- Check available VRAM (need ~1.5GB free)

**"OSC connection failed"**
- Ensure VRChat OSC is enabled in settings
- Verify port 9000 is not blocked

### Performance Tips
- Close other GPU-intensive applications
- Lower screen resolution if experiencing lag
- Models load faster on SSD storage

## 🚧 Future Enhancements

### Planned Features
- **Movement Actions**: VRChat locomotion controls
- **Voice Integration**: Speech-to-text responses
- **Personality Tuning**: Customizable AI behavior
- **Multi-Language Support**: Non-English conversations
- **Advanced Vision**: Object/person recognition

### Potential Improvements
- **Model Optimization**: Quantized models for lower VRAM
- **Real-time Processing**: Reduced latency
- **Custom Training**: Fine-tuned for VRChat scenarios

## 📄 License

This project is open source. See LICENSE file for details.

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## 📞 Support

For issues or questions:
- Open a GitHub issue
- Check the troubleshooting section
- Review the surroundings.log for debugging

---

**Built with ❤️ for the VRChat community**
