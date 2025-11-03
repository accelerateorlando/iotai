# MakerFaire Photobooth - Kivy UI

A touch-friendly photobooth application built with Kivy that captures photos, accepts speech or text input, and uses Google's Gemini AI to generate stylized images.

## Features

- **Touch-friendly interface** perfect for Maker Faire kiosks
- **Live camera preview** with mirrored display
- **Speech recognition** for hands-free input
- **Text input** for typed descriptions
- **AI-powered image generation** using Google Gemini
- **Large, colorful buttons** optimized for touch screens
- **Modern UI** with smooth transitions between screens

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up your Google API key:
```bash
export GOOGLE_API_KEY="your-api-key-here"
```

Or create a `.env` file in the project directory:
```
GOOGLE_API_KEY=your-api-key-here
```

## Usage

### Run the Kivy version (recommended for touch screens):
```bash
python run_kivy.py
```

### Run the original OpenCV version:
```bash
python photobooth.py
```

## Kivy UI Workflow

1. **Camera Screen**: Shows live camera feed with a large "CAPTURE" button
2. **Input Method Screen**: After capture, choose between speech or text input
3. **Text Input Screen**: Type your description (if text input chosen)
4. **Result Screen**: View the AI-generated image with options to take another photo or quit

## Command Line Options

- `--camera N`: Use camera index N (default: 0)
- `--model MODEL`: Specify Gemini model (default: gemini-2.5-flash-image-preview)
- `--speech-timeout SECONDS`: Speech detection timeout (default: 5.0)
- `--phrase-time-limit SECONDS`: Maximum speech duration (default: 8.0)

## Touch Screen Optimization

The Kivy interface is optimized for touch screens with:
- Large buttons (minimum 44px touch targets)
- High contrast colors
- Clear visual feedback
- Intuitive navigation flow
- Error handling with user-friendly messages

## Requirements

- Python 3.7+
- Camera (USB webcam or built-in)
- Microphone (for speech input)
- Google API key for Gemini
- Kivy 2.2.0+
- OpenCV
- SpeechRecognition
- Google GenAI

## Troubleshooting

- **Camera not working**: Try different camera indices with `--camera 1`, `--camera 2`, etc.
- **Microphone issues**: Check microphone permissions and availability
- **API errors**: Verify your Google API key is correct and has Gemini access
- **Touch not working**: Ensure you're running the Kivy version (`run_kivy.py`)

## Original vs Kivy Version

- **Original (`photobooth.py`)**: Keyboard-controlled, OpenCV windows, good for development
- **Kivy (`photobooth_kivy.py`)**: Touch-controlled, modern UI, perfect for public kiosks
