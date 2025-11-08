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

2. Set up your FAL AI API key:
```bash
export FAL_KEY="your-api-key-here" SENDGRID_API_KEY="your-api-key-here"
```

Or create a `.env` file in the project directory:
```
FAL_KEY=your-api-key-here
SENDGRID_API_KEY=your-api-key-here
```

## Usage

```bash
python run_kivy.py
```


## Kivy UI Workflow

1. **Camera Screen**: Shows live camera feed with a large "CAPTURE" button
2. **Input Method Screen**: After capture, choose between speech or text input
3. **Text Input Screen**: Type your description (if text input chosen)
4. **Result Screen**: View the AI-generated image with options to take another photo or quit

## Command Line Options

- `--camera N`: Use camera index N (default: 0)
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
- FAL API key for Gemini
- Sendgrid API key if you want the app to send emails
- Kivy 2.2.0+
- OpenCV
- SpeechRecognition

## Troubleshooting

- **Camera not working**: Try different camera indices with `--camera 1`, `--camera 2`, etc.
- **Microphone issues**: Check microphone permissions and availability
- **Touch not working**: Ensure you're running the Kivy version (`run_kivy.py`)
