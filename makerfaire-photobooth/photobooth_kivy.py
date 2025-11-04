#!/usr/bin/env python3
"""Kivy-based Photobooth UI that captures an image, transcribes speech, and uses FAL AI nano-banana model for image editing."""

import argparse
import base64
import os
import sys
import time
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional
from io import BytesIO

import cv2
import numpy as np
import speech_recognition as sr
import fal_client
from PIL import Image
import requests
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail, Attachment, FileContent, FileName, FileType, Disposition

from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.uix.image import Image as KivyImage
from kivy.uix.popup import Popup
from kivy.uix.gridlayout import GridLayout
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.core.window import Window
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.scrollview import ScrollView
from kivy.metrics import dp
from kivy.animation import Animation

PRINTER_URL = "http://127.0.0.1:8095"
PRINTER_DEVICE_ID = "X5,6B:71:62:47:E4:05"


class CameraScreen(Screen):
    """Main screen showing camera feed and capture controls."""
    
    def __init__(self, photobooth_app, **kwargs):
        super().__init__(**kwargs)
        self.photobooth_app = photobooth_app
        self.camera_texture = None
        self.captured_frame = None
        
        # Main layout
        layout = FloatLayout()
        
        # Camera display
        self.camera_image = KivyImage()
        self.camera_image.size_hint = (0.8, 0.6)
        self.camera_image.pos_hint = {'center_x': 0.5, 'center_y': 0.6}
        layout.add_widget(self.camera_image)
        
        # Title
        title = Label(
            text='MakerFaire Photobooth',
            size_hint=(0.8, 0.1),
            pos_hint={'center_x': 0.5, 'top': 0.95},
            font_size=dp(24),
            bold=True,
            color=(1, 1, 1, 1)
        )
        layout.add_widget(title)
        
        # Instructions
        instructions = Label(
            text='Tap CAPTURE to take a photo, then choose how to describe it',
            size_hint=(0.8, 0.08),
            pos_hint={'center_x': 0.5, 'top': 0.35},
            font_size=dp(16),
            color=(1, 1, 1, 1)
        )
        layout.add_widget(instructions)
        
        # Capture button
        self.capture_btn = Button(
            text='📸 CAPTURE',
            size_hint=(0.3, 0.12),
            pos_hint={'center_x': 0.5, 'top': 0.25},
            font_size=dp(20),
            bold=True,
            background_color=(0.2, 0.7, 0.2, 1)
        )
        self.capture_btn.bind(on_press=self.on_capture)
        layout.add_widget(self.capture_btn)
        
        # Status label
        self.status_label = Label(
            text='Ready to capture',
            size_hint=(0.8, 0.05),
            pos_hint={'center_x': 0.5, 'top': 0.1},
            font_size=dp(14),
            color=(1, 1, 1, 1)
        )
        layout.add_widget(self.status_label)
        
        self.add_widget(layout)
        
        # Start camera update
        Clock.schedule_interval(self.update_camera, 1.0/30.0)  # 30 FPS
    
    def update_camera(self, dt):
        """Update camera display."""
        if self.photobooth_app.capture and self.photobooth_app.capture.isOpened():
            grabbed, frame = self.photobooth_app.capture.read()
            if grabbed:
                # Convert frame to texture
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_rgb = cv2.flip(frame_rgb, 1)  # Mirror the image
                
                # Resize for display
                height, width = frame_rgb.shape[:2]
                max_width = int(Window.width * 0.8)
                max_height = int(Window.height * 0.6)
                
                if width > max_width or height > max_height:
                    scale = min(max_width/width, max_height/height)
                    new_width = int(width * scale)
                    new_height = int(height * scale)
                    frame_rgb = cv2.resize(frame_rgb, (new_width, new_height))
                
                # Create texture
                texture = Texture.create(size=(frame_rgb.shape[1], frame_rgb.shape[0]), colorfmt='rgb')
                texture.blit_buffer(frame_rgb.tobytes(), colorfmt='rgb', bufferfmt='ubyte')
                texture.flip_vertical()
                
                self.camera_image.texture = texture
                self.captured_frame = frame.copy()
    
    def on_capture(self, instance):
        """Handle capture button press."""
        if self.captured_frame is not None:
            self.status_label.text = "Photo captured! Choose input method..."
            self.photobooth_app.show_input_methods(self.captured_frame)
        else:
            self.status_label.text = "Camera not ready. Please wait..."


class InputMethodScreen(Screen):
    """Screen for choosing input method after capture."""
    
    def __init__(self, photobooth_app, **kwargs):
        super().__init__(**kwargs)
        self.photobooth_app = photobooth_app
        
        layout = BoxLayout(orientation='vertical', padding=dp(20), spacing=dp(20))
        
        # Title
        title = Label(
            text='How would you like to describe your photo?',
            size_hint_y=0.2,
            font_size=dp(24),
            bold=True,
            color=(0.2, 0.2, 0.2, 1)
        )
        layout.add_widget(title)
        
        # Captured image preview
        self.preview_image = KivyImage(size_hint_y=0.4)
        layout.add_widget(self.preview_image)
        
        # Button container
        button_layout = BoxLayout(orientation='horizontal', size_hint_y=0.2, spacing=dp(20))
        
        # Speech button
        speech_btn = Button(
            text='🎤 SPEECH',
            font_size=dp(20),
            bold=True,
            background_color=(0.2, 0.5, 0.8, 1)
        )
        speech_btn.bind(on_press=self.on_speech)
        button_layout.add_widget(speech_btn)
        
        # Text button
        text_btn = Button(
            text='⌨️ TYPE',
            font_size=dp(20),
            bold=True,
            background_color=(0.8, 0.5, 0.2, 1)
        )
        text_btn.bind(on_press=self.on_text)
        button_layout.add_widget(text_btn)
        
        layout.add_widget(button_layout)
        
        # Back button
        back_btn = Button(
            text='← BACK TO CAMERA',
            size_hint_y=0.1,
            font_size=dp(16),
            background_color=(0.6, 0.6, 0.6, 1)
        )
        back_btn.bind(on_press=self.on_back)
        layout.add_widget(back_btn)
        
        self.add_widget(layout)
    
    def show_captured_image(self, frame):
        """Display the captured image."""
        # Convert frame to texture
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb = cv2.flip(frame_rgb, 1)  # Mirror the image
        
        # Resize for preview
        height, width = frame_rgb.shape[:2]
        max_size = 400
        if width > max_size or height > max_size:
            scale = min(max_size/width, max_size/height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            frame_rgb = cv2.resize(frame_rgb, (new_width, new_height))
        
        texture = Texture.create(size=(frame_rgb.shape[1], frame_rgb.shape[0]), colorfmt='rgb')
        texture.blit_buffer(frame_rgb.tobytes(), colorfmt='rgb', bufferfmt='ubyte')
        texture.flip_vertical()
        
        self.preview_image.texture = texture
    
    def on_speech(self, instance):
        """Handle speech input."""
        self.photobooth_app.capture_speech()
    
    def on_text(self, instance):
        """Handle text input."""
        self.photobooth_app.show_text_input()
    
    def on_back(self, instance):
        """Go back to camera screen."""
        self.photobooth_app.screen_manager.current = 'camera'


class TextInputScreen(Screen):
    """Screen for text input."""
    
    def __init__(self, photobooth_app, **kwargs):
        super().__init__(**kwargs)
        self.photobooth_app = photobooth_app
        
        layout = BoxLayout(orientation='vertical', padding=dp(20), spacing=dp(20))
        
        # Title
        title = Label(
            text='Describe your photo',
            size_hint_y=0.15,
            font_size=dp(24),
            bold=True,
            color=(0.2, 0.2, 0.2, 1)
        )
        layout.add_widget(title)
        
        # Transcription display
        self.transcription_label = Label(
            text='',
            size_hint_y=0.15,
            font_size=dp(16),
            color=(0.2, 0.6, 0.2, 1),
            text_size=(None, None),
            halign='center'
        )
        layout.add_widget(self.transcription_label)
        
        # Text input
        self.text_input = TextInput(
            text='',
            size_hint_y=0.3,
            multiline=True,
            font_size=dp(18),
            hint_text='Type your description here...'
        )
        layout.add_widget(self.text_input)
        
        # Button container
        button_layout = BoxLayout(orientation='horizontal', size_hint_y=0.15, spacing=dp(20))
        
        # Submit button
        submit_btn = Button(
            text='GENERATE',
            font_size=dp(20),
            bold=True,
            background_color=(0.2, 0.7, 0.2, 1)
        )
        submit_btn.bind(on_press=self.on_submit)
        button_layout.add_widget(submit_btn)
        
        # Back button
        back_btn = Button(
            text='BACK',
            font_size=dp(20),
            background_color=(0.6, 0.6, 0.6, 1)
        )
        back_btn.bind(on_press=self.on_back)
        button_layout.add_widget(back_btn)
        
        layout.add_widget(button_layout)
        
        self.add_widget(layout)
    
    def on_submit(self, instance):
        """Submit text input."""
        text = self.text_input.text.strip()
        if text:
            self.photobooth_app.process_input(text)
        else:
            self.photobooth_app.show_error("Please enter a description")
    
    def on_back(self, instance):
        """Go back to input method screen."""
        self.photobooth_app.screen_manager.current = 'input_method'


class EmailInputScreen(Screen):
    """Screen for email input."""
    
    def __init__(self, photobooth_app, **kwargs):
        super().__init__(**kwargs)
        self.photobooth_app = photobooth_app
        
        layout = BoxLayout(orientation='vertical', padding=dp(20), spacing=dp(20))
        
        # Title
        title = Label(
            text='Enter your email address',
            size_hint_y=0.15,
            font_size=dp(24),
            bold=True,
            color=(0.2, 0.2, 0.2, 1)
        )
        layout.add_widget(title)
        
        # Instructions
        instructions = Label(
            text='We\'ll send your photo to this email address',
            size_hint_y=0.1,
            font_size=dp(16),
            color=(0.4, 0.4, 0.4, 1)
        )
        layout.add_widget(instructions)
        
        # Email input
        self.email_input = TextInput(
            text='',
            size_hint_y=0.2,
            multiline=False,
            font_size=dp(20),
            hint_text='your.email@example.com',
            input_type='text'
        )
        layout.add_widget(self.email_input)
        
        # Button container
        button_layout = BoxLayout(orientation='horizontal', size_hint_y=0.15, spacing=dp(20))
        
        # Send button
        send_btn = Button(
            text='📧 SEND EMAIL',
            font_size=dp(20),
            bold=True,
            background_color=(0.2, 0.7, 0.2, 1)
        )
        send_btn.bind(on_press=self.on_send)
        button_layout.add_widget(send_btn)
        
        # Back button
        back_btn = Button(
            text='BACK',
            font_size=dp(20),
            background_color=(0.6, 0.6, 0.6, 1)
        )
        back_btn.bind(on_press=self.on_back)
        button_layout.add_widget(back_btn)
        
        layout.add_widget(button_layout)
        
        self.add_widget(layout)
    
    def on_send(self, instance):
        """Send email with photo."""
        email = self.email_input.text.strip()
        if email:
            # Basic email validation
            if '@' in email and '.' in email.split('@')[1]:
                self.photobooth_app.send_email(email)
            else:
                self.photobooth_app.show_error("Please enter a valid email address")
        else:
            self.photobooth_app.show_error("Please enter an email address")
    
    def on_back(self, instance):
        """Go back to result screen."""
        self.email_input.text = ''
        self.photobooth_app.screen_manager.current = 'result'


class ResultScreen(Screen):
    """Screen for displaying generated results."""
    
    def __init__(self, photobooth_app, **kwargs):
        super().__init__(**kwargs)
        self.photobooth_app = photobooth_app
        
        layout = BoxLayout(orientation='vertical', padding=dp(20), spacing=dp(20))
        
        # Title
        title = Label(
            text='Your Generated Photo',
            size_hint_y=0.08,
            font_size=dp(24),
            bold=True,
            color=(0.2, 0.2, 0.2, 1)
        )
        layout.add_widget(title)
        
        # Saved path label
        self.saved_path_label = Label(
            text='',
            size_hint_y=0.05,
            font_size=dp(12),
            color=(0.4, 0.4, 0.4, 1),
            text_size=(None, None),
            halign='center'
        )
        layout.add_widget(self.saved_path_label)
        
        # Result image
        self.result_image = KivyImage(size_hint_y=0.7)
        layout.add_widget(self.result_image)
        
        # Button container
        button_layout = BoxLayout(orientation='horizontal', size_hint_y=0.1, spacing=dp(20))
        
        # New photo button
        new_btn = Button(
            text='📸 NEW PHOTO',
            font_size=dp(18),
            bold=True,
            background_color=(0.2, 0.7, 0.2, 1)
        )
        new_btn.bind(on_press=self.on_new_photo)
        button_layout.add_widget(new_btn)
        
        # Print button (only show if not hidden)
        if not self.photobooth_app.hide_print:
            print_btn = Button(
                text='🖨️ PRINT',
                font_size=dp(18),
                bold=True,
                background_color=(0.2, 0.5, 0.8, 1)
            )
            print_btn.bind(on_press=self.on_print)
            button_layout.add_widget(print_btn)
        
        # Email button
        email_btn = Button(
            text='📧 EMAIL',
            font_size=dp(18),
            bold=True,
            background_color=(0.8, 0.5, 0.2, 1)
        )
        email_btn.bind(on_press=self.on_email)
        button_layout.add_widget(email_btn)
        
        # Quit button
        quit_btn = Button(
            text='QUIT',
            font_size=dp(18),
            background_color=(0.8, 0.2, 0.2, 1)
        )
        quit_btn.bind(on_press=self.on_quit)
        button_layout.add_widget(quit_btn)
        
        layout.add_widget(button_layout)
        
        self.add_widget(layout)
    
    def show_result(self, image, saved_path=None):
        """Display the generated result."""
        # Convert image to texture
        if len(image.shape) == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
        
        # Resize for display
        height, width = image_rgb.shape[:2]
        max_width = int(Window.width * 0.9)
        max_height = int(Window.height * 0.6)
        
        if width > max_width or height > max_height:
            scale = min(max_width/width, max_height/height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            image_rgb = cv2.resize(image_rgb, (new_width, new_height))
        
        texture = Texture.create(size=(image_rgb.shape[1], image_rgb.shape[0]), colorfmt='rgb')
        texture.blit_buffer(image_rgb.tobytes(), colorfmt='rgb', bufferfmt='ubyte')
        texture.flip_vertical()
        
        self.result_image.texture = texture
        
        # Show saved path
        if saved_path:
            filename = Path(saved_path).name
            self.saved_path_label.text = f"Saved as: {filename}"
    
    def on_new_photo(self, instance):
        """Start a new photo session."""
        # Clear text input and transcription
        if hasattr(self.photobooth_app.text_input_screen, 'text_input'):
            self.photobooth_app.text_input_screen.text_input.text = ''
        if hasattr(self.photobooth_app.text_input_screen, 'transcription_label'):
            self.photobooth_app.text_input_screen.transcription_label.text = ''
        # Reload photos and go to start screen
        self.photobooth_app.start_screen.load_recent_photos()
        self.photobooth_app.screen_manager.current = 'start'
    
    def on_print(self, instance):
        """Print the current image."""
        self.photobooth_app.print_current_image()
    
    def on_email(self, instance):
        """Send email with the current image."""
        self.photobooth_app.show_email_input()
    
    def on_quit(self, instance):
        """Quit the application."""
        App.get_running_app().stop()


class StartScreen(Screen):
    """Start screen showing recent photos in a grid."""
    
    def __init__(self, photobooth_app, **kwargs):
        super().__init__(**kwargs)
        self.photobooth_app = photobooth_app
        
        layout = BoxLayout(orientation='vertical', padding=dp(20), spacing=dp(20))
        
        # Title
        title = Label(
            text='Accelerate Orlando 2025 Photobooth',
            size_hint_y=0.1,
            font_size=dp(24),
            bold=True,
            color=(0.2, 0.2, 0.2, 1)
        )
        layout.add_widget(title)
        
        # Recent photos grid
        self.photos_grid = GridLayout(
            cols=3,
            size_hint_y=0.7,
            spacing=dp(2),
            padding=dp(5)
        )
        layout.add_widget(self.photos_grid)
        
        # Button container
        button_layout = BoxLayout(
            orientation='horizontal',
            size_hint_y=0.1,
            spacing=dp(20)
        )
        
        # Start new photo button
        start_btn = Button(
            text='📸 START NEW PHOTO',
            font_size=dp(20),
            bold=True,
            background_color=(0.2, 0.7, 0.2, 1)
        )
        start_btn.bind(on_press=self.on_start_new)
        button_layout.add_widget(start_btn)
        
        # Refresh button
        refresh_btn = Button(
            text='🔄 REFRESH',
            font_size=dp(18),
            background_color=(0.2, 0.5, 0.8, 1)
        )
        refresh_btn.bind(on_press=self.on_refresh)
        button_layout.add_widget(refresh_btn)
        
        layout.add_widget(button_layout)
        
        self.add_widget(layout)
        
        # Load recent photos
        self.load_recent_photos()
    
    def load_recent_photos(self):
        """Load and display the 9 most recent photos."""
        # Clear existing photos
        self.photos_grid.clear_widgets()
        
        # Get photos directory
        script_dir = Path(__file__).parent
        photos_dir = script_dir / "photos"
        
        if not photos_dir.exists():
            # Show placeholder if no photos directory
            placeholder = Label(
                text='No photos yet!\nTake your first photo!',
                font_size=dp(16),
                color=(0.6, 0.6, 0.6, 1)
            )
            self.photos_grid.add_widget(placeholder)
            return
        
        # Get all image files and sort by modification time
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            image_files.extend(photos_dir.glob(ext))
        
        # Sort by modification time (newest first) and take first 9
        recent_files = sorted(image_files, key=lambda x: x.stat().st_mtime, reverse=True)[:9]
        
        # Add photo widgets to grid
        for i, photo_path in enumerate(recent_files):
            try:
                # Load and resize image for thumbnail
                image = cv2.imread(str(photo_path))
                if image is not None:
                    # Resize to thumbnail size
                    thumbnail = cv2.resize(image, (525, 300))
                    thumbnail_rgb = cv2.cvtColor(thumbnail, cv2.COLOR_BGR2RGB)
                    
                    # Create texture
                    texture = Texture.create(size=(525, 300), colorfmt='rgb')
                    texture.blit_buffer(thumbnail_rgb.tobytes(), colorfmt='rgb', bufferfmt='ubyte')
                    texture.flip_vertical()
                    
                    # Create image widget
                    photo_widget = KivyImage(texture=texture)
                    photo_widget.bind(on_touch_down=lambda w, touch: self.on_photo_click(photo_path, touch))
                    
                    self.photos_grid.add_widget(photo_widget)
                else:
                    # Add placeholder for corrupted images
                    placeholder = Label(
                        text='📷',
                        font_size=dp(24),
                        color=(0.6, 0.6, 0.6, 1)
                    )
                    self.photos_grid.add_widget(placeholder)
            except Exception as e:
                print(f"Error loading photo {photo_path}: {e}")
                # Add placeholder for error
                placeholder = Label(
                    text='❌',
                    font_size=dp(24),
                    color=(0.8, 0.2, 0.2, 1)
                )
                self.photos_grid.add_widget(placeholder)
        
        # Fill remaining slots with empty placeholders
        while len(self.photos_grid.children) < 9:
            placeholder = Label(
                text='',
                font_size=dp(16),
                color=(0.9, 0.9, 0.9, 1)
            )
            self.photos_grid.add_widget(placeholder)
    
    def on_photo_click(self, photo_path, touch):
        """Handle photo click to view full size."""
        if touch.is_double_tap:
            # Show full-size photo in a popup
            self.show_full_photo(photo_path)
    
    def show_full_photo(self, photo_path):
        """Show full-size photo in a popup."""
        try:
            # Load full image
            image = cv2.imread(str(photo_path))
            if image is None:
                return
            
            # Convert to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Resize for display (max 600px width)
            height, width = image_rgb.shape[:2]
            max_width = 600
            if width > max_width:
                scale = max_width / width
                new_width = int(width * scale)
                new_height = int(height * scale)
                image_rgb = cv2.resize(image_rgb, (new_width, new_height))
            
            # Create texture
            texture = Texture.create(size=(image_rgb.shape[1], image_rgb.shape[0]), colorfmt='rgb')
            texture.blit_buffer(image_rgb.tobytes(), colorfmt='rgb', bufferfmt='ubyte')
            texture.flip_vertical()
            
            # Create popup content
            content = BoxLayout(orientation='vertical', spacing=dp(10))
            
            # Image widget
            image_widget = KivyImage(texture=texture, size_hint_y=0.8)
            content.add_widget(image_widget)
            
            # Close button
            close_btn = Button(
                text='Close',
                size_hint_y=0.2,
                font_size=dp(16)
            )
            content.add_widget(close_btn)
            
            # Create and show popup
            popup = Popup(
                title=f'Photo: {photo_path.name}',
                content=content,
                size_hint=(0.8, 0.8)
            )
            close_btn.bind(on_press=popup.dismiss)
            popup.open()
            
        except Exception as e:
            print(f"Error showing full photo {photo_path}: {e}")
    
    def on_start_new(self, instance):
        """Start a new photo session."""
        # Clear text input and transcription
        if hasattr(self.photobooth_app.text_input_screen, 'text_input'):
            self.photobooth_app.text_input_screen.text_input.text = ''
        if hasattr(self.photobooth_app.text_input_screen, 'transcription_label'):
            self.photobooth_app.text_input_screen.transcription_label.text = ''
        self.photobooth_app.screen_manager.current = 'camera'
    
    def on_refresh(self, instance):
        """Refresh the photo grid."""
        self.load_recent_photos()


class LoadingScreen(Screen):
    """Screen for showing loading state during image generation."""
    
    def __init__(self, photobooth_app, **kwargs):
        super().__init__(**kwargs)
        self.photobooth_app = photobooth_app
        
        layout = BoxLayout(orientation='vertical', padding=dp(20), spacing=dp(20))
        
        # Title
        title = Label(
            text='Generating Your Photo',
            size_hint_y=0.2,
            font_size=dp(24),
            bold=True,
            color=(0.2, 0.2, 0.2, 1)
        )
        layout.add_widget(title)
        
        # Loading spinner container
        spinner_container = BoxLayout(
            orientation='vertical',
            size_hint_y=0.4,
            spacing=dp(20)
        )
        
        # Loading spinner (animated dots)
        self.spinner_label = Label(
            text='⏳',
            font_size=dp(48),
            size_hint_y=0.6
        )
        spinner_container.add_widget(self.spinner_label)
        
        # Status text
        self.status_label = Label(
            text='Processing with AI...',
            font_size=dp(18),
            size_hint_y=0.4,
            color=(0.4, 0.4, 0.4, 1)
        )
        spinner_container.add_widget(self.status_label)
        
        layout.add_widget(spinner_container)
        
        # Progress indicator
        self.progress_label = Label(
            text='This may take a few moments',
            font_size=dp(14),
            size_hint_y=0.1,
            color=(0.6, 0.6, 0.6, 1)
        )
        layout.add_widget(self.progress_label)
        
        # Cancel button
        cancel_btn = Button(
            text='CANCEL',
            size_hint_y=0.1,
            font_size=dp(16),
            background_color=(0.8, 0.2, 0.2, 1)
        )
        cancel_btn.bind(on_press=self.on_cancel)
        layout.add_widget(cancel_btn)
        
        self.add_widget(layout)
        
        # Start spinner animation
        self.start_spinner_animation()
    
    def start_spinner_animation(self):
        """Start the spinner animation."""
        def animate_spinner(dt):
            # Rotate the emoji to create a simple spinner effect
            current_text = self.spinner_label.text
            if current_text == '⏳':
                self.spinner_label.text = '⏰'
            elif current_text == '⏰':
                self.spinner_label.text = '⏱️'
            elif current_text == '⏱️':
                self.spinner_label.text = '⏲️'
            else:
                self.spinner_label.text = '⏳'
        
        # Schedule the animation to run every 0.5 seconds
        self.animation_event = Clock.schedule_interval(animate_spinner, 0.5)
    
    def update_status(self, message):
        """Update the status message."""
        self.status_label.text = message
    
    def on_cancel(self, instance):
        """Handle cancel button press."""
        # Stop the animation
        if hasattr(self, 'animation_event'):
            self.animation_event.cancel()
        
        # Clear text input and go back to start screen
        if hasattr(self.photobooth_app.text_input_screen, 'text_input'):
            self.photobooth_app.text_input_screen.text_input.text = ''
        if hasattr(self.photobooth_app.text_input_screen, 'transcription_label'):
            self.photobooth_app.text_input_screen.transcription_label.text = ''
        # Reload photos and go to start screen
        self.photobooth_app.start_screen.load_recent_photos()
        self.photobooth_app.screen_manager.current = 'start'


class PhotoboothKivyApp(App):
    """Main Kivy application."""
    
    def __init__(self, camera_index, api_key, speech_timeout, phrase_time_limit, hide_print=False):
        super().__init__()
        self.camera_index = camera_index
        self.api_key = api_key
        self.speech_timeout = speech_timeout
        self.phrase_time_limit = phrase_time_limit
        self.hide_print = hide_print
        
        # Initialize camera
        self.capture = cv2.VideoCapture(camera_index)
        if not self.capture.isOpened():
            raise RuntimeError(f"Unable to open camera index {camera_index}")
        
        # FAL AI uses environment variable for API key, but we can set it if provided
        if api_key:
            os.environ["FAL_KEY"] = api_key
        
        # Initialize SendGrid client
        sendgrid_api_key = os.getenv("SENDGRID_API_KEY")
        if not sendgrid_api_key:
            sendgrid_api_key = _load_sendgrid_api_key_from_env_files()
        self.sendgrid_client = SendGridAPIClient(api_key=sendgrid_api_key) if sendgrid_api_key else None
        
        # Current captured frame
        self.current_frame = None
        self.current_saved_path = None
        
        # Screen manager
        self.screen_manager = ScreenManager()
        
        # Create screens with names
        self.start_screen = StartScreen(self, name='start')
        self.camera_screen = CameraScreen(self, name='camera')
        self.input_method_screen = InputMethodScreen(self, name='input_method')
        self.text_input_screen = TextInputScreen(self, name='text_input')
        self.loading_screen = LoadingScreen(self, name='loading')
        self.result_screen = ResultScreen(self, name='result')
        self.email_input_screen = EmailInputScreen(self, name='email_input')
        
        # Add screens to manager
        self.screen_manager.add_widget(self.start_screen)
        self.screen_manager.add_widget(self.camera_screen)
        self.screen_manager.add_widget(self.input_method_screen)
        self.screen_manager.add_widget(self.text_input_screen)
        self.screen_manager.add_widget(self.loading_screen)
        self.screen_manager.add_widget(self.result_screen)
        self.screen_manager.add_widget(self.email_input_screen)
        
        # Set initial screen
        self.screen_manager.current = 'start'
        
        # Setup printer on startup
        self.setup_printer()
    
    def build(self):
        """Build the Kivy app."""
        # Set window properties
        Window.clearcolor = (0.1, 0.1, 0.1, 1)  # Dark background
        
        # Set fullscreen
        Window.fullscreen = 'auto'
        
        return self.screen_manager
    
    def show_input_methods(self, frame):
        """Show input method selection screen."""
        self.current_frame = frame
        self.input_method_screen.show_captured_image(frame)
        self.screen_manager.current = 'input_method'
    
    def capture_speech(self):
        """Capture speech input in a separate thread."""
        def speech_thread():
            recognizer = sr.Recognizer()
            try:
                with sr.Microphone(device_index=2, sample_rate=16000, chunk_size=1024) as source:
                    recognizer.adjust_for_ambient_noise(source, duration=0.5)
                    Clock.schedule_once(lambda dt: self.show_status("Listening..."), 0)
                    audio = recognizer.listen(
                        source,
                        timeout=self.speech_timeout,
                        phrase_time_limit=self.phrase_time_limit,
                    )
            except sr.WaitTimeoutError:
                Clock.schedule_once(lambda dt: self.show_error("No speech detected"), 0)
                return
            except OSError as exc:
                Clock.schedule_once(lambda dt: self.show_error("Microphone not available"), 0)
                return

            try:
                transcript = recognizer.recognize_google(audio)
                Clock.schedule_once(lambda dt: self.process_input(transcript), 0)
            except sr.UnknownValueError:
                Clock.schedule_once(lambda dt: self.show_error("Could not understand speech"), 0)
            except sr.RequestError as err:
                Clock.schedule_once(lambda dt: self.show_error(f"Speech recognition error: {err}"), 0)
        
        threading.Thread(target=speech_thread, daemon=True).start()
    
    def show_text_input(self):
        """Show text input screen."""
        self.screen_manager.current = 'text_input'
    
    def process_input(self, transcript):
        """Process user input and generate image."""
        if not transcript or self.current_frame is None:
            return
        
        # Show transcription in text input screen
        if hasattr(self.text_input_screen, 'transcription_label'):
            self.text_input_screen.transcription_label.text = f"🎤 Heard: {transcript}"
        
        # Show loading screen
        self.screen_manager.current = 'loading'
        
        def generate_thread():
            try:
                generated = self.call_fal_ai(self.current_frame, transcript)
                Clock.schedule_once(lambda dt: self.show_result(generated), 0)
            except Exception as exc:
                Clock.schedule_once(lambda dt: self.show_error(f"Generation failed: {exc}"), 0)
        
        threading.Thread(target=generate_thread, daemon=True).start()
    
    def call_fal_ai(self, frame, transcript):
        """Call FAL AI nano-banana model to generate image."""
        # Convert OpenCV frame to PIL Image
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)

        prompt = (
            "You are an event photobooth artist. Use the visitor photo as reference and produce a fun, "
            "stylized fake photo inspired by what they said. The generated image should be suitable for "
            f"display on screen.  Label it as from the Accelerate Orlando 2025 Photobooth, with today's date, which is {datetime.now().strftime('%Y-%m-%d')}"
            " Transcript from the visitor:\n"
            f"{transcript}"
        )

        try:
            # Encode image as data URI
            image_data_uri = fal_client.encode_image(pil_image, format="jpeg")
            
            # Call FAL AI nano-banana model
            response = fal_client.run(
                "fal-ai/nano-banana/edit",
                arguments={
                    "prompt": prompt,
                    "image_urls": [image_data_uri],
                    "num_images": 1,
                    "output_format": "jpeg"
                }
            )
            
            # Process the response to extract the generated image
            if "images" in response and len(response["images"]) > 0:
                # Download the generated image from URL
                image_url = response["images"][0]["url"]
                img_response = requests.get(image_url)
                img_response.raise_for_status()
                
                # Convert the generated image to OpenCV format
                generated_image = Image.open(BytesIO(img_response.content))
                generated_array = np.array(generated_image)
                # Convert RGB to BGR for OpenCV
                result = cv2.cvtColor(generated_array, cv2.COLOR_RGB2BGR)
                return result
            else:
                raise RuntimeError("No image data found in FAL AI response")
            
        except Exception as e:
            raise RuntimeError(f"FAL AI API call failed: {e}")
    
    def save_generated_image(self, image):
        """Save the generated image with timestamp filename in photos directory."""
        # Create timestamp filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"photobooth_generated_{timestamp}.jpg"
        
        # Get the directory where the script is located
        script_dir = Path(__file__).parent
        photos_dir = script_dir / "photos"
        
        # Create photos directory if it doesn't exist
        photos_dir.mkdir(exist_ok=True)
        
        filepath = photos_dir / filename
        
        # Save the image
        cv2.imwrite(str(filepath), image)
        
        print(f"Generated image saved to: {filepath}")
        return str(filepath)
    
    def show_result(self, image):
        """Show the generated result."""
        # Save the image with timestamp
        saved_path = self.save_generated_image(image)
        self.current_saved_path = saved_path
        self.result_screen.show_result(image, saved_path)
        self.screen_manager.current = 'result'
    
    def show_status(self, message):
        """Show status message."""
        if hasattr(self.camera_screen, 'status_label'):
            self.camera_screen.status_label.text = message
    
    def show_error(self, message):
        """Show error message."""
        # Stop loading animation if it's running
        if hasattr(self.loading_screen, 'animation_event'):
            self.loading_screen.animation_event.cancel()
        
        # Create error popup
        content = BoxLayout(orientation='vertical', spacing=dp(10))
        content.add_widget(Label(text=message, text_size=(dp(300), None)))
        
        btn = Button(text='OK', size_hint_y=None, height=dp(40))
        popup = Popup(title='Error', content=content, size_hint=(0.8, 0.4))
        btn.bind(on_press=popup.dismiss)
        content.add_widget(btn)
        
        popup.open()
        
        # Clear text input and transcription, then go back to start screen after error
        if hasattr(self.text_input_screen, 'text_input'):
            self.text_input_screen.text_input.text = ''
        if hasattr(self.text_input_screen, 'transcription_label'):
            self.text_input_screen.transcription_label.text = ''
        # Reload photos and go to start screen
        self.start_screen.load_recent_photos()
        self.screen_manager.current = 'start'
    
    def setup_printer(self):
        """Setup printer by calling /set and /connect endpoints."""
        def setup_thread():
            try:
                # Scan devices
                scan_response = requests.post(f"{PRINTER_URL}/devices", json={"everything": True}, timeout=30)
                scan_response.raise_for_status()
                print(f"Scanner response: {scan_response.status_code}")
                print(f"Scanner response: {scan_response.text}")

                # Call /set endpoint
                set_payload = {"printer": PRINTER_DEVICE_ID}
                set_response = requests.post(f"{PRINTER_URL}/set", json=set_payload, timeout=30)
                set_response.raise_for_status()
                print(f"Printer setup successful: {set_response.status_code}")
                
                # Call /connect endpoint
                connect_payload = {"device": PRINTER_DEVICE_ID}
                connect_response = requests.post(f"{PRINTER_URL}/connect", json=connect_payload, timeout=30)
                connect_response.raise_for_status()
                print(f"Printer connection successful: {connect_response.status_code}")
                
            except requests.exceptions.RequestException as e:
                print(f"Printer setup failed: {e}")
            except Exception as e:
                print(f"Unexpected error during printer setup: {e}")
        
        threading.Thread(target=setup_thread, daemon=True).start()
    
    def print_current_image(self):
        """Print the current saved image."""
        if not self.current_saved_path:
            self.show_error("No image to print")
            return
        
        def print_thread():
            try:
                # Load the image
                image = cv2.imread(self.current_saved_path)
                if image is None:
                    Clock.schedule_once(lambda dt: self.show_error("Could not load image for printing"), 0)
                    return
                
                # Convert to grayscale
                gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                
                # Convert to PBM format
                pbm_data = self.convert_to_pbm(gray_image)
                
                # Send to printer
                print_response = requests.post(
                    f"{PRINTER_URL}/print",
                    data=pbm_data,
                    headers={'Content-Type': 'application/octet-stream'},
                    timeout=30
                )
                print_response.raise_for_status()
                
                Clock.schedule_once(lambda dt: self.show_status("Print job sent successfully!"), 0)
                
            except requests.exceptions.RequestException as e:
                error_msg = f"Print failed: {e}"
                Clock.schedule_once(lambda dt: self.show_error(error_msg), 0)
            except Exception as e:
                error_msg = f"Print error: {e}"
                Clock.schedule_once(lambda dt: self.show_error(error_msg), 0)
        
        threading.Thread(target=print_thread, daemon=True).start()
    
    def show_email_input(self):
        """Show email input screen."""
        self.screen_manager.current = 'email_input'
    
    def send_email(self, recipient_email):
        """Send email with the current saved image."""
        if not self.current_saved_path:
            self.show_error("No image to email")
            return
        
        if not self.sendgrid_client:
            self.show_error("SendGrid API key not configured. Please set SENDGRID_API_KEY environment variable.")
            return
        
        def email_thread():
            try:
                # Load the image
                with open(self.current_saved_path, 'rb') as f:
                    image_data = f.read()
                
                # Read the image to get dimensions for the email
                image = cv2.imread(self.current_saved_path)
                if image is None:
                    Clock.schedule_once(lambda dt: self.show_error("Could not load image for email"), 0)
                    return
                
                # Encode image as base64 for attachment
                encoded_image = base64.b64encode(image_data).decode()
                
                # Create email message
                from_email = "help@maven.ly" #os.getenv("SENDGRID_FROM_EMAIL", "photobooth@makerfaire.com")
                subject = "Your MakerFaire Photobooth Photo"
                
                # Get the image filename
                image_filename = Path(self.current_saved_path).name
                
                # Create HTML email content
                html_content = f"""
                <html>
                <body>
                    <h2>Your Photobooth Photo</h2>
                    <p>Thank you for visiting the Accelerate Orlando 2025 Photobooth!</p>
                    <p>Your stylized photo is attached.</p>
                    <p>Enjoy your memories!</p>
                </body>
                </html>
                """
                
                # Create plain text content
                plain_content = """
                Your Photobooth Photo
                
                Thank you for visiting the Accelerate Orlando 2025 Photobooth!
                Your stylized photo is attached.
                
                Enjoy your memories!
                """
                
                # Create mail message
                message = Mail(
                    from_email=from_email,
                    to_emails=recipient_email,
                    subject=subject,
                    html_content=html_content,
                    plain_text_content=plain_content
                )
                
                # Attach the image
                attachment = Attachment()
                attachment.file_content = FileContent(encoded_image)
                attachment.file_name = FileName(image_filename)
                attachment.file_type = FileType('image/jpeg')
                attachment.disposition = Disposition('attachment')
                message.attachment = attachment
                
                # Send email
                response = self.sendgrid_client.send(message)
                
                # Check if email was sent successfully (status code 202 is success)
                if response.status_code == 202:
                    # Clear email input
                    if hasattr(self.email_input_screen, 'email_input'):
                        self.email_input_screen.email_input.text = ''
                    # Navigate back to result screen and show success message
                    def show_success(dt):
                        self.screen_manager.current = 'result'
                        self.show_status("Email sent successfully!")
                    Clock.schedule_once(show_success, 0)
                else:
                    error_msg = f"Email failed with status code: {response.status_code}"
                    Clock.schedule_once(lambda dt: self.show_error(error_msg), 0)
                
            except Exception as e:
                error_msg = f"Email error: {str(e)}"
                Clock.schedule_once(lambda dt: self.show_error(error_msg), 0)
        
        threading.Thread(target=email_thread, daemon=True).start()
    
    def convert_to_pbm(self, gray_image):
        """Convert grayscale image to PBM format."""
        # Resize image to fit thermal printer dimensions (384x672 pixels)
        rotated_image = cv2.rotate(gray_image, cv2.ROTATE_90_CLOCKWISE)

        target_width = 384
        target_height = 672
        resized = cv2.resize(rotated_image, (target_width, target_height))

        # Convert to binary (threshold at 128)
        _, binary = cv2.threshold(resized, 128, 255, cv2.THRESH_BINARY)
        
        # Convert to PBM format
        pbm_header = f"P4\n{target_width} {target_height}\n".encode('ascii')
        
        # Convert binary image to packed bits
        binary_data = (binary == 0).astype(np.uint8)  # Invert: 0=black, 1=white
        packed_bits = np.packbits(binary_data.flatten())
        
        return pbm_header + packed_bits.tobytes()
    
    def on_stop(self):
        """Clean up when app stops."""
        if self.capture:
            self.capture.release()


def _load_api_key_from_env_files() -> Optional[str]:
    """Load API key from .env files."""
    search_paths = [
        Path(__file__).resolve().parent / ".env",
        Path(__file__).resolve().parent.parent / ".env",
    ]
    for env_path in search_paths:
        if not env_path.exists():
            continue
        try:
            for line in env_path.read_text(encoding="utf-8").splitlines():
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                if "=" not in stripped:
                    continue
                key, value = stripped.split("=", 1)
                if key.strip() == "FAL_KEY":
                    return value.strip().strip('"\'')
        except OSError:
            continue
    return None


def _load_sendgrid_api_key_from_env_files() -> Optional[str]:
    """Load SendGrid API key from .env files."""
    search_paths = [
        Path(__file__).resolve().parent / ".env",
        Path(__file__).resolve().parent.parent / ".env",
    ]
    for env_path in search_paths:
        if not env_path.exists():
            continue
        try:
            for line in env_path.read_text(encoding="utf-8").splitlines():
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                if "=" not in stripped:
                    continue
                key, value = stripped.split("=", 1)
                if key.strip() == "SENDGRID_API_KEY":
                    return value.strip().strip('"\'')
        except OSError:
            continue
    return None


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Kivy-based Photobooth that uses FAL AI nano-banana model.")
    parser.add_argument("--camera", type=int, default=0, help="Camera index to open (default: 0)")
    parser.add_argument(
        "--speech-timeout",
        type=float,
        default=5.0,
        help="Seconds to wait for speech to start (default: 5.0)",
    )
    parser.add_argument(
        "--phrase-time-limit",
        type=float,
        default=8.0,
        help="Maximum speech duration to capture (default: 8.0)",
    )
    return parser.parse_args()


def main() -> None:
    """Main function."""
    args = parse_args()
    api_key = os.getenv("FAL_KEY")
    if not api_key:
        api_key = _load_api_key_from_env_files()
    if not api_key:
        print(
            "Set FAL_KEY as an environment variable or provide it in a .env file.",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        app = PhotoboothKivyApp(
            camera_index=args.camera,
            api_key=api_key,
            speech_timeout=args.speech_timeout,
            phrase_time_limit=args.phrase_time_limit,
            hide_print=True,
        )
        app.run()
    except Exception as exc:
        print(f"Fatal error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
