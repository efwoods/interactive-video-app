# interactive-video-app

This application records audio and video and allows you to query an LLM to ask about the content of what you just heard within the context of the past 30 seconds. This turns any video into a 'closed-loop' interactive experience rather than a standalone 'open-loop' offer.

## [Getting Started](https://chatgpt.com/share/6825e5f5-66dc-8011-89c2-11910e9602e5)

Below is a complete proof-of-concept Python script that lets you:

    Continuously capture your system audio and keep a rolling buffer of the last 30 s.

    Listen for a single hotkey (Ctrl+Shift+Q) to:

        Pause whatever video player has focus (simulates a spacebar).

        Grab the last 30 s of audio, transcribe it with a local Whisper model.

        Launch a simple REPL so you can ask a local LLM (via llama_cpp) questions about what you just heard.

        Resume the video (simulates spacebar again).

All models run locally, no API keys needed. You control everything from the keyboard.

#!/usr/bin/env python3
"""
interactive_video_llm.py

Proof-of-Concept: Turn any video into an interactive, conversational experience.
Press Ctrl+Shift+Q to pause, transcribe, ask, and resume.
"""

import threading
import queue
import time
import sounddevice as sd
import numpy as np
import keyboard # pip install keyboard
from faster_whisper import WhisperModel # pip install faster-whisper
from llama_cpp import Llama # pip install llama-cpp-python
from pynput.keyboard import Controller as KbController, Key

# ——— CONFIG ——————————————————————————————————————————————————————————————————

AUDIO_RATE = 16_000 # sample rate for Whisper
BUFFER_SECONDS = 30 # how many seconds to keep in rolling audio buffer
HOTKEY = "ctrl+shift+q" # key combo to trigger pause/transcribe/ask
WHISPER_SIZE = "small" # whisper model size: tiny, base, small, medium, large
LLAMA_PATH = "/path/to/model.bin" # download a ggml quantized LLaMA-2 or similar

# —————————————————————————————————————————————————————————————————————————

# Rolling audio buffer

audio_buffer = queue.deque(maxlen=BUFFER_SECONDS \* AUDIO_RATE)

# Initialize Whisper once

whisper = WhisperModel(WHISPER_SIZE, device="cpu")

# Initialize LLM once

llm = Llama(model_path=LLAMA_PATH)

# Keyboard controller for simulating spacebar

kb = KbController()

def audio_callback(indata, frames, time_info, status):
"""Collect incoming audio into our buffer."""
if status:
print("Audio status:", status)
audio_buffer.extend(indata[:, 0].tolist())

def start_audio_stream():
"""Begin capturing system audio (loopback)."""
sd.default.samplerate = AUDIO_RATE
sd.default.channels = 1 # On Windows, you may need to set `device` to the loopback device.
stream = sd.InputStream(callback=audio_callback)
stream.start()
return stream

def transcribe*buffer() -> str:
"""Dump the current buffer to a NumPy array and transcribe with Whisper."""
data = np.array(audio_buffer, dtype=np.float32) # Whisper expects batch shape (n_samples,)
segments, * = whisper.transcribe(
audio=data,
beam_size=5,
vad_filter=True,
suppress_blank=True
)
transcript = "\n".join([seg.text for seg in segments])
return transcript

def pause_video():
"""Simulate spacebar to pause the video player."""
kb.press(Key.space)
kb.release(Key.space)

def resume_video():
"""Simulate spacebar to resume the video player."""
kb.press(Key.space)
kb.release(Key.space)

def interactive_session(transcript: str):
"""Launch a REPL: show transcript, then let user type questions."""
print("\n──── Transcript ─────────────────────────────────────────")
print(transcript or "[No speech detected]")
print("─────────────────────────────────────────────────────────")
print("You can now ask questions. Empty line to finish and resume video.\n")

    conversation = [{"role": "system", "content": "You are a helpful assistant."}]
    conversation.append({"role": "user", "content": transcript})

    while True:
        user_q = input("You> ").strip()
        if not user_q:
            break
        conversation.append({"role": "user", "content": user_q})
        # Call local LLM
        resp = llm.chat(completion={"messages": conversation, "max_tokens": 256})
        answer = resp["choices"][0]["message"]["content"]
        print("\nLLM>", answer, "\n")
        conversation.append({"role": "assistant", "content": answer})

def on_hotkey():
"""When hotkey is pressed: pause, transcribe, chat, resume."""
print("\n[Hotkey pressed] Pausing video…")
pause_video()
print("Transcribing last {} seconds…".format(BUFFER_SECONDS))
text = transcribe_buffer()
interactive_session(text)
print("[Resuming video]\n")
resume_video()

def main():
print("Starting audio capture… Press {} to interact.".format(HOTKEY))
stream = start_audio_stream() # Register hotkey
keyboard.add_hotkey(HOTKEY, on_hotkey)
try: # Keep main thread alive
while True:
time.sleep(1)
except KeyboardInterrupt:
print("Exiting.")
finally:
stream.stop()
stream.close()

if **name** == "**main**":
main()

Setup & Usage

    Install dependencies

pip install sounddevice numpy keyboard pynput faster-whisper llama-cpp-python

Download models

    Whisper quantized weights: e.g. via HuggingFace openai/whisper.

    LLaMA-2 or GPT-4-All quantized .bin for llama_cpp.

Adjust loopback device

    On Windows, set sd.default.device to your audio loopback.

    On Linux/macOS, use pavucontrol or BlackHole/Soundflower.

Run

    python interactive_video_llm.py

    Workflow

        Start any video in your favorite player.

        Hit Ctrl+Shift+Q to pause & transcribe.

        Read the transcript, then type questions about what you last heard.

        Hit Enter on a blank line to finish and resume playback.

This gives you a fully local, keyboard-driven pipeline turning any video or podcast into an interactive conversational agent.
