# Plagon Voice Architecture

This module implements the **Live Voice Interface** for Plagon LLM, enabling near real-time verbal interaction.

## 🎤 Pipeline Overview

The voice system operates in a continuous **Audio Loop**:

```mermaid
graph TD
    A[🎤 Microphone Input] -->|Audio Chunks| B(VAD - Voice Activity Detection)
    B -->|Speech Detected| C[📝 STT Engine (Whisper)]
    C -->|Transcribed Text| D[🧠 Plagon LLM (Reasoning)]
    D -->|Response Text| E[🗣️ TTS Engine]
    E -->|Audio Waveform| F[🔊 Speaker Output]
```

## 🧩 Components

1.  **Input Layer (Hearing)**
    *   **VAD (Voice Activity Detection):** Monitors ambient audio. Filters out silence/noise to trigger valid recording only when someone speaks.
    *   **STT (Speech-to-Text):** Converts raw audio into text strings. We use **Whisper** (tiny/base model) for low latency and high accuracy in English/Bangla.

2.  **Cognitive Layer (Thinking)**
    *   **Voice Prompt:** A specialized system prompt that forces Plagon into "Conversational Mode" (brief, direct answers, no markdown).
    *   **Inference:** Uses the standard Plagon Transformer but optimized for speed (potentially shorter context).

3.  **Output Layer (Speaking)**
    *   **TTS (Text-to-Speech):** Converts generated text into audio.
    *   **Audio Sink:** Plays the audio stream to the user.

## ⚡ Latency Strategy

To achieve a "natural" feel, we optimize for **Time-to-First-Audio**:
*   **Streaming STT:** Process audio as soon as a phrase ends? (We will use VAD chunks for simplicity first).
*   **Streaming TTS:** (Advanced) Start speaking the first sentence while the second is still generating.

## 🛡️ Safety
*   **Interrupt Handling:** If the user speaks *while* the AI is talking, the system must pause playback (Echo Cancellation/Barge-in).
*   **Timeout:** Stops listening if no speech for N seconds to save resources.
