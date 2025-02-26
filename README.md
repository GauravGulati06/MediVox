---
title: MediVox - AI Doctor with Vision and Voice
emoji: 👨‍⚕️
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.16.0
app_file: app.py
pinned: false
---

# AI Doctor with Vision and Voice

This is an AI-powered medical assistant that can:
- Accept voice input from patients
- Analyze medical images
- Provide medical insights using RAG (Retrieval Augmented Generation)
- Respond with natural voice output

## Features

- Speech-to-Text using Whisper
- Image Analysis using LLaVA
- RAG using FAISS and medical knowledge base
- Text-to-Speech using ElevenLabs
- Context-aware responses using medical domain knowledge

## Environment Variables Required

```bash
GROQ_API_KEY=your_groq_api_key
ELEVENLABS_API_KEY=your_elevenlabs_api_key
```

## Usage

1. Click the microphone button to record your question
2. Upload or take a picture of the medical condition
3. Wait for the AI doctor to analyze and respond
4. Listen to the voice response or read the text output

## Model Details

- Vision Model: LLaVA 3.2 11B
- Speech-to-Text: Whisper Large V3
- Text Generation: Groq
- Voice Generation: ElevenLabs
- Embeddings: sentence-transformers/all-mpnet-base-v2

## Citation

If you use this space, please cite:
```
@misc{medivoicebot2024,
  author = {Your Name},
  title = {AI Doctor with Vision and Voice},
  year = {2024},
  publisher = {Hugging Face Spaces},
}
``` 