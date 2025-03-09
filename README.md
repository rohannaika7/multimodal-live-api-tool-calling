Multimodal Live API with Gemini
This repository contains a Python-based WebSocket server and a web-based client application for interacting with Google's Gemini AI models in real-time. The system supports multimodal inputs (text, audio, and images) and outputs (text and audio), leveraging Retrieval-Augmented Generation (RAG) for enhanced responses. It integrates various technologies including WebSockets, Google Generative AI, and Vertex AI embeddings.

Features
Real-Time Interaction: Communicate with the Gemini API via WebSockets for low-latency, two-way interactions.
Multimodal Support: Handles text, audio, and image inputs, with text and audio outputs.
Retrieval-Augmented Generation (RAG): Enhances responses using a precomputed document store and embeddings.
Web Interface: A user-friendly HTML/JavaScript frontend for interacting with the system, including camera, microphone, and screen-sharing capabilities.
Cookie Management: Persists user settings (e.g., API tokens) using a custom CookieJar utility.
Audio Processing: Real-time audio input/output using Web Audio API and PCM processing.
Repository Structure
main.py: The core WebSocket server implementation using Python, integrating with the Gemini API and RAG.
index.html: The web frontend for user interaction, built with Material Web Components.
script.js: Client-side JavaScript logic for WebSocket communication and media handling.
gemini-live-api.js: A JavaScript class for interacting with the Gemini API over WebSockets.
live-media-manager.js: Manages live audio and video input/output.
cookieJar.js: A utility for managing cookies tied to HTML input elements.
pcm_processor.js: An AudioWorklet processor for handling PCM audio data.
style.css: Styling for the web interface.
requirements.txt: Python dependencies for the server.
Prerequisites
Python 3.8+: Required for running the server.
Node.js: Optional, for local development of the frontend (if modifying JavaScript).
Google Cloud Account: For API key and project ID setup.
Web Browser: A modern browser (e.g., Chrome) with WebSocket and Web Audio API support.
Setup
1. Clone the Repository
bash

Collapse

Wrap

Copy
git clone https://github.com/your-username/multimodal-live-api.git
cd multimodal-live-api
2. Install Python Dependencies
Install the required Python packages listed in requirements.txt:

bash

Collapse

Wrap

Copy
pip install -r requirements.txt
Note: Ensure you have the Google Cloud SDK installed and configured if using Vertex AI features.

3. Configure Environment Variables
Set your Google API key as an environment variable:

bash

Collapse

Wrap

Copy
export GOOGLE_API_KEY='YOUR_API_KEY'
Replace 'YOUR_API_KEY' with your actual Google Cloud API key.

4. Update Configuration
In main.py, replace 'YOUR_API_KEY' with your API key (if not using environment variables).
In script.js, update PROJECT_ID with your Google Cloud project ID.
5. Serve the Frontend
You can use a simple HTTP server to serve the frontend files:

bash

Collapse

Wrap

Copy
python -m http.server 8000
Access the interface at http://localhost:8000.

6. Run the WebSocket Server
Start the WebSocket server:

bash

Collapse

Wrap

Copy
python main.py
The server will run on localhost:9011.

Usage
Open the Web Interface: Navigate to http://localhost:8000 in your browser.
Configure Settings:
Enter your Google Cloud project ID and access token in the provided fields.
Set system instructions (optional) to guide the AI's behavior.
Choose the response modality (audio or text).
Connect: Click the "Connect" button to establish a WebSocket connection to the server.
Interact:
Use the microphone, camera, or screen-sharing buttons to send real-time inputs.
Type and send text messages via the text input field.
Receive Responses: The AI will respond with either audio (played through your speakers) or text (displayed in the chat).
Technical Details
Server (main.py)
WebSocket Handling: Uses websockets to manage client connections and forward messages to the Gemini API.
RAG Implementation: Employs langchain_google_vertexai for embeddings and document retrieval.
Models: Configured with gemini-2.0-flash-exp for generation and gemini-1.5-flash-8b for transcription.
Client (index.html, script.js, etc.)
WebSocket Communication: Managed by gemini-live-api.js for real-time interaction with the server.
Media Handling: live-media-manager.js processes audio and video inputs, converting them to base64 for transmission.
UI: Built with Material Web Components for a modern, responsive design.
Audio Processing
Input: Captures microphone audio, converts it to PCM, and sends it as base64-encoded chunks.
Output: Receives PCM audio from the server, processes it with pcm_processor.js, and plays it via the Web Audio API.
Dependencies
See requirements.txt for the full list. Key dependencies include:

websockets: WebSocket server implementation.
google-generativeai: Access to Gemini AI models.
langchain_google_vertexai: Vertex AI embeddings for RAG.
pydub: Audio processing utilities.
Contributing
Fork the repository.
Create a feature branch (git checkout -b feature/your-feature).
Commit your changes (git commit -m "Add your feature").
Push to the branch (git push origin feature/your-feature).
Open a pull request.
License