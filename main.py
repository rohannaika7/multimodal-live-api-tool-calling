import asyncio
import json
import os
import websockets
from google import genai
import base64
from langchain_core.documents import Document
from langchain_google_vertexai import VertexAIEmbeddings
from google.generativeai import types
import google.generativeai as generative

# Configuration and Initialization
########################################

# Load API key from environment variable
os.environ['GOOGLE_API_KEY'] = 'YOUR_API_KEY'  # Set Google API key as environment variable
generative.configure(api_key=os.environ["GOOGLE_API_KEY"])  # Configure generative AI client with API key

# Model configurations
MODEL = "gemini-2.0-flash-exp"  # Primary model for text generation
TRANSCRIPTION_MODEL = 'gemini-1.5-flash-8b'  # Specialized model for transcription tasks

# Initialize Gemini client with custom API version
client = genai.Client(
    http_options={'api_version': 'v1alpha'}  # Custom API version for Gemini client
)

# Initialize embeddings for document similarity
embeddings = VertexAIEmbeddings(
    requests_per_minute=150,  # Rate limit for embedding requests
    model_name='textembedding-gecko@003'  # Specific embedding model version
)

# Sample Documents
########################################

# In-memory document store for Retrieval-Augmented Generation (RAG)
DOCUMENTS = [
    Document(
        page_content="The sky is blue because of Rayleigh scattering.",
        metadata={"source": "science"}  # Metadata for document tracking
    ),
    Document(
        page_content="Python is a popular programming language.",
        metadata={"source": "tech"}
    ),
    Document(
        page_content="AI is transforming industries rapidly.",
        metadata={"source": "tech"}
    ),
]

# Precompute embeddings for all documents
document_embeddings = embeddings.embed_documents([doc.page_content for doc in DOCUMENTS])

interrupt = False  # Global flag to track interruption state

# Retrieval-Augmented Generation (RAG)
########################################

def rag(query: str) -> str:
    """
    Retrieve relevant documents based on the query and generate a response.
    
    Args:
        query (str): The user's question or input query
        
    Returns:
        str: A prompt containing the query and relevant document information
    """
    # Convert query to vector embedding
    query_embedding = embeddings.embed_query(query)
    
    # Calculate similarity scores using dot product between query and document embeddings
    similarities = [
        sum(a * b for a, b in zip(query_embedding, doc_emb))
        for doc_emb in document_embeddings
    ]
    
    # Identify most relevant document based on highest similarity score
    top_doc_idx = similarities.index(max(similarities))
    retrieved_doc = DOCUMENTS[top_doc_idx].page_content
    
    # Construct prompt with query and retrieved document content
    prompt = f"Query: {query}\nRelevant Information: {retrieved_doc}\nAnswer the query based on the provided information."
    return prompt

# Tool Configuration
########################################

tool_rag_values = {
    "function_declarations": [
        {
            "name": "set_rag_values",
            "description": """An extensive data repository built to gather, examine, and provide 
                            detailed insights for inquiries. Make sure all required details are 
                            included; if anything is missing or ambiguous, request clarification 
                            from the user before moving forward.""",
            "parameters": {
                "type": "OBJECT",
                "properties": {
                    "query": {
                        "type": "STRING",
                        "description": "The query to process"
                    }
                },
                "required": ["query"]  # Query parameter is mandatory
            }
        }
    ]
}

# WebSocket Handler
########################################

async def gemini_session_handler(client_websocket: websockets.WebSocketServerProtocol):
    """
    Handles WebSocket connections and interactions with the Gemini API.
    
    Args:
        client_websocket: The WebSocket connection object for client communication
    """
    try:
        # Receive and parse initial configuration from client
        config_message = await client_websocket.recv()
        config_data = json.loads(config_message)
        
        # Extract generation configuration settings
        generation_config = config_data["setup"].get("generation_config", {})
        response_modalities = generation_config.get("response_modalities", [])
        
        # Configure session settings
        config = config_data.get("setup", {})
        config["tools"] = [tool_rag_values]  # Add RAG tool to configuration
        config["system_instruction"] = """
        INSTRUCTIONS:
- Rely solely on the `tool_rag_values` tool to obtain information.
- Avoid using any built-in or prior knowledge, and base responses entirely on the data provided by the specified tool."""
        config["generation_config"] = {"response_modalities": response_modalities}

        # Establish asynchronous connection to Gemini API
        async with client.aio.live.connect(model=MODEL, config=config) as session:
            print("Connected to Gemini API")

            async def send_to_gemini():
                """
                Forward client messages to Gemini API.
                Handles text, audio, and image inputs asynchronously.
                """
                try:
                    async for message in client_websocket:
                        try:
                            data = json.loads(message)
                            
                            # Process turn-based content from client
                            if "client_content" in data:
                                for chunk in data["client_content"]["turns"]:
                                    if "parts" in chunk and "text" in chunk["parts"][0]:
                                        await session.send(
                                            input=chunk["parts"][0]["text"] or "."  # Default to "." if empty
                                        )
                            
                            # Process real-time media inputs
                            if "realtime_input" in data:
                                for chunk in data["realtime_input"]["media_chunks"]:
                                    mime_type = chunk["mime_type"]
                                    if mime_type in ["audio/pcm", "image/jpeg", "text"]:
                                        await session.send(
                                            input={"mime_type": mime_type, "data": chunk["data"]}
                                            if mime_type != "text" else chunk["data"] or "."
                                        )
                        except Exception as e:
                            print(f"Error sending to Gemini: {e}")
                except Exception as e:
                    print(f"Error in send_to_gemini: {e}")
                finally:
                    print("send_to_gemini closed")

            async def receive_from_gemini():
                """
                Receive responses from Gemini API and forward to client.
                Handles text, audio, and tool call responses.
                """
                try:
                    global interrupt
                    while True:
                        async for response in session.receive():
                            if response.server_content is None:
                                # Process tool calls from the model
                                if response.tool_call is not None:
                                    function_calls = response.tool_call.function_calls
                                    function_responses = []

                                    for function_call in function_calls:
                                        if function_call.name == "set_rag_values":
                                            try:
                                                result = rag(function_call.args["query"])
                                                call_id = function_call.id
                                                function_responses.append({
                                                    "name": function_call.name,
                                                    "response": {"result": result},
                                                    "id": call_id
                                                })
                                                tool_response = types.LiveClientToolResponse(
                                                    function_responses=[
                                                        types.FunctionResponse(
                                                            name=function_call.name,
                                                            id=call_id,
                                                            response=result
                                                        )
                                                    ]
                                                )
                                                await client_websocket.send(json.dumps({"text": json.dumps(function_responses)}))
                                                await session.send(input=tool_response)
                                            except Exception as e:
                                                print(f"Error executing function: {e}")
                                                continue

                            # Process model-generated content
                            server_content = response.server_content
                            
                            if server_content is not None:
                                if server_content.interrupted:
                                    interrupt = True
                                model_turn = server_content.model_turn
                                if model_turn:
                                    if response.server_content.interrupted:
                                        interrupt = True
                                    for part in model_turn.parts:
                                        if interrupt:
                                            interrupt = False
                                            break
                                        # Handle text response
                                        if hasattr(part, 'text') and part.text:
                                            await client_websocket.send(
                                                json.dumps({
                                                    "serverContent": {
                                                        "modelTurn": {
                                                            "role": "model",
                                                            "parts": [{"text": part.text}]
                                                        }
                                                    }
                                                })
                                            )
                                        # Handle audio response
                                        elif hasattr(part, 'inline_data') and part.inline_data:
                                            base64_audio = base64.b64encode(part.inline_data.data).decode('utf-8')
                                            await client_websocket.send(
                                                json.dumps({
                                                    "serverContent": {
                                                        "modelTurn": {
                                                            "role": "model",
                                                            "parts": [{
                                                                "inlineData": {
                                                                    "mimeType": "audio/pcm",
                                                                    "data": base64_audio
                                                                }
                                                            }]
                                                        }
                                                    }
                                                })
                                            )

                            if response.server_content.turn_complete:
                                print('\n<Turn complete>')
                except Exception as e:
                    print(f"Error receiving from Gemini: {e}")
                finally:
                    print("Gemini connection closed (receive)")

            # Execute send and receive operations concurrently
            send_task = asyncio.create_task(send_to_gemini())
            receive_task = asyncio.create_task(receive_from_gemini())
            await asyncio.gather(send_task, receive_task)

    except Exception as e:
        print(f"Error in Gemini session: {e}")
    finally:
        print("Gemini session closed.")

# Main Execution
########################################

async def main() -> None:
    """Start the WebSocket server on localhost:9011."""
    async with websockets.serve(gemini_session_handler, "localhost", 9011):
        print("Running websocket server localhost:9011...")
        await asyncio.Future()  # Keep server running indefinitely

if __name__ == "__main__":
    asyncio.run(main())  # Run the main async function