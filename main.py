from fastapi import FastAPI, HTTPException, Body, BackgroundTasks, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import uvicorn
import os
import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import uuid
import time
import tempfile
import shutil
import ffmpeg
from pathlib import Path
from langchain_ollama import ChatOllama
from langchain.schema import HumanMessage, SystemMessage
import logging
from faster_whisper import WhisperModel

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure environment variables for optimal performance on CPU
os.environ["OMP_NUM_THREADS"] = str(max(1, os.cpu_count() // 2))  # Set optimal number of OpenMP threads
os.environ["CT2_VERBOSE"] = "0"  # Reduce CTranslate2 verbosity

# Configure thread pool - use fewer workers for transcription to avoid memory contention
CPU_COUNT = os.cpu_count() or 4
MAX_WORKERS = max(4, CPU_COUNT)  # For general tasks
TRANSCRIPTION_WORKERS = max(2, CPU_COUNT // 2)  # Fewer workers for memory-intensive transcription
executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)
transcription_executor = ThreadPoolExecutor(max_workers=TRANSCRIPTION_WORKERS)
logger.info(f"Initialized thread pools with {MAX_WORKERS} general workers and {TRANSCRIPTION_WORKERS} transcription workers")

# Request tracking
active_requests: Dict[str, Dict[str, Any]] = {}

# Transcription tracking
active_transcriptions: Dict[str, Dict[str, Any]] = {}

app = FastAPI(title="Gemma 3 and Whisper API", description="API for interacting with Gemma 3 and Fast-Whisper using LangChain and Ollama")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Models
class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]
    system_prompt: Optional[str] = None
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 1024

class ChatResponse(BaseModel):
    response: str
    usage: dict
    request_id: str

class StatusResponse(BaseModel):
    status: str
    active_requests: int
    active_transcriptions: int

class TranscriptionRequest(BaseModel):
    language: Optional[str] = None
    task: Optional[str] = "transcribe"  # 'transcribe' or 'translate'
    beam_size: Optional[int] = 1  # Reduced from 5 to 1 for better CPU performance
    word_timestamps: Optional[bool] = False
    vad_filter: Optional[bool] = True  # Enabled by default for faster processing
    model_size: Optional[str] = "tiny"  # Default to tiny for lower memory usage

class TranscriptionResponse(BaseModel):
    transcription_id: str
    status: str
    message: str

class TranscriptionResult(BaseModel):
    text: str
    segments: Optional[List[Dict[str, Any]]] = None
    language: Optional[str] = None
    task: str
    duration_seconds: float

# LLM instance cache to reuse connections
llm_cache = {}
whisper_model_cache = {}

def get_llm(model: str, temperature: float, max_tokens: int, base_url: str):
    # Create a cache key based on the parameters
    cache_key = f"{model}_{temperature}_{max_tokens}_{base_url}"
    
    if cache_key not in llm_cache:
        logger.info(f"Creating new LLM instance for {cache_key}")
        llm_cache[cache_key] = ChatOllama(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            base_url=base_url
        )
    
    return llm_cache[cache_key]

def get_whisper_model(model_size: str):
    """Get or create a cached Whisper model instance optimized for CPU."""
    if model_size not in whisper_model_cache:
        logger.info(f"Loading Fast-Whisper {model_size} model...")
        
        # Set optimal compute type for CPU (int8 for best memory usage)
        compute_type = "int8"
        
        # Configure number of threads
        threads = min(4, os.cpu_count() or 4)  # Limit to 4 threads to avoid memory spikes
        
        # Load the model with CPU optimization
        whisper_model_cache[model_size] = WhisperModel(
            model_size,
            device="cpu",
            compute_type=compute_type,
            cpu_threads=threads,
            download_root=os.path.join(os.path.expanduser("~"), ".cache", "whisper")
        )
        
        logger.info(f"Fast-Whisper {model_size} model loaded on CPU with {compute_type} precision and {threads} threads")
    
    return whisper_model_cache[model_size]

# Process LLM requests in a separate thread
def process_llm_request(request_id: str, messages: List, model: str, temperature: float, max_tokens: int, base_url: str):
    try:
        # Update request status
        active_requests[request_id]["status"] = "processing"
        
        # Get or create LLM instance
        llm = get_llm(model, temperature, max_tokens, base_url)
        
        # Track start time
        start_time = time.time()
        
        # Generate response
        response = llm.invoke(messages)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Extract the response content
        response_content = response.content if hasattr(response, 'content') else str(response)
        
        # Prepare result
        result = {
            "response": response_content,
            "usage": {
                "processing_time_seconds": processing_time,
                "estimated_tokens": len(response_content.split()) * 1.3  # Very rough approximation
            }
        }
        
        # Update request status
        active_requests[request_id]["status"] = "completed"
        active_requests[request_id]["result"] = result
        
        return result
    
    except Exception as e:
        logger.error(f"Error processing request {request_id}: {str(e)}")
        active_requests[request_id]["status"] = "failed"
        active_requests[request_id]["error"] = str(e)
        return {"error": str(e)}

def extract_audio_from_video(video_path, output_path):
    """Extract audio from video file using ffmpeg."""
    try:
        # Ensure ffmpeg is installed and accessible
        if not shutil.which("ffmpeg"):
            raise Exception("FFmpeg is not installed or not in PATH")

        # Use ffmpeg to extract audio as WAV file
        ffmpeg.input(video_path).output(output_path, acodec='pcm_s16le', ac=1, ar='16k').run(quiet=True, overwrite_output=True)
        return True
    except Exception as e:
        logger.error(f"Error extracting audio: {str(e)}")
        return False

def process_transcription(transcription_id: str, file_path: str, params: Dict[str, Any]):
    """Process transcription in a separate thread with optimized memory usage."""
    try:
        # Update transcription status
        active_transcriptions[transcription_id]["status"] = "processing"
        
        start_time = time.time()
        file_extension = Path(file_path).suffix.lower()
        
        # Create temporary directory for processing
        with tempfile.TemporaryDirectory() as temp_dir:
            audio_path = file_path
            
            # If video file, extract audio first
            if file_extension in ['.mp4', '.avi', '.mov', '.mkv', '.webm']:
                logger.info(f"Extracting audio from video file {file_path}")
                audio_path = os.path.join(temp_dir, "audio.wav")
                if not extract_audio_from_video(file_path, audio_path):
                    raise Exception("Failed to extract audio from video file")
            
            # Load the model
            model = get_whisper_model(params.get("model_size", "tiny"))
            
            # Set optimal batch size based on documentation
            batch_size = 8 if params.get("model_size") in ["tiny", "base"] else 4

            # Run transcription with optimized parameters
            logger.info(f"Running transcription with params: {params}")
            
            # Set up VAD parameters for optimal processing
            vad_parameters = {
                "threshold": 0.5,  # Lower threshold for faster processing
                "min_speech_duration_ms": 250,  # Shorter minimum speech duration
                "max_speech_duration_s": 15,  # Reasonable limit
                "min_silence_duration_ms": 1000,  # 1 second minimum silence
                "window_size_samples": 512,  # Smaller window for better CPU performance
            }
            
            result = model.transcribe(
                audio_path,
                beam_size=params.get("beam_size", 1),  # Use beam size 1 for CPU
            )
            
            # Process results
            segments = []
            text = ""
            
            for segment in result[0]:
                text += segment.text + " "
                segment_dict = {
                    "id": len(segments),
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text,
                }
                
                # Only include word-level info if requested to save memory
                if params.get("word_timestamps", False):
                    segment_dict["words"] = [
                        {"word": w.word, "start": w.start, "end": w.end} 
                        for w in segment.words
                    ]
                
                segments.append(segment_dict)
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Prepare result (minimal version to save memory)
            transcription_result = {
                "text": text.strip(),
                "segments": segments,
                "language": result[1],
                "task": params.get("task", "transcribe"),
                "duration_seconds": processing_time
            }
            
            # Update status
            active_transcriptions[transcription_id]["status"] = "completed"
            active_transcriptions[transcription_id]["result"] = transcription_result
            
            return transcription_result
            
    except Exception as e:
        logger.error(f"Error in transcription {transcription_id}: {str(e)}")
        active_transcriptions[transcription_id]["status"] = "failed"
        active_transcriptions[transcription_id]["error"] = str(e)
        return {"error": str(e)}
    finally:
        # Clean up the original file after processing
        if os.path.exists(file_path) and not file_path.startswith('/tmp'):
            try:
                os.remove(file_path)
            except Exception as e:
                logger.error(f"Failed to clean up file {file_path}: {e}")

# Initialize LLM with default parameters
@app.on_event("startup")
async def startup_event():
    logger.info("Initializing application...")
    # Check if Ollama is running and Gemma is available
    try: 
        # Initialize ChatOllama with Gemma 3 1b model
        llm = ChatOllama(model="gemma3:1b")
        # Test connection with a simple query
        messages = [HumanMessage(content="Hello")]
        llm.invoke(messages)
        logger.info("Successfully connected to Ollama with Gemma 3:1b")
    except Exception as e:
        logger.error(f"Failed to initialize Gemma 3 model: {e}")
        logger.error("Please ensure Ollama is running and Gemma 3:1b model is available")
        logger.error("You may need to run: ollama run gemma3:1b")
    
    # Don't pre-load Whisper model to save memory, will load on first use
    logger.info("Fast-Whisper models will be loaded on first use to conserve memory")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down thread pool executors")
    executor.shutdown(wait=False)
    transcription_executor.shutdown(wait=False)
    
    # Clear model caches to free memory
    whisper_model_cache.clear()
    llm_cache.clear()

# Health check endpoint
@app.get("/health", response_model=StatusResponse)
async def health_check():
    return StatusResponse(
        status="ok",
        active_requests=len([req for req in active_requests.values() if req["status"] == "processing"]),
        active_transcriptions=len([req for req in active_transcriptions.values() if req["status"] == "processing"])
    )

async def cleanup_after_delay(dictionary, key, delay):
    await asyncio.sleep(delay)
    dictionary.pop(key, None)

# Chat completion endpoint - asynchronous version
@app.post("/v1/chat/completions", response_model=ChatResponse)
async def chat_completion(request: ChatRequest, background_tasks: BackgroundTasks):
    try:
        logger.info(f"Received chat request with {len(request.messages)} messages")
        
        # Generate a unique ID for this request
        request_id = str(uuid.uuid4())
        
        # Convert messages to LangChain format
        messages = []
        
        # Add system prompt if provided
        if request.system_prompt:
            messages.append(SystemMessage(content=request.system_prompt))
        
        # Add conversation messages
        for msg in request.messages:
            if msg.role.lower() == "user":
                messages.append(HumanMessage(content=msg.content))
            elif msg.role.lower() == "system":
                messages.append(SystemMessage(content=msg.content))
        
        # Get Ollama host from environment or use default
        ollama_host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        
        # Register the request
        active_requests[request_id] = {
            "status": "queued",
            "start_time": time.time(),
            "result": None
        }
        
        # Process the request in a background task using thread pool
        loop = asyncio.get_event_loop()
        process_func = partial(
            process_llm_request,
            request_id,
            messages,
            "gemma3:1b",
            request.temperature,
            request.max_tokens,
            ollama_host
        )
        
        # Run in the thread pool and handle the result
        future = loop.run_in_executor(executor, process_func)
        
        # Add a background task to clean up the request data after completion
        background_tasks.add_task(cleanup_after_delay, active_requests, request_id, 300)
        
        # Wait for the result 
        result = await future
        
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
            
        # Return the response with request ID
        return ChatResponse(
            response=result["response"],
            usage=result["usage"],
            request_id=request_id
        )
        
    except Exception as e:
        logger.error(f"Error in chat completion: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

# Video transcription endpoint
@app.post("/v1/transcribe", response_model=TranscriptionResponse)
async def transcribe_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    beam_size: Optional[int] = Form(1) # Default to 1 for CPU
):
    try:
        # Validate file type
        allowed_extensions = ['.mp3', '.mp4', '.wav', '.m4a', '.webm', '.mov', '.avi', '.mkv']
        file_extension = os.path.splitext(file.filename)[1].lower()
        
        if file_extension not in allowed_extensions:
            raise HTTPException(
                status_code=400, 
                detail=f"File type not supported. Allowed types: {', '.join(allowed_extensions)}"
            )
        
        # Create a unique transcription ID
        transcription_id = str(uuid.uuid4())
        
        # Create a temporary file to store the uploaded content
        temp_dir = tempfile.gettempdir()
        temp_file_path = os.path.join(temp_dir, f"{transcription_id}{file_extension}")
        
        # Save the uploaded file to the temporary location
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Register the transcription request
        active_transcriptions[transcription_id] = {
            "status": "queued",
            "start_time": time.time(),
            "file_path": temp_file_path,
            "original_filename": file.filename,
            "result": None
        }
        
        # Parameters for transcription - optimized for CPU
        params = {
            "beam_size": beam_size
        }
        
        # Process the transcription in the specialized transcription thread pool
        loop = asyncio.get_event_loop()
        process_func = partial(
            process_transcription,
            transcription_id,
            temp_file_path,
            params
        )
        
        # Run in the thread pool
        future = loop.run_in_executor(transcription_executor, process_func)
        
        # Schedule cleanup of transcription data after some time (reduced to 30 minutes to save memory)
        background_tasks.add_task(cleanup_after_delay, active_transcriptions, transcription_id, 1800)
        
        # Don't wait for the result - return immediately
        return TranscriptionResponse(
            transcription_id=transcription_id,
            status="processing",
            message="Transcription job started. Use GET /v1/transcribe/{transcription_id} to check status."
        )
        
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error in transcription request: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing transcription: {str(e)}")

# Get transcription status and results
@app.get("/v1/transcribe/{transcription_id}", response_model=Dict[str, Any])
async def get_transcription_status(transcription_id: str):
    if transcription_id not in active_transcriptions:
        raise HTTPException(status_code=404, detail="Transcription job not found")
    
    transcription_data = active_transcriptions[transcription_id]
    response = {
        "transcription_id": transcription_id,
        "status": transcription_data["status"],
        "original_filename": transcription_data.get("original_filename", "unknown"),
        "elapsed_time": time.time() - transcription_data["start_time"]
    }
    
    # Include result or error if available
    if transcription_data["status"] == "completed":
        response["result"] = transcription_data.get("result")
    elif transcription_data["status"] == "failed":
        response["error"] = transcription_data.get("error")
    
    return response

# Get specific request status
@app.get("/v1/requests/{request_id}")
async def get_request_status(request_id: str):
    if request_id not in active_requests:
        raise HTTPException(status_code=404, detail="Request not found")
    
    request_data = active_requests[request_id]
    return {
        "request_id": request_id,
        "status": request_data["status"],
        "elapsed_time": time.time() - request_data["start_time"],
        "result": request_data.get("result", None) if request_data["status"] == "completed" else None,
        "error": request_data.get("error", None) if request_data["status"] == "failed" else None
    }

# Model info endpoint
@app.get("/v1/models")
async def get_models():
    return {
        "models": [
            {
                "id": "gemma3:1b",
                "name": "Gemma 3 1B",
                "description": "Google's Gemma 3 1B open model served via Ollama and LangChain",
                "max_tokens": 8192
            },
            {
                "id": "whisper",
                "name": "Fast-Whisper",
                "description": "Faster implementation of OpenAI's Whisper model with CTranslate2",
                "sizes": ["tiny", "base", "small", "medium"]  # Removed "large" as it uses too much memory for CPU
            }
        ]
    }

# Run the API with uvicorn
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)