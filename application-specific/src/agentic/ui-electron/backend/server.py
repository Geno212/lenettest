"""
Flask Backend Server for Electron UI

This server provides HTTP/WebSocket API for the Electron frontend
to communicate with the LangGraph workflow.
"""

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from backend/.env
backend_dir = Path(__file__).parent
env_file = backend_dir / ".env"
if env_file.exists():
    print(f"Loading environment from: {env_file}")
    load_dotenv(env_file)
else:
    print(f"Warning: .env file not found at {env_file}")
    print("Create backend/.env file with your OPENAI_API_KEY")

from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_socketio import SocketIO, emit

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.agentic.src.core.graph import create_graph
from src.agentic.src.utils.graph_utils import (
    create_user_message_input,
    create_conversation_config
)
from src.agentic.src.mcp import get_mcp_helper, cleanup_global_mcp

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'agentic-nn-generator-secret'
CORS(app)
socketio = SocketIO(
    app, 
    cors_allowed_origins="*",
    ping_timeout=600,  # 10 minutes (graph processing can take long)
    ping_interval=25,  # Send ping every 25 seconds
    async_mode='gevent',
    logger=True,
    engineio_logger=True
)

# Global state
graph = None
thread_id = None
session_id = None
current_state = {}
is_processing = False
first_message = True  # Track first message like run_chatbot.py


async def initialize_graph():
    """Initialize the LangGraph workflow."""
    global graph, thread_id, session_id
    
    try:
        print("Initializing graph...")
        graph = await create_graph()
        
        # Get MCP helper for session info
        mcp = await get_mcp_helper()
        session_id = mcp.client.session_id if mcp.client else "default"
        thread_id = f"mcp_{session_id[:8]}"
        
        print(f"Graph initialized successfully!")
        print(f"Session ID: {session_id}")
        print(f"Thread ID: {thread_id}")
        
        return True
    except Exception as e:
        print(f"Failed to initialize graph: {e}")
        return False


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'ok',
        'graph_initialized': graph is not None,
        'session_id': session_id or 'unknown',
        'thread_id': thread_id,
        'is_processing': is_processing
    })


@app.route('/api/upload-image', methods=['POST'])
def upload_image():
    """Handle image upload for model testing."""
    from werkzeug.utils import secure_filename
    import os
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    # Check if file is an image
    allowed_extensions = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
    filename = secure_filename(file.filename)
    file_ext = filename.rsplit('.', 1)[1].lower() if '.' in filename else ''
    
    if file_ext not in allowed_extensions:
        return jsonify({'error': 'Invalid file type. Please upload an image file.'}), 400
    
    try:
        # Save to temp directory
        upload_dir = Path('./uploads')
        upload_dir.mkdir(exist_ok=True)
        
        file_path = upload_dir / filename
        file.save(str(file_path))
        
        return jsonify({
            'success': True,
            'file_path': str(file_path.absolute()),
            'filename': filename
        })
        
    except Exception as e:
        print(f"Error uploading image: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/message', methods=['POST'])
def send_message():
    """Send a message to the graph."""
    global is_processing, current_state
    
    if is_processing:
        return jsonify({'error': 'Already processing a message'}), 400
    
    if not graph:
        return jsonify({'error': 'Graph not initialized'}), 500
    
    data = request.json
    message = data.get('message', '')
    
    if not message:
        return jsonify({'error': 'Message is required'}), 400
    
    try:
        is_processing = True
        
        # Emit processing started via socketio
        socketio.emit('processing_started', {'message': message})
        
        # Process message asynchronously in background
        socketio.start_background_task(process_message_background, message)
        
        # Return immediately
        return jsonify({
            'success': True,
            'message': 'Processing started'
        })
        
    except Exception as e:
        is_processing = False
        print(f"Error starting message processing: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


def process_message_background(message: str):
    """Background task to process message."""
    global is_processing, current_state
    
    import threading
    
    # Use a list to make it mutable and accessible from nested function
    heartbeat_control = {'active': True}
    
    def send_heartbeat():
        """Send periodic heartbeat to keep connection alive."""
        while heartbeat_control['active']:
            try:
                socketio.emit('heartbeat', {'timestamp': datetime.now().isoformat()})
                socketio.sleep(5)  # Send every 5 seconds
            except:
                break
    
    try:
        print(f"\n[BACKEND] Processing message: {message}")
        
        # Start heartbeat thread
        heartbeat_thread = threading.Thread(target=send_heartbeat, daemon=True)
        heartbeat_thread.start()
        
        # Create new event loop for this background task
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        result = loop.run_until_complete(process_message_async(message))
        
        loop.close()
        
        # Stop heartbeat
        heartbeat_control['active'] = False
        
        print(f"[BACKEND] Processing complete. Response length: {len(result.get('response', ''))}")
        is_processing = False
        
    except Exception as e:
        heartbeat_control['active'] = False
        is_processing = False
        print(f"[BACKEND] Error processing message: {e}")
        import traceback
        traceback.print_exc()
        
        # Emit error to frontend
        socketio.emit('processing_error', {'error': str(e)})


async def process_message_async(message: str) -> Dict[str, Any]:
    """Process a message through the graph - mimics run_chatbot.py."""
    global current_state, first_message
    
    from src.agentic.src.core.state import create_initial_state
    from src.agentic.src.utils.graph_utils import format_graph_output
    
    # Emit progress event
    socketio.emit('processing_started', {'message': message})
    
    # Create input (just like run_chatbot.py)
    if first_message:
        initial_state = create_initial_state()
        input_dict, config = create_user_message_input(
            message,
            thread_id=thread_id,
            existing_state=initial_state
        )
        first_message = False
        print(f"[BACKEND] First message - using initial state")
    else:
        input_dict, config = create_user_message_input(
            message,
            thread_id=thread_id
        )
    
    print(f"[BACKEND] Invoking graph with message: {message[:50]}...")
    
    # Invoke graph synchronously (like run_chatbot.py does)
    result = await graph.ainvoke(input_dict, config=config)
    
    print(f"[BACKEND] Graph invocation complete")
    
    # Get final state
    final_state = graph.get_state(config)
    current_state = final_state.values if final_state else {}
    
    # Extract response message using format_graph_output (like run_chatbot.py)
    response_message = format_graph_output(result)
    
    print(f"[BACKEND] Response: {response_message[:100]}...")
    
    # Emit completion event
    socketio.emit('processing_complete', {
        'response': response_message,
        'state': serialize_state(current_state)
    })
    
    return {
        'success': True,
        'response': response_message,
        'state': serialize_state(current_state)
    }


@app.route('/api/state', methods=['GET'])
def get_state():
    """Get current graph state."""
    if not graph:
        return jsonify({'error': 'Graph not initialized'}), 500
    
    try:
        config = create_conversation_config(thread_id)
        state = graph.get_state(config)
        
        if state and state.values:
            return jsonify({
                'success': True,
                'state': serialize_state(state.values)
            })
        else:
            return jsonify({
                'success': True,
                'state': {}
            })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/reset', methods=['POST'])
def reset_session():
    """Reset the current session."""
    global current_state, first_message
    
    current_state = {}
    first_message = True  # Reset first message flag
    
    # Could reinitialize graph here if needed
    
    return jsonify({
        'success': True,
        'message': 'Session reset'
    })


def serialize_state(state: Dict[str, Any]) -> Dict[str, Any]:
    """Serialize state for JSON response."""
    serialized = {}
    
    # Only include serializable fields
    simple_fields = [
        'project_name', 'project_path',
        'architecture_name', 'architecture_path',
        'config_path', 'training_status',
        'dialog_state',
        'awaiting_test_image',  # UI control flag for upload button
        'awaiting_new_design_choice',  # Post-training UI state
        'needs_user_input',  # General UI state
    ]
    
    for field in simple_fields:
        if field in state:
            serialized[field] = state[field]
    
    # Handle complex fields
    if 'generated_code_paths' in state:
        serialized['generated_code_paths'] = state['generated_code_paths']
    
    if 'training_results' in state and state['training_results']:
        serialized['training_results'] = str(state['training_results'])[:200]
    
    # Message count
    if 'messages' in state:
        serialized['message_count'] = len(state['messages'])
    
    return serialized


@socketio.on('connect')
def handle_connect():
    """Handle client connection."""
    print('Client connected')
    emit('connected', {
        'session_id': session_id or 'unknown',
        'thread_id': thread_id,
        'graph_ready': graph is not None
    })


@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection."""
    print('Client disconnected')


def main():
    """Main entry point."""
    print("=" * 60)
    print("AGENTIC NN GENERATOR - BACKEND SERVER")
    print("=" * 60)
    
    # Initialize graph
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    success = loop.run_until_complete(initialize_graph())
    
    if not success:
        print("Failed to initialize graph. Exiting...")
        sys.exit(1)
    
    print("\nStarting Flask server on http://127.0.0.1:5000")
    print("WebSocket available at ws://127.0.0.1:5000")
    print("\n" + "=" * 60 + "\n")
    
    # Run Flask app
    socketio.run(app, host='127.0.0.1', port=5000, debug=False)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nShutting down...")
        
        # Cleanup
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(cleanup_global_mcp())
        loop.close()
        
        print("Goodbye!")
