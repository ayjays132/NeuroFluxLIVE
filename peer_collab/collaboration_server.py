from flask import Flask, request, jsonify, g
import torch
from collections import defaultdict
import time
import os
import logging
from functools import wraps
from typing import List, Dict, Any, Set, Optional

# --- Configuration Management ---
class Config:
    """
    Application configuration class.
    Uses environment variables for sensitive data and provides defaults.
    In a production environment, these would be loaded securely (e.g., from Kubernetes secrets, HashiCorp Vault).
    """
    SECRET_KEY = os.environ.get('FLASK_SECRET_KEY', 'super_secret_dev_key_change_in_prod_12345')
    # Example for database URI (conceptual, not implemented in this version)
    DATABASE_URI = os.environ.get('DATABASE_URI', 'sqlite:///dev_collaboration.db')
    LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO').upper() # Default log level

# --- Custom Logging Setup for Structured and Colored Output ---
class ColoredFormatter(logging.Formatter):
    """
    A custom logging formatter that adds ANSI escape codes for colored console output,
    enhancing readability of log messages for different severity levels.
    """
    FORMAT = "%(asctime)s | %(levelname)s | %(message)s"
    DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
    
    LOG_COLORS = {
        logging.DEBUG: "\033[36m", # Cyan
        logging.INFO: "\033[34m",  # Blue
        logging.WARNING: "\033[33m", # Yellow
        logging.ERROR: "\033[31m\033[1m", # Red + Bold
        logging.CRITICAL: "\033[91m\033[1m\033[4m" # Bright Red + Bold + Underline
    }
    RESET_COLOR = "\033[0m"

    def format(self, record):
        log_fmt = self.LOG_COLORS.get(record.levelno, self.RESET_COLOR) + self.FORMAT + self.RESET_COLOR
        formatter = logging.Formatter(log_fmt, datefmt=self.DATE_FORMAT)
        return formatter.format(record)

# Initialize application-wide logger
logger = logging.getLogger(__name__)
# Clear existing handlers to prevent duplicate logs if reloaded
if logger.handlers:
    for handler in logger.handlers:
        logger.removeHandler(handler)
# Set logging level from Config
logger.setLevel(Config.LOG_LEVEL)
# Add console handler with custom formatter
console_handler = logging.StreamHandler()
console_handler.setFormatter(ColoredFormatter())
logger.addHandler(console_handler)

# --- Authentication and Authorization Decorators (Conceptual for Production) ---
def login_required(f):
    """
    A decorator to simulate user authentication. In a production environment,
    this would involve verifying a token (e.g., JWT) or session.
    For this example, it checks for an 'X-User-Id' header.
    """
    @wraps(f)
    def decorated_function(*args, **kwargs):
        user_id = request.headers.get('X-User-Id')
        if not user_id:
            logger.warning("Authentication failed: 'X-User-Id' header missing.")
            return jsonify({'message': 'Authentication required. Please provide X-User-Id header.'}), 401
        
        # Store user_id in Flask's global context for easy access throughout the request
        g.user_id = user_id 
        logger.debug(f"User '{user_id}' authenticated for endpoint '{request.path}'")
        return f(*args, **kwargs)
    return decorated_function

def project_permission_required(permission_level: str):
    """
    A decorator to simulate project-specific authorization.
    In a real system, this would involve a more robust RBAC (Role-Based Access Control) system
    querying a database for user roles within a specific project.
    
    For demonstration, it's a placeholder.
    """
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            project_id = kwargs.get('project_id')
            user_id = g.user_id # Assumes login_required has already run
            
            # --- Placeholder for real RBAC logic ---
            # if user_id not in server.project_members[project_id]: # Access server instance via closure
            #     logger.warning(f"Authorization failed: User '{user_id}' not a member of project '{project_id}'.")
            #     return jsonify({'message': f'Access denied. User {user_id} is not a member of project {project_id}.'}), 403
            
            # Further checks for 'permission_level' (e.g., 'read', 'write', 'admin')
            # role = server.project_roles[project_id].get(user_id)
            # if permission_level == 'write' and role not in ['editor', 'admin']:
            #     logger.warning(f"Authorization failed: User '{user_id}' has insufficient role '{role}' for write access on project '{project_id}'.")
            #     return jsonify({'message': 'Permission denied. Insufficient role for write access.'}), 403
            # --- End Placeholder ---

            logger.debug(f"User '{user_id}' authorized for '{permission_level}' on project '{project_id}'")
            return f(*args, **kwargs)
        return decorated_function
    return decorator


class CollaborationServer:
    """
    A robust Flask-based REST API for multi-user collaboration on research projects.
    It provides secure (conceptually) endpoints for managing shared notes, feedback,
    and project membership, designed with extensibility for real-time updates and
    integration with PyTorch-based AI services.

    Key Features:
    - Centralized state management for notes, feedback, and members (in-memory for demo, easily extensible to DB).
    - Simulated PyTorch computations for showcasing potential AI backend integrations.
    - Basic API routing with HTTP method handling.
    - Enhanced logging for operational visibility.
    - Conceptual authentication and authorization decorators for security.
    - Structured error handling.
    """
<<<<<<< HEAD
    def __init__(self, host='0.0.0.0', port=5000, device='cpu'):
=======
    def __init__(self, host: str = '0.0.0.0', port: int = 5000, device: str = 'cpu'):
        """
        Initializes the CollaborationServer.

        Args:
            host (str): The host IP address for the Flask server to listen on.
            port (int): The port number for the Flask server.
            device (str): The PyTorch device ('cuda' or 'cpu') to perform dummy computations on.
                          Automatically falls back to 'cpu' if 'cuda' is not available.
        """
>>>>>>> c9d035a (WIP: saving my local progress)
        self.app = Flask(__name__)
        self.app.config.from_object(Config) # Load configurations

        self.host = host
        self.port = port
        
        # --- PyTorch Device Initialization ---
        if device == 'cuda':
            if torch.cuda.is_available():
                self.device = torch.device(device)
                logger.info(f"CollaborationServer running dummy PyTorch computations on GPU: {self.device}")
            else:
                logger.warning(f"CUDA requested but not available. Falling back to CPU for PyTorch operations.")
                self.device = torch.device('cpu')
                logger.info(f"CollaborationServer running dummy PyTorch computations on CPU (CUDA unavailable).")
        else: # Default to 'cpu' if not 'cuda' or explicitly 'cpu'
            self.device = torch.device('cpu')
            logger.info(f"CollaborationServer running dummy PyTorch computations on CPU.")

        # --- In-Memory Data Stores (for demonstration purposes only) ---
        # In a production environment, these would be replaced by database connections (e.g., SQLAlchemy, Firestore).
        self.shared_notes: Dict[str, str] = defaultdict(str) # project_id -> notes content
        self.feedback_data: Dict[str, List[Dict[str, Any]]] = defaultdict(list) # project_id -> list of feedback items
        self.project_members: Dict[str, Set[str]] = defaultdict(set) # project_id -> set of user_ids
        self.project_roles: Dict[str, Dict[str, str]] = defaultdict(dict) # project_id -> {user_id -> role}

        self._setup_routes()
        self._setup_error_handlers() # Register global error handlers
        
        logger.info(f"CollaborationServer fully initialized and ready to serve on {self.host}:{self.port}.")

    def _setup_error_handlers(self):
        """Registers global error handlers for common HTTP status codes."""
        @self.app.errorhandler(400)
        def bad_request_error(error):
            logger.error(f"Bad Request (400): {error.description}")
            return jsonify({'message': 'Bad Request', 'error': error.description}), 400

        @self.app.errorhandler(401)
        def unauthorized_error(error):
            logger.warning(f"Unauthorized (401): {error.description}")
            return jsonify({'message': 'Unauthorized', 'error': error.description}), 401

        @self.app.errorhandler(403)
        def forbidden_error(error):
            logger.warning(f"Forbidden (403): {error.description}")
            return jsonify({'message': 'Forbidden', 'error': error.description}), 403

        @self.app.errorhandler(404)
        def not_found_error(error):
            logger.warning(f"Not Found (404): {error.description}")
            return jsonify({'message': 'Resource not found', 'error': error.description}), 404

        @self.app.errorhandler(405)
        def method_not_allowed_error(error):
            logger.warning(f"Method Not Allowed (405): {error.description}")
            return jsonify({'message': 'Method not allowed', 'error': error.description}), 405

        @self.app.errorhandler(500)
        def internal_server_error(error):
            # Log the full exception traceback for debugging
            logger.exception(f"Internal Server Error (500): {error.description}")
            return jsonify({'message': 'Internal Server Error', 'error': "An unexpected error occurred."}), 500

    def _setup_routes(self):
<<<<<<< HEAD
        @self.app.route('/notes/<project_id>', methods=['GET', 'POST'])
        def handle_notes(project_id):
            # In a real app, you'd verify user authentication and permissions here
            if request.method == 'GET':
                # Simulate some PyTorch computation
                dummy_tensor = torch.randn(3, 3, device=self.device)
                _ = torch.inverse(dummy_tensor) if dummy_tensor.det() != 0 else dummy_tensor # Dummy computation
                return jsonify({'notes': self.shared_notes[project_id]})
            elif request.method == 'POST':
                data = request.json
                notes = data.get('notes', '')
                user_id = data.get('user_id', 'anonymous') # Simulate user
                self.shared_notes[project_id] = notes
                # Simulate some PyTorch computation
                dummy_tensor = torch.tensor([len(notes)], dtype=torch.float32, device=self.device)
                _ = torch.log(dummy_tensor + 1) # Dummy computation
                print(f"Project {project_id} notes updated by {user_id}")
                return jsonify({'message': 'Notes updated successfully', 'notes': notes})

        @self.app.route('/feedback/<project_id>', methods=['GET', 'POST'])
        def handle_feedback(project_id):
            if request.method == 'GET':
                # Simulate some PyTorch computation
                dummy_tensor = torch.randn(2, 2, device=self.device)
                _ = torch.exp(dummy_tensor) # Dummy computation
                return jsonify({'feedback': self.feedback_data[project_id]})
=======
        """
        Defines and registers all API routes for the collaboration server.
        Each route includes logging, basic validation, and conceptual security checks.
        """
        @self.app.route('/notes/<project_id>', methods=['GET', 'POST'])
        @login_required # Requires 'X-User-Id' header
        @project_permission_required(permission_level='read_write_notes') # Conceptual permission
        def handle_notes(project_id: str):
            """
            Handles operations for shared project notes.
            GET: Retrieves notes content for a given project.
            POST: Updates notes content for a given project.
            """
            logger.info(f"Received {request.method} request for notes on project: {project_id}")
            
            if request.method == 'GET':
                try:
                    # Simulate some PyTorch computation to demonstrate backend integration
                    dummy_tensor = torch.randn(3, 3, device=self.device)
                    # Perform a simple operation to involve PyTorch
                    _ = torch.inverse(dummy_tensor) if dummy_tensor.det() != 0 else dummy_tensor.sum() 
                    
                    notes_content = self.shared_notes[project_id]
                    logger.info(f"Fetched notes for project '{project_id}'. Length: {len(notes_content)} chars.")
                    return jsonify({'notes': notes_content})
                except Exception as e:
                    logger.error(f"Error fetching notes for project '{project_id}': {e}", exc_info=True)
                    # Raise an exception to be caught by the 500 error handler
                    raise Exception(f"Failed to retrieve notes: {e}") 

>>>>>>> c9d035a (WIP: saving my local progress)
            elif request.method == 'POST':
                data = request.json
                if not data:
                    return jsonify({'message': 'Request body must be JSON'}), 400
                
                notes = data.get('notes')
                if not isinstance(notes, str):
                    return jsonify({'message': 'Invalid input: "notes" must be a string.'}), 400
                
                # Update in-memory store
                self.shared_notes[project_id] = notes
                
                try:
                    # Simulate PyTorch computation based on notes length
                    dummy_tensor = torch.tensor([len(notes)], dtype=torch.float32, device=self.device)
                    _ = torch.log(dummy_tensor + 1) # Ensure log doesn't take log of zero
                    
                    logger.info(f"Project '{project_id}' notes updated by '{g.user_id}'. New length: {len(notes)} chars.")
                    return jsonify({'message': 'Notes updated successfully', 'notes': notes})
                except Exception as e:
                    logger.error(f"Error updating notes for project '{project_id}': {e}", exc_info=True)
                    raise Exception(f"Failed to update notes: {e}")

        @self.app.route('/feedback/<project_id>', methods=['GET', 'POST'])
        @login_required
        @project_permission_required(permission_level='read_write_feedback') # Conceptual permission
        def handle_feedback(project_id: str):
            """
            Handles operations for project feedback.
            GET: Retrieves all feedback items for a given project.
            POST: Adds a new feedback item to a project.
            """
            logger.info(f"Received {request.method} request for feedback on project: {project_id}")

            if request.method == 'GET':
                try:
                    # Simulate PyTorch computation
                    dummy_tensor = torch.randn(2, 2, device=self.device)
                    _ = torch.exp(dummy_tensor)
                    
                    feedback_items = self.feedback_data[project_id]
                    logger.info(f"Fetched {len(feedback_items)} feedback items for project '{project_id}'.")
                    return jsonify({'feedback': feedback_items})
                except Exception as e:
                    logger.error(f"Error fetching feedback for project '{project_id}': {e}", exc_info=True)
                    raise Exception(f"Failed to retrieve feedback: {e}")

            elif request.method == 'POST':
                data = request.json
                if not data:
                    return jsonify({'message': 'Request body must be JSON'}), 400

                comment = data.get('comment')
                if not isinstance(comment, str) or not comment.strip():
                    return jsonify({'message': 'Invalid input: "comment" must be a non-empty string.'}), 400
                
                feedback_item = {
<<<<<<< HEAD
                    "user_id": data.get('user_id', 'anonymous'),
                    "comment": data.get('comment'),
                    "timestamp": time.time()
                }
                self.feedback_data[project_id].append(feedback_item)
                # Simulate some PyTorch computation
                dummy_tensor = torch.tensor([len(self.feedback_data[project_id])], dtype=torch.float32, device=self.device)
                _ = torch.sin(dummy_tensor) # Dummy computation
                print(f"Feedback added to project {project_id} by {feedback_item['user_id']}")
                return jsonify({'message': 'Feedback added successfully', 'feedback': feedback_item})

        @self.app.route('/projects/<project_id>/members', methods=['GET', 'POST', 'DELETE'])
        def manage_members(project_id):
            if request.method == 'GET':
                return jsonify({'members': list(self.project_members[project_id])})
            elif request.method == 'POST':
                data = request.json
                user_id = data.get('user_id')
                role = data.get('role', 'member')
                if user_id:
                    self.project_members[project_id].add(user_id)
                    self.project_roles[project_id][user_id] = role
                    print(f"User {user_id} added to project {project_id} with role {role}")
                    return jsonify({'message': 'Member added', 'user_id': user_id, 'role': role})
                return jsonify({'message': 'User ID required'}), 400
            elif request.method == 'DELETE':
                data = request.json
                user_id = data.get('user_id')
                if user_id and user_id in self.project_members[project_id]:
                    self.project_members[project_id].remove(user_id)
                    if user_id in self.project_roles[project_id]:
                        del self.project_roles[project_id][user_id]
                    print(f"User {user_id} removed from project {project_id}")
                    return jsonify({'message': 'Member removed', 'user_id': user_id})
                return jsonify({'message': 'User ID not found or required'}), 400
=======
                    "user_id": g.user_id, # Use the authenticated user ID
                    "comment": comment,
                    "timestamp": datetime.now().isoformat() # ISO format for better readability and parsing
                }
                self.feedback_data[project_id].append(feedback_item)
                
                try:
                    # Simulate PyTorch computation based on number of feedback items
                    dummy_tensor = torch.tensor([len(self.feedback_data[project_id])], dtype=torch.float32, device=self.device)
                    _ = torch.sin(dummy_tensor) # Dummy computation
                    
                    logger.info(f"Feedback added to project '{project_id}' by '{feedback_item['user_id']}'. Total items: {len(self.feedback_data[project_id])}.")
                    return jsonify({'message': 'Feedback added successfully', 'feedback': feedback_item}), 201 # 201 Created
                except Exception as e:
                    logger.error(f"Error adding feedback for project '{project_id}': {e}", exc_info=True)
                    raise Exception(f"Failed to add feedback: {e}")


        @self.app.route('/projects/<project_id>/members', methods=['GET', 'POST', 'DELETE'])
        @login_required
        @project_permission_required(permission_level='manage_members') # Conceptual permission for project admins
        def manage_members(project_id: str):
            """
            Handles operations for managing project members.
            GET: Retrieves the list of members for a project.
            POST: Adds a user to a project with a specified role.
            DELETE: Removes a user from a project.
            """
            logger.info(f"Received {request.method} request for members on project: {project_id}")

            if request.method == 'GET':
                members_with_roles = {
                    user_id: self.project_roles[project_id].get(user_id, 'member')
                    for user_id in self.project_members[project_id]
                }
                logger.info(f"Fetched {len(members_with_roles)} members for project '{project_id}'.")
                return jsonify({'members': members_with_roles})

            elif request.method == 'POST':
                data = request.json
                if not data:
                    return jsonify({'message': 'Request body must be JSON'}), 400

                user_id = data.get('user_id')
                role = data.get('role', 'member') # Default role
                
                if not isinstance(user_id, str) or not user_id.strip():
                    return jsonify({'message': 'Invalid input: "user_id" must be a non-empty string.'}), 400
                if not isinstance(role, str) or not role.strip():
                    return jsonify({'message': 'Invalid input: "role" must be a non-empty string.'}), 400

                self.project_members[project_id].add(user_id)
                self.project_roles[project_id][user_id] = role
                logger.info(f"User '{user_id}' added to project '{project_id}' with role '{role}' by '{g.user_id}'.")
                return jsonify({'message': 'Member added successfully', 'user_id': user_id, 'role': role}), 201

            elif request.method == 'DELETE':
                data = request.json
                if not data:
                    return jsonify({'message': 'Request body must be JSON'}), 400

                user_id = data.get('user_id')
                if not isinstance(user_id, str) or not user_id.strip():
                    return jsonify({'message': 'Invalid input: "user_id" must be a non-empty string.'}), 400

                if user_id not in self.project_members[project_id]:
                    logger.warning(f"Attempt to remove non-existent user '{user_id}' from project '{project_id}'.")
                    return jsonify({'message': f'User ID "{user_id}" not found in project "{project_id}".'}), 404
                
                self.project_members[project_id].remove(user_id)
                if user_id in self.project_roles[project_id]:
                    del self.project_roles[project_id][user_id]
                
                logger.info(f"User '{user_id}' removed from project '{project_id}' by '{g.user_id}'.")
                return jsonify({'message': 'Member removed successfully', 'user_id': user_id}), 200 # 200 OK

            # This line handles any other HTTP methods not explicitly defined for the route
            # The @app.errorhandler(405) will catch this if it's not GET/POST/DELETE
            return jsonify({'message': 'Method not allowed for this endpoint.'}), 405

>>>>>>> c9d035a (WIP: saving my local progress)

    def run(self):
        """
        Runs the Flask application.
        For development, this uses Flask's built-in server.
        For production, it is strongly recommended to use a WSGI server like Gunicorn or uWSGI.
        """
        logger.critical(f"\n{'-'*80}\n{' '*15}{Colors.BOLD}WARNING: Running in Development Mode. NOT FOR PRODUCTION.{Colors.RESET}\n{' '*15}Use a WSGI server (e.g., Gunicorn) for production deployments.\n{'-'*80}\n")
        
        # In a real production setup, you would typically not call app.run() directly here.
        # Instead, the WSGI server (e.g., Gunicorn) would import and run `self.app`.
        # Example Gunicorn command: gunicorn -w 4 'your_app_file:server.app' -b 0.0.0.0:5000
        self.app.run(host=self.host, port=self.port, debug=False) # debug should be False in production

<<<<<<< HEAD
if __name__ == '__main__':
    if torch.cuda.is_available():
        device = 'cuda'
        print("CUDA is available! Using GPU.")
    else:
        device = 'cpu'
        print("CUDA not available. Using CPU.")

    server = CollaborationServer(device=device)
    print("To run the server, uncomment `server.run()` and execute this file.")
    print("Example usage with curl:")
    print("\n--- Notes API ---")
    print("GET notes: curl http://127.0.0.1:5000/notes/project_alpha")
    print("POST notes: curl -X POST -H \"Content-Type: application/json\" -d '{\"user_id\": \"user1\", \"notes\": \"Initial research notes for project Alpha.\"}' http://127.0.0.1:5000/notes/project_alpha")
    print("\n--- Feedback API ---")
    print("GET feedback: curl http://127.0.0.1:5000/feedback/project_alpha")
    print("POST feedback: curl -X POST -H \"Content-Type: application/json\" -d '{\"user_id\": \"user2\", \"comment\": \"Great progress on the literature review!\"}' http://127.0.0.1:5000/feedback/project_alpha")
    print("\n--- Members API ---")
    print("GET members: curl http://127.0.0.1:5000/projects/project_alpha/members")
    print("POST member: curl -X POST -H \"Content-Type: application/json\" -d '{\"user_id\": \"user3\", \"role\": \"contributor\"}' http://127.0.0.1:5000/projects/project_alpha/members")
    print("DELETE member: curl -X DELETE -H \"Content-Type: application/json\" -d '{\"user_id\": \"user3\"}' http://127.0.0.1:5000/projects/project_alpha/members")
=======
# --- Main execution block for direct testing ---
if __name__ == '__main__':
    # Initialize the server
    selected_device = 'cpu'
    if torch.cuda.is_available():
        selected_device = 'cuda'

    server = CollaborationServer(device=selected_device)
    
    # --- Instructions for Running and Testing ---
    logger.info(f"\n{Colors.BRIGHT_MAGENTA}{Colors.BOLD}╔═════════════════════════════════════════════════════════════════════════╗{Colors.RESET}")
    logger.info(f"{Colors.BRIGHT_MAGENTA}{Colors.BOLD}║            🚀 Collaboration Server - Local Development Guide             ║{Colors.RESET}")
    logger.info(f"{Colors.BRIGHT_MAGENTA}{Colors.BOLD}╚═════════════════════════════════════════════════════════════════════════╝{Colors.RESET}\n")
    
    logger.info(f"{Colors.CYAN}To run the server locally, uncomment `server.run()` below and execute this file.{Colors.RESET}")
    logger.info(f"{Colors.YELLOW}For production, always use a WSGI server (e.g., Gunicorn) with HTTPS.{Colors.RESET}")
    
    logger.info(f"\n{Colors.BRIGHT_GREEN}Example usage with `curl` (simulating authentication with X-User-Id header):{Colors.RESET}")
>>>>>>> c9d035a (WIP: saving my local progress)

    logger.info(f"\n{Colors.BLUE}--- 1. Notes API ---{Colors.RESET}")
    logger.info(f"{Colors.GREEN}  POST notes (create/update):{Colors.RESET}")
    logger.info(f"    curl -X POST -H \"Content-Type: application/json\" -H \"X-User-Id: user1\" \\")
    logger.info(f"         -d '{{\"notes\": \"Initial research notes for Project Alpha. Focus on data acquisition.\"}}' \\")
    logger.info(f"         http://127.0.0.1:5000/notes/project_alpha")
    logger.info(f"{Colors.GREEN}  GET notes (retrieve):{Colors.RESET}")
    logger.info(f"    curl -H \"X-User-Id: user1\" http://127.0.0.1:5000/notes/project_alpha")

    logger.info(f"\n{Colors.BLUE}--- 2. Feedback API ---{Colors.RESET}")
    logger.info(f"{Colors.GREEN}  POST feedback (add):{Colors.RESET}")
    logger.info(f"    curl -X POST -H \"Content-Type: application/json\" -H \"X-User-Id: user2\" \\")
    logger.info(f"         -d '{{\"comment\": \"Great progress on the literature review! Let's discuss methodology.\"}}' \\")
    logger.info(f"         http://127.0.0.1:5000/feedback/project_alpha")
    logger.info(f"{Colors.GREEN}  GET feedback (retrieve):{Colors.RESET}")
    logger.info(f"    curl -H \"X-User-Id: user1\" http://127.0.0.1:5000/feedback/project_alpha")

    logger.info(f"\n{Colors.BLUE}--- 3. Members API ---{Colors.RESET}")
    logger.info(f"{Colors.GREEN}  POST member (add):{Colors.RESET}")
    logger.info(f"    curl -X POST -H \"Content-Type: application/json\" -H \"X-User-Id: admin_user\" \\")
    logger.info(f"         -d '{{\"user_id\": \"user3\", \"role\": \"contributor\"}}' \\")
    logger.info(f"         http://127.0.0.1:5000/projects/project_alpha/members")
    logger.info(f"{Colors.GREEN}  GET members (retrieve):{Colors.RESET}")
    logger.info(f"    curl -H \"X-User-Id: user1\" http://127.0.0.1:5000/projects/project_alpha/members")
    logger.info(f"{Colors.GREEN}  DELETE member (remove):{Colors.RESET}")
    logger.info(f"    curl -X DELETE -H \"Content-Type: application/json\" -H \"X-User-Id: admin_user\" \\")
    logger.info(f"         -d '{{\"user_id\": \"user3\"}}' \\")
    logger.info(f"         http://127.0.0.1:5000/projects/project_alpha/members")

    logger.info(f"\n{Colors.BRIGHT_MAGENTA}{Colors.BOLD}╔═════════════════════════════════════════════════════════════════════════╗{Colors.RESET}")
    logger.info(f"{Colors.BRIGHT_MAGENTA}{Colors.BOLD}║             ⭐ End of Collaboration Server Setup Instructions             ║{Colors.RESET}")
    logger.info(f"{Colors.BRIGHT_MAGENTA}{Colors.BOLD}╚═════════════════════════════════════════════════════════════════════════╝{Colors.RESET}")

    # --- UNCOMMENT THE LINE BELOW TO RUN THE FLASK SERVER ---
    # server.run() 
