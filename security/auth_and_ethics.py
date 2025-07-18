
import torch
from pathlib import Path
import hashlib
import os
from datetime import datetime

class AuthAndEthics:
    """
    Handles user authentication, access control, and ethical review flagging.
    Enhanced with more robust role management and detailed ethical flagging.
    """

    def __init__(self, device: str = 'cpu'):
        """
        Initializes the AuthAndEthics module.
        Args:
            device (str): The device to use for computations ('cuda' or 'cpu').
        """
        self.device = torch.device(device)
        self.users = {}
        self.roles = {
            "admin": {"permissions": ["create_user", "manage_roles", "flag_ethics", "review_ethics", "access_all_data"]},
            "researcher": {"permissions": ["submit_research", "access_data", "collaborate"]},
            "student": {"permissions": ["submit_research", "collaborate"]},
            "guest": {"permissions": ["view_public_content"]}
        }
        self.ethical_flags = []
        print(f"AuthAndEthics initialized on device: {self.device}")

    def _hash_password(self, password: str) -> str:
        """
        Hashes a password using SHA256.
        Args:
            password (str): The plain-text password.
        Returns:
            str: The hashed password.
        """
        return hashlib.sha256(password.encode()).hexdigest()

    def register_user(self, username: str, password: str, role: str = "student") -> bool:
        """
        Registers a new user.
        Args:
            username (str): The username.
            password (str): The password.
            role (str): The role of the user (e.g., "student", "researcher", "admin").
        Returns:
            bool: True if registration is successful, False otherwise.
        """
        if username in self.users:
            print(f"User {username} already exists.")
            return False
        if role not in self.roles:
            print(f"Invalid role: {role}")
            return False

        self.users[username] = {"password_hash": self._hash_password(password), "role": role, "created_at": datetime.now().isoformat()}
        # Simulate some PyTorch computation
        dummy_tensor = torch.tensor([len(self.users)], dtype=torch.float32, device=self.device)
        _ = torch.sigmoid(dummy_tensor) # Dummy computation
        print(f"User {username} registered with role {role}.")
        return True

    def authenticate_user(self, username: str, password: str) -> bool:
        """
        Authenticates a user.
        Args:
            username (str): The username.
            password (str): The plain-text password.
        Returns:
            bool: True if authentication is successful, False otherwise.
        """
        user_data = self.users.get(username)
        if not user_data:
            print(f"User {username} not found.")
            return False

        if user_data["password_hash"] == self._hash_password(password):
            # Simulate some PyTorch computation
            dummy_tensor = torch.tensor([1.0], dtype=torch.float32, device=self.device)
            _ = torch.tanh(dummy_tensor) # Dummy computation
            print(f"User {username} authenticated.")
            return True
        else:
            print(f"Incorrect password for user {username}.")
            return False

    def has_permission(self, username: str, action: str) -> bool:
        """
        Checks if a user has permission for a specific action.
        Args:
            username (str): The username.
            action (str): The action to check permission for.
        Returns:
            bool: True if the user has permission, False otherwise.
        """
        user_data = self.users.get(username)
        if not user_data:
            return False
        
        user_role = user_data["role"]
        # Simulate some PyTorch computation
        dummy_tensor = torch.tensor([len(action)], dtype=torch.float32, device=self.device)
        _ = torch.relu(dummy_tensor) # Dummy computation
        return action in self.roles.get(user_role, {}).get("permissions", [])

    def flag_ethical_concern(self, description: str, research_id: str = None, flagged_by: str = "system") -> None:
        """
        Flags an ethical concern with more details.
        Args:
            description (str): Description of the ethical concern.
            research_id (str): Optional ID of the research project related to the concern.
            flagged_by (str): The user or system component that flagged the concern.
        """
        concern = {
            "description": description,
            "research_id": research_id,
            "status": "pending",
            "flagged_by": flagged_by,
            "timestamp": datetime.now().isoformat()
        }
        self.ethical_flags.append(concern)
        # Simulate some PyTorch computation
        dummy_tensor = torch.tensor([len(self.ethical_flags)], dtype=torch.float32, device=self.device)
        _ = torch.log1p(dummy_tensor) # Dummy computation
        print(f"Ethical concern flagged: {description}")

    def review_ethical_concern(self, concern_index: int, new_status: str, reviewed_by: str) -> bool:
        """
        Reviews and updates the status of an ethical concern.
        Args:
            concern_index (int): The index of the concern in the ethical_flags list.
            new_status (str): The new status (e.g., "resolved", "dismissed", "under_review").
            reviewed_by (str): The user who reviewed the concern.
        Returns:
            bool: True if the concern was updated, False otherwise.
        """
        if 0 <= concern_index < len(self.ethical_flags):
            self.ethical_flags[concern_index]["status"] = new_status
            self.ethical_flags[concern_index]["reviewed_by"] = reviewed_by
            self.ethical_flags[concern_index]["reviewed_at"] = datetime.now().isoformat()
            # Simulate some PyTorch computation
            dummy_tensor = torch.tensor([concern_index], dtype=torch.float32, device=self.device)
            _ = torch.exp(dummy_tensor) # Dummy computation
            print(f"Ethical concern {concern_index} updated to {new_status} by {reviewed_by}.")
            return True
        return False

    def get_ethical_flags(self, status: str = None) -> list:
        """
        Retrieves ethical flags, optionally filtered by status.
        Returns:
            list: A list of ethical concerns.
        """
        # Simulate some PyTorch computation
        dummy_tensor = torch.tensor([len(self.ethical_flags)], dtype=torch.float32, device=self.device)
        _ = torch.sqrt(dummy_tensor) # Dummy computation
        if status:
            return [flag for flag in self.ethical_flags if flag["status"] == status]
        return self.ethical_flags

# --- ANSI Escape Codes for Colors and Styles ---
# These codes work in most modern terminals (like PowerShell, VS Code terminal, Linux terminals)
class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    ITALIC = "\033[3m"
    UNDERLINE = "\033[4m"
    BLINK = "\033[5m"
    INVERT = "\033[7m"
    HIDDEN = "\033[8m"
    STRIKETHROUGH = "\033[9m"

    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"

    BRIGHT_BLACK = "\033[90m"
    BRIGHT_RED = "\033[91m"
    BRIGHT_GREEN = "\033[92m"
    BRIGHT_YELLOW = "\033[93m"
    BRIGHT_BLUE = "\033[94m"
    BRIGHT_MAGENTA = "\033[95m"
    BRIGHT_CYAN = "\033[96m"
    BRIGHT_WHITE = "\033[97m"

    BG_BLACK = "\033[40m"
    BG_RED = "\033[41m"
    BG_GREEN = "\033[42m"
    BG_YELLOW = "\033[43m"
    BG_BLUE = "\033[44m"
    BG_MAGENTA = "\033[45m"
    BG_CYAN = "\033[46m"
    BG_WHITE = "\033[47m"

    BG_BRIGHT_BLACK = "\033[100m"
    BG_BRIGHT_RED = "\033[101m"
    BG_BRIGHT_GREEN = "\033[102m"
    BG_BRIGHT_YELLOW = "\033[103m"
    BG_BRIGHT_BLUE = "\033[104m"
    BG_BRIGHT_MAGENTA = "\033[105m"
    BG_BRIGHT_CYAN = "\033[106m"
    BG_BRIGHT_WHITE = "\033[107m"

# Your existing AuthAndEthics class (or relevant parts for context)
class AuthAndEthics:
    def __init__(self, device='cpu'):
        self.device = device
        self.users = {}
        self.ethical_flags = []
        # Define permissions for each role
        self.permissions = {
            "admin": ["manage_roles", "flag_ethics", "review_ethics", "submit_research"],
            "researcher": ["submit_research", "flag_ethics"],
            "student": ["view_research"]
        }
        print(f"{Colors.BLUE}AuthAndEthics system initialized on {self.device}.{Colors.RESET}")


    def register_user(self, username, password, role):
        if username in self.users:
            print(f"{Colors.YELLOW}User '{username}' already exists.{Colors.RESET}")
            return False
        if role not in self.permissions:
            print(f"{Colors.RED}Invalid role '{role}'.{Colors.RESET}")
            return False
        self.users[username] = {"password": password, "role": role}
        print(f"{Colors.GREEN}User '{username}' registered with role '{role}'.{Colors.RESET}")
        return True

    def authenticate_user(self, username, password):
        user_info = self.users.get(username)
        if user_info and user_info["password"] == password:
            print(f"{Colors.GREEN}{Colors.BOLD}Authentication successful for '{username}' ({user_info['role']}).{Colors.RESET}")
            return True
        print(f"{Colors.RED}{Colors.BOLD}Authentication failed for '{username}'.{Colors.RESET}")
        return False

    def is_authorized(self, username: str) -> bool:
        """
        Checks if a user is authorized to perform a high-level action like running training.
        This simplified version checks if the user exists and has a role other than 'guest' or 'student'.
        You can customize this logic based on your authorization rules.
        Args:
            username (str): The username to check.
        Returns:
            bool: True if authorized, False otherwise.
        """
        user_data = self.users.get(username)
        if not user_data:
            print(f"{Colors.RED}Authorization failed: User '{username}' not found.{Colors.RESET}")
            return False

        # Example: Only 'admin' and 'researcher' roles are authorized to run training
        authorized_roles = ["admin", "researcher"]
        if user_data["role"] in authorized_roles:
            print(f"{Colors.GREEN}Authorization granted for user '{username}' (Role: {user_data['role']}).{Colors.RESET}")
            return True
        else:
            print(f"{Colors.YELLOW}Authorization denied for user '{username}' (Role: {user_data['role']}). Insufficient privileges.{Colors.RESET}")
            return False

    def has_permission(self, username, permission):
        user_info = self.users.get(username)
        if not user_info:
            print(f"{Colors.RED}User '{username}' not found.{Colors.RESET}")
            return False
        
        user_role = user_info["role"]
        if permission in self.permissions.get(user_role, []):
            print(f"{Colors.CYAN}User '{username}' ({user_role}) {Colors.BOLD}HAS{Colors.RESET}{Colors.CYAN} permission: {permission}.{Colors.RESET}")
            return True
        print(f"{Colors.YELLOW}User '{username}' ({user_role}) {Colors.RED}DOES NOT HAVE{Colors.RESET}{Colors.YELLOW} permission: {permission}.{Colors.RESET}")
        return False

    def flag_ethical_concern(self, concern_description, project_id=None, flagged_by="unknown"):
        concern = {
            "id": len(self.ethical_flags),
            "description": concern_description,
            "project_id": project_id,
            "flagged_by": flagged_by,
            "status": "pending" # pending, resolved, dismissed
        }
        self.ethical_flags.append(concern)
        print(f"{Colors.MAGENTA}Ethical concern flagged (ID: {concern['id']}): {concern_description}{Colors.RESET}")
        return concern['id']

    def get_ethical_flags(self, status=None):
        if status:
            return [flag for flag in self.ethical_flags if flag["status"] == status]
        return self.ethical_flags

    def review_ethical_concern(self, concern_id, new_status, reviewed_by):
        if not (0 <= concern_id < len(self.ethical_flags)):
            print(f"{Colors.RED}Invalid concern ID.{Colors.RESET}")
            return False
        
        concern = self.ethical_flags[concern_id]
        if new_status not in ["pending", "resolved", "dismissed"]:
            print(f"{Colors.YELLOW}Invalid status '{new_status}'.{Colors.RESET}")
            return False

        concern["status"] = new_status
        concern["reviewed_by"] = reviewed_by
        print(f"{Colors.BRIGHT_GREEN}Ethical concern {concern_id} updated to '{new_status}' by {reviewed_by}.{Colors.RESET}")
        return True


# --- Main Execution Block ---
if __name__ == "__main__":
    # Ensure correct f-string syntax in print statements
    # Example Usage
    if torch.cuda.is_available():
        device = 'cuda'
        print(f"{Colors.GREEN}{Colors.BOLD}✨ CUDA is available! Using GPU. ✨{Colors.RESET}")
    else:
        device = 'cpu'
        print(f"{Colors.YELLOW}{Colors.BOLD}⚠️ CUDA not available. Using CPU. ⚠️{Colors.RESET}")

    auth_ethics = AuthAndEthics(device=device)

    # Register users
    print(f"\n{Colors.BLUE}--- User Registration ---{Colors.RESET}")
    auth_ethics.register_user("admin_user", "admin_pass", "admin")
    auth_ethics.register_user("researcher1", "research_pass", "researcher")
    auth_ethics.register_user("student1", "student_pass", "student")
    auth_ethics.register_user("admin_user", "admin_pass", "admin") # Attempt to re-register

    # Authenticate users
    print(f"\n{Colors.BLUE}--- User Authentication ---{Colors.RESET}")
    print(f'Admin login: {auth_ethics.authenticate_user("admin_user", "admin_pass")}')
    print(f'Researcher login (wrong pass): {auth_ethics.authenticate_user("researcher1", "wrong_pass")}')
    print(f'Student login: {auth_ethics.authenticate_user("student1", "student_pass")}')
    print(f'Non-existent user login: {auth_ethics.authenticate_user("ghost_user", "1234")}')

    # Check permissions
    print(f"\n{Colors.BLUE}--- Permission Checks ---{Colors.RESET}")
    print(f'Admin can flag ethics: {auth_ethics.has_permission("admin_user", "flag_ethics")}')
    print(f'Researcher can submit research: {auth_ethics.has_permission("researcher1", "submit_research")}')
    print(f'Student can manage roles: {auth_ethics.has_permission("student1", "manage_roles")}')
    print(f'Admin can review ethics: {auth_ethics.has_permission("admin_user", "review_ethics")}')
    print(f'Researcher can review ethics: {auth_ethics.has_permission("researcher1", "review_ethics")}') # Expected False
    print(f'Student can view research: {auth_ethics.has_permission("student1", "view_research")}')

    # Flag ethical concerns
    print(f"\n{Colors.BLUE}--- Ethical Concerns System ---{Colors.RESET}")
    concern_id_1 = auth_ethics.flag_ethical_concern("Data collection method might violate privacy.", "proj_001", "researcher1")
    concern_id_2 = auth_ethics.flag_ethical_concern("Potential bias in AI model training data.", flagged_by="system")
    concern_id_3 = auth_ethics.flag_ethical_concern("AI model's outputs are racially biased.", "report_AI_bias", "admin_user")

    # Get ethical flags
    print(f"\n{Colors.BLUE}--- All Ethical Flags Log ---{Colors.RESET}")
    for flag in auth_ethics.get_ethical_flags():
        status_color = Colors.YELLOW if flag['status'] == 'pending' else (Colors.GREEN if flag['status'] == 'resolved' else Colors.RED)
        print(f"  [{Colors.BOLD}ID:{flag['id']}{Colors.RESET}] {flag['description']} {Colors.DIM}(Project: {flag['project_id']}, Flagged by: {flag['flagged_by']}){Colors.RESET} {status_color}[{flag['status'].upper()}]{Colors.RESET}")

    # Review an ethical concern
    print(f"\n{Colors.BLUE}--- Reviewing Ethical Concern ---{Colors.RESET}")
    if auth_ethics.review_ethical_concern(concern_id_1, "resolved", "admin_user"):
        print(f"{Colors.GREEN}Successfully updated concern {concern_id_1}.{Colors.RESET}")
    if auth_ethics.review_ethical_concern(999, "resolved", "admin_user"): # Invalid ID
        pass
    if auth_ethics.review_ethical_concern(concern_id_2, "invalid_status", "admin_user"): # Invalid Status
        pass

    print(f"\n{Colors.BLUE}--- Updated Ethical Flags Log ---{Colors.RESET}")
    for flag in auth_ethics.get_ethical_flags():
        status_color = Colors.YELLOW if flag['status'] == 'pending' else (Colors.GREEN if flag['status'] == 'resolved' else Colors.RED)
        print(f"  [{Colors.BOLD}ID:{flag['id']}{Colors.RESET}] {flag['description']} {Colors.DIM}(Project: {flag['project_id']}, Flagged by: {flag['flagged_by']}){Colors.RESET} {status_color}[{flag['status'].upper()}]{Colors.RESET}")

    print(f"\n{Colors.BLUE}--- Pending Ethical Flags (Filtered) ---{Colors.RESET}")
    pending_flags = auth_ethics.get_ethical_flags(status="pending")
    if pending_flags:
        for flag in pending_flags:
            print(f"  [{Colors.BOLD}ID:{flag['id']}{Colors.RESET}] {Colors.YELLOW}{flag['description']}{Colors.RESET} {Colors.DIM}(Project: {flag['project_id']}, Flagged by: {flag['flagged_by']}){Colors.RESET}")
    else:
        print(f"{Colors.GREEN}No pending ethical flags found.{Colors.RESET}")

    print(f"\n{Colors.BLUE}--- System Shutdown ---{Colors.RESET}")
    print(f"{Colors.BRIGHT_WHITE}Thank you for using the NeuroResearcher Auth & Ethics System. Session terminated.{Colors.RESET}")

