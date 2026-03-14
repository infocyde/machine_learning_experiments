import os

def clear_console():
    """Clears the console based on the operating system."""
    os.system('cls' if os.name == 'nt' else 'clear')    