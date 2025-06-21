import os
import sys
from datetime import datetime

def get_cyrrent_time():
    """Returns the current time in the format YYYY-MM-DD HH:MM:SS."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")