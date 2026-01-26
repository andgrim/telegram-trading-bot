import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import json
import os

class DataManager:
    """Manage data storage and retrieval"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
    
    def save_watchlist(self, user_id: int, watchlist: List[str]):
        """Save user's watchlist"""
        filepath = os.path.join(self.data_dir, f"watchlist_{user_id}.json")
        with open(filepath, 'w') as f:
            json.dump({'watchlist': watchlist}, f)
    
    def load_watchlist(self, user_id: int) -> List[str]:
        """Load user's watchlist"""
        filepath = os.path.join(self.data_dir, f"watchlist_{user_id}.json")
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                data = json.load(f)
                return data.get('watchlist', [])
        return []
    
    def save_alerts(self, user_id: int, alerts: List[Dict]):
        """Save user's price alerts"""
        filepath = os.path.join(self.data_dir, f"alerts_{user_id}.json")
        with open(filepath, 'w') as f:
            json.dump({'alerts': alerts}, f)
    
    def load_alerts(self, user_id: int) -> List[Dict]:
        """Load user's price alerts"""
        filepath = os.path.join(self.data_dir, f"alerts_{user_id}.json")
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                data = json.load(f)
                return data.get('alerts', [])
        return []