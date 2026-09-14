import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from time import gmtime, strftime
from zoneinfo import ZoneInfo
import os
import io

class AlertManager:
    def __init__(self):
        
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass
        
        self._message = []
        self.token = os.getenv("TELE_TOKEN")
        self.chat_id = os.getenv("TELE_CHAT_ID")

    def send_chart_alert(self, s_message):
        url = f"https://api.telegram.org/bot{self.token}/sendMessage?chat_id={self.chat_id}&text={s_message}"
        return requests.get(url).json()
    
    def send_photo_alert(self, image_buffer: io.BytesIO,filename:     str = "sp.png", set_title = ""):
        image_buffer.seek(0)
        data  = {"chat_id": self.chat_id, "caption": set_title, "parse_mode": "HTML"}
        files = {"photo": (filename, image_buffer, "image/png")}        
        
        url = f"https://api.telegram.org/bot{self.token}/sendPhoto"
        resp = requests.post(url, data=data, files=files, timeout=20)
        resp.raise_for_status()
        result = resp.json()
        if result.get("ok"):
            print(f"[Telegram] ✓ Photo sent successfully " )        
        return 

    def get_message(self):
        return self._message
    
    def set_message(self, new_message):
        self._message = new_message


