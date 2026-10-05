from datetime import datetime

def log(message: str):

    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print(f"[{current_time}] {message}")