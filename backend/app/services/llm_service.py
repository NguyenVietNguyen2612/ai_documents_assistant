import os
import time
from functools import wraps
from google import genai
from dotenv import load_dotenv

def rate_limit_retry(max_retries=5, initial_delay=10):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            delay = initial_delay
            for i in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    error_msg = str(e)
                    if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg or "quota" in error_msg.lower():
                        print(f"[LLMService] Rate limit (429) hit. Retrying in {delay} seconds... ({i+1}/{max_retries})")
                        time.sleep(delay)
                        delay *= 2  # Exponential backoff
                    else:
                        raise e
            return func(*args, **kwargs)
        return wrapper
    return decorator

class LLMService:
    def __init__(self, api_key: str = None):
        # Tải các biến môi trường từ file .env
        load_dotenv()
        
        # Lấy API key từ tham số truyền vào hoặc từ biến môi trường
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("Vui lòng cung cấp API key hoặc đặt biến GEMINI_API_KEY trong file .env")
            
        # Khởi tạo client với SDK mới
        self.client = genai.Client(api_key=self.api_key)
        
        # Đặt tên model (bạn có thể thay đổi thành gemini-2.5-flash hoặc version mới hơn)
        self.model_name = "gemini-3.1-flash-lite"

    @rate_limit_retry()
    def generate(self, prompt: str) -> str:
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
        )
        return response.text

    @rate_limit_retry()
    def generate_structured(
        self,
        prompt: str,
        response_model,
    ):

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "response_json_schema": (
                    response_model.model_json_schema()
                ),
            },
        )

        return response_model.model_validate_json(
            response.text
        )