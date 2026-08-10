import os
from google import genai
from dotenv import load_dotenv


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
        self.model_name = "gemini-3.5-flash-lite"

    def generate(self, prompt: str) -> str:
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
        )
        return response.text