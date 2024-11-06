from typing import List, Dict
from langchain_ollama import OllamaEmbeddings
import requests
import json


class LLM():
    """ Класс-обертка для запуска локальных моделей """
    # Модели: llama3.2:3b или llama3.2:1b и mxbai-embed-large
    # todo добавить бд и индексацию (возможно)
    # Добавлять изображения можно добавив в json поле "images": List[Base64]
    
    def __init__(self, host: str, port: int or str, model: str, stream: bool=False) -> None:
        self.model = model
        self.host = host
        self.port = port
        self.stream = stream
    
    def generate(self, promt: str, images: List[str], **params):
        """ Ответ одним сообщением (ответ) """
        json_data={"model": self.model, "prompt": promt, "stream": self.stream, "images": images, **params}
        data = self.send(json_data, path="generate")
        return json.loads(data.text)["response"]
    
    def chat(self, messages: List[Dict], images: List[str], **params):
        """ Ответ на основе множества сообщений (чат) 
        - messages: пример сообщения [{"role": "user", "content": "question"}]
        """
        json_data={"model": self.model, "messages": messages, "stream": self.stream, "images": images, **params}
        data = self.send(json_data, path="chat")
        return json.loads(data.text)["message"]["content"]
    
    def get_embeding_model(self, model: str="mxbai-embed-large"):
        """ Ембеддинги текстов """
        return OllamaEmbeddings(model=model, base_url=f"http://{self.host}:{self.port}")
    
    def send(self, json, path):
        """ Метод отправки данных в модель """
        return requests.post(f"http://{self.host}:{self.port}/api/{path}", json=json)