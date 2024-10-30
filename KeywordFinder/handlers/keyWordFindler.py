import os
import json
import logging
# Для ембеддинга и модели
from KeywordFinder.handlers.llm import LLM

from aiogram import Bot, Dispatcher, Router, F, types
from aiogram.types import Message
from aiogram.client.bot import DefaultBotProperties
from aiogram.fsm.state import StatesGroup, State, default_state
from aiogram.enums import ParseMode
from keyboards import reply
from aiogram.fsm.context import FSMContext
from KeywordFinder.utils.states import FSMAdmin

import pandas as pd # если нет импорта пандаса, то остальной код не имеет смысла
from pandas import DataFrame

from langchain_community.document_loaders import DataFrameLoader 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

TOKEN = os.getenv('TOKEN')
bot = Bot(token=TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.HTML))
dp = Dispatcher()
knowledge_base_router = Router()

# пример документов
documents1 = [
  {"id": 1, "question": "Как восстановить пароль?", "answer": "Для восстановления пароля перейдите по ссылке 'Забыли пароль?' на странице входа. Введите свой адрес электронной почты, и мы вышлем вам инструкции по восстановлению пароля.", "url": "https://example.com/confluence/recover-password"},
]
documents2 = [
  {"id": 2, "question": "Как связаться со службой поддержки?", "answer": "Вы можете связаться со службой поддержки, написав нам на электронную почту support@example.com или позвонив по телефону +1 (123) 456-7890.", "url": "https://example.com/confluence/contact-support"},
  {"id": 3, "question": "Как настроить двухфакторную аутентификацию?", "answer": "Для настройки двухфакторной аутентификации перейдите в раздел 'Настройки безопасности' вашего аккаунта и следуйте инструкциям.", "url": "https://example.com/confluence/2fa-setup"},
]
all_documents = documents1 + documents2
# todo создать интерфейс для загрузки документов (парсер например)

class VectorStore():
    """ Векторная база данных """
    
    def __init__(self, embedding_model, path=None, name="VectorStore") -> None:
        self.embedding_model = embedding_model
        self.is_loaded = False
        self.name = name
        self.store = None 
        self.load_storage(path)

    def load_storage(self, path):
        if path is not None and os.path.exists(path):
            self.store = FAISS.load_local(self.name, self.embedding_model)
            self.is_loaded = True
        else:
            # Создаем пустое хранилище только при наличии документов
            self.store = None  # Инициализируем как None



    def add(self, docs: pd.DataFrame):
        loader = DataFrameLoader(docs, page_content_column='question')  # Индексация по вопросам
        documents = loader.load()
        
        # Делим документ
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_documents(documents)
        
        # Создаем хранилище, если оно не загружено
        if not self.is_loaded:
            self.store = FAISS.from_documents(texts, self.embedding_model)
        else:
            # Убедитесь, что `self.store` является объектом FAISS, который поддерживает добавление документов
            self.store.add_documents(texts)  # Теперь это должно работать, если self.store корректно инициализировано

        # Обновляем
        self.store.save_local(self.name)


# Далее ваш код


df = pd.DataFrame(all_documents)


model = LLM(model="llama3.2:3b", host="127.0.0.1", port=11434)
db = VectorStore(embedding_model=model.get_embeding_model())

# Добавляем документы в хранилище
db.add(df)


@knowledge_base_router.message(F.text.lower() == "ответ по базе знаний")
async def menu_cmd(message: types.Message, state:FSMAdmin):
    await message.answer("Задайте мне вопрос, и я найду релевантные документы.", reply_markup=reply.start_kb)
    await state.set_state(FSMAdmin.input)

@knowledge_base_router.message(FSMAdmin.input)
async def process_message(message: types.Message, state: FSMContext):
    query = message.text
    results = db.store.similarity_search_with_score(query, k=1)  # Поиск наиболее релевантного вопроса
    print("Search results:", results)
    
    if results:
        best_match = results[0]
        print("Best match:", best_match)
        document_id = best_match[0].metadata['id']  # Получаем ID из метаданных
        print("ID best match:", document_id)
        
        # Отладочная информация для проверки, какие документы у нас есть
        print("Documents in memory:", documents1 + documents2)

        answer = next((doc['answer'] for doc in documents1 + documents2 if doc['id'] == document_id), None)
        if answer is not None:
            await message.answer(answer)
        else:
            await message.answer("Извините, я не смог найти ответ на ваш вопрос.")
    else:
        await message.answer("Извините, я не смог найти ответ на ваш вопрос.")
