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

# todo создать интерфейс для загрузки документов (парсер например)

class VectorStore():
  """ Векторная база данных """
  
  def __init__(self, embedding_model, path=None, name="VectorStore") -> None:
    self.embedding_model = embedding_model
    self.is_loaded = False
    self.name = name
    self.load_storage(path)

  def load_storage(self, path):
      if path is not None and os.path.exists(path):
        self.store = FAISS.load_local(self.name, self.embedding_model)
        self.is_loaded = True
    
  def add(self, docs: DataFrame, index_col: str):
    loader = DataFrameLoader(docs, page_content_column=index_col)
    documents = loader.load()
    
    # Делим документ
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    
    embeddings = self.embedding_model # задаем векторайзер

    # Создаем хранилище, если оно не загружено
    if not self.is_loaded:
      self.store = FAISS.from_documents(texts, embeddings)
    else:
      self.store.add_documents(texts)

    # Обновляем
    self.store.save_local(self.name)
    # Подготовка к запросам
    self.store.as_retriever()

  def as_retriever(self):
    return self.store.as_retriever()

df1 = pd.DataFrame(documents1)
df2 = pd.DataFrame(documents2)
  
model = LLM(model="llama3.2:3b", host="127.0.0.1", port=11434)
db = VectorStore(embedding_model=model.get_embeding_model())
  
# # Добавление 1 части вопросов и поиск
db.add(docs=df1, index_col="answer")
data = db.store.similarity_search_with_score('не знаю как прикрепить сотрудника')
print(data)
  
# # Добавление 2 части вопросов и поиск
db.add(docs=df2, index_col="answer")
# тестируем ретривер k=1 самый вероятный
data = db.store.similarity_search_with_score('не знаю как прикрепить сотрудника')
print(data)

@knowledge_base_router.message(F.text.lower() == "ответ по базе знаний")
async def menu_cmd(message: types.Message, state:FSMAdmin):
    await message.answer("Задайте мне вопрос, и я найду релевантные документы.", reply_markup=reply.start_kb)
    await state.set_state(FSMAdmin.input)


  # Обработка запросов пользователя
@knowledge_base_router.message(FSMAdmin.input)
async def process_message(message: Message, state: FSMContext):
    query = message.text
    results = db.store.similarity_search_with_score(query, k=1)
    print(results)
    if results:
        best_match = results[0]
        answer = best_match[0].page_content
        await message.answer(answer)
    else:
        await message.answer("Извините, я не смог найти ответ на ваш вопрос.")




# from dotenv import load_dotenv
# load_dotenv()

# # Настройка логирования
# logging.basicConfig(level=logging.INFO)

# # Стоп-слова
# stop_words = {"как", "в", "и", "на", "по", "для", "с", "от", "до", "за", "из", "у", "о", "об", "под", "над", "при", "к", "а", "но", "или", "что", "кто", "где", "когда", "почему", "как", "чем", "ли", "же", "бы", "не", "ни", "то", "так", "вот", "да", "нет", "если", "чтобы", "как"}





# # База знаний
# knowledge_base = {
#   "documents": [
#     {
#       "id": 1,
#       "title": "Регистрация в приложении",
#       "content": "Для регистрации в приложении необходимо перейти на главную страницу, нажать кнопку «Регистрация» и заполнить все обязательные поля. После заполнения формы вам придет письмо с подтверждением на указанный email адрес. Перейдите по ссылке в письме, чтобы завершить регистрацию."
#     },
#     {
#       "id": 2,
#       "title": "Связь с поддержкой",
#       "content": "Вы можете связаться с поддержкой через форму обратной связи на сайте или по электронной почте support@example.com. Мы ответим на ваш запрос в течение 24 часов. Также вы можете найти ответы на часто задаваемые вопросы в разделе «Помощь» на нашем сайте."
#     },
#     {
#       "id": 3,
#       "title": "Способы оплаты",
#       "content": "Мы принимаем оплату банковскими картами Visa и MasterCard, а также через платежные системы PayPal и Яндекс.Деньги. Для оплаты выберите нужный способ на странице оформления заказа и следуйте инструкциям на экране."
#     }
#   ]
# }




# # Обработка текстовых сообщений
# @knowledge_base_router.message(FSMAdmin.input)
# async def process_message(message: Message, state: FSMContext):
#     user_query = message.text.lower()
#     keywords = [word for word in user_query.split() if word not in stop_words]

#     relevant_documents = []

#     for doc in knowledge_base['documents']:
#         title_matches = sum(1 for keyword in keywords if keyword in doc['title'].lower())
#         content_matches = sum(1 for keyword in keywords if keyword in doc['content'].lower())
#         total_matches = title_matches + content_matches

#         if total_matches > 0:
#             relevant_documents.append((doc, total_matches))

#     if relevant_documents:
#         relevant_documents.sort(key=lambda x: x[1], reverse=True)
#         response = "Вот что я нашел по вашему запросу:\n\n"
#         for doc, _ in relevant_documents[:3]:  # Ограничим количество ответов тремя наиболее релевантными
#             response += f"{doc['title']}:\n{doc['content']}\n\n"
#     else:
#         response = "К сожалению, я не нашел ответа в базе знаний."

#     await message.answer(response)
