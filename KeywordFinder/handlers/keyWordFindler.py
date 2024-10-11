import os
import json
import logging

from aiogram import Bot, Dispatcher, Router, F, types
from aiogram.types import Message
from aiogram.client.bot import DefaultBotProperties
from aiogram.enums import ParseMode
from keyboards import reply
from aiogram.fsm.context import FSMContext

from KeywordFinder.utils.states import FSMAdmin

from dotenv import load_dotenv
load_dotenv()

knowledge_base_router = Router()

# Настройка логирования
logging.basicConfig(level=logging.INFO)

# Стоп-слова
stop_words = {"как", "в", "и", "на", "по", "для", "с", "от", "до", "за", "из", "у", "о", "об", "под", "над", "при", "к", "а", "но", "или", "что", "кто", "где", "когда", "почему", "как", "чем", "ли", "же", "бы", "не", "ни", "то", "так", "вот", "да", "нет", "если", "чтобы", "как"}

TOKEN = os.getenv('TOKEN')
bot = Bot(token=TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.HTML))
dp = Dispatcher()

@knowledge_base_router.message(F.text.lower() == "ответ по базе знаний")
async def menu_cmd(message: types.Message):
    await message.answer("Задайте мне вопрос, и я найду релевантные документы.", reply_markup=reply.start_kb)

# База знаний
knowledge_base = {
  "documents": [
    {
      "id": 1,
      "title": "Регистрация в приложении",
      "content": "Для регистрации в приложении необходимо перейти на главную страницу, нажать кнопку «Регистрация» и заполнить все обязательные поля. После заполнения формы вам придет письмо с подтверждением на указанный email адрес. Перейдите по ссылке в письме, чтобы завершить регистрацию."
    },
    {
      "id": 2,
      "title": "Связь с поддержкой",
      "content": "Вы можете связаться с поддержкой через форму обратной связи на сайте или по электронной почте support@example.com. Мы ответим на ваш запрос в течение 24 часов. Также вы можете найти ответы на часто задаваемые вопросы в разделе «Помощь» на нашем сайте."
    },
    {
      "id": 3,
      "title": "Способы оплаты",
      "content": "Мы принимаем оплату банковскими картами Visa и MasterCard, а также через платежные системы PayPal и Яндекс.Деньги. Для оплаты выберите нужный способ на странице оформления заказа и следуйте инструкциям на экране."
    }
  ]
}




# Обработка текстовых сообщений
@knowledge_base_router.message(lambda message: FSMAdmin.input)
async def process_message(message: Message, state: FSMContext):
    user_query = message.text.lower()
    keywords = [word for word in user_query.split() if word not in stop_words]

    relevant_documents = []

    for doc in knowledge_base['documents']:
        title_matches = sum(1 for keyword in keywords if keyword in doc['title'].lower())
        content_matches = sum(1 for keyword in keywords if keyword in doc['content'].lower())
        total_matches = title_matches + content_matches

        if total_matches > 0:
            relevant_documents.append((doc, total_matches))

    if relevant_documents:
        relevant_documents.sort(key=lambda x: x[1], reverse=True)
        response = "Вот что я нашел по вашему запросу:\n\n"
        for doc, _ in relevant_documents[:3]:  # Ограничим количество ответов тремя наиболее релевантными
            response += f"{doc['title']}:\n{doc['content']}\n\n"
    else:
        response = "К сожалению, я не нашел ответа в базе знаний."

    await message.answer(response)
