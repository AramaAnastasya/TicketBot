import os
import json
import logging
# Для ембеддинга и модели
from key_words_finder.handlers.llm import LLM

from aiogram import Bot, Dispatcher, Router, F, types
from aiogram.types import Message
from aiogram.client.bot import DefaultBotProperties
from aiogram.fsm.state import StatesGroup, State, default_state
from aiogram.enums import ParseMode
from keyboards import reply
from aiogram.fsm.context import FSMContext
from key_words_finder.utils.states import FSMAdmin

import pandas as pd # если нет импорта пандаса, то остальной код не имеет смысла
from pandas import DataFrame

from langchain_community.document_loaders import DataFrameLoader 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

import pytesseract

from PIL import Image
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Убедитесь, что путь корректный

# Настройка логирования
logging.basicConfig(level=logging.INFO)



TOKEN = os.getenv('TOKEN')
bot = Bot(token=TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.HTML))
dp = Dispatcher()
knowledge_base_router = Router()


# пример документов
documents1 = [
  {"id": 1,
        "question": "Как восстановить пароль?",
        "answer": "Для восстановления пароля перейдите по ссылке 'Забыли пароль?' на странице входа. Введите свой адрес электронной почты, и мы вышлем вам инструкции по восстановлению пароля. Если вы не получили письмо с инструкциями, проверьте папку со спамом или повторите запрос через несколько минут. Если у вас по-прежнему возникают проблемы, свяжитесь со службой поддержки по адресу support@company.com.",
        "url": "https://example.com/confluence/recover-password"}, 
  {"id": 2,
        "question": "Как настроить двухфакторную аутентификацию?",
        "answer": "Для настройки двухфакторной аутентификации перейдите в раздел 'Настройки безопасности' вашего аккаунта и следуйте инструкциям. Выберите метод двухфакторной аутентификации, который вам удобен, например, SMS или приложение для аутентификации. Введите код подтверждения, который вы получите на ваш телефон или в приложении. После успешной настройки двухфакторной аутентификации вам будет необходимо вводить код подтверждения каждый раз при входе в систему. Это значительно повышает безопасность вашего аккаунта.",
        "url": "https://example.com/confluence/2fa-setup"},
  {"id": 3,
        "question": "Как связаться с поддержкой?",
        "answer": "Контактные данные службы поддержки:\nТелефон: +7 (495) 123-45-67\nЭлектронная почта: support@company.com\nФорма обратной связи на сайте: https://company.com/support\n\nВремя работы службы поддержки:\nНаша служба поддержки работает круглосуточно, без выходных. Вы можете обратиться к нам в любое удобное для вас время.\n\nКак оставить заявку на техническую помощь:\nЧтобы оставить заявку на техническую помощь, выполните следующие действия:\n1. Перейдите на страницу технической поддержки на нашем сайте: https://company.com/support.\n2. Заполните форму заявки, указав необходимую информацию:\n   - Ваше имя и контактные данные.\n   - Описание проблемы.\n   - Желаемое время для связи.\n3. Нажмите кнопку 'Отправить'.\nМы постараемся решить вашу проблему как можно быстрее.",
        "url": "https://company.com/support"},
    {"id": 4,
        "question": "Как использовать корпоративный мессенджер?",
        "answer": "Основные функции мессенджера:\nНаш корпоративный мессенджер предоставляет следующие основные функции:\n- Обмен текстовыми сообщениями в режиме реального времени.\n- Создание групповых чатов для общения с коллегами.\n- Обмен файлами и документами.\n- Видеозвонки и аудиозвонки.\n- Интеграция с другими корпоративными сервисами.\n\nКак добавить контакт:\nЧтобы добавить контакт в корпоративный мессенджер, выполните следующие действия:\n1. Откройте приложение мессенджера.\n2. Нажмите на кнопку 'Контакты'.\n3. Нажмите на кнопку 'Добавить контакт'.\n4. Введите имя пользователя или адрес электронной почты контакта.\n5. Нажмите на кнопку 'Добавить'.\n\nКак создать групповой чат:\nЧтобы создать групповой чат, выполните следующие действия:\n1. Откройте приложение мессенджера.\n2. Нажмите на кнопку 'Создать чат'.\n3. Выберите 'Групповой чат'.\n4. Добавьте участников чата, выбрав их из списка контактов.\n5. Придумайте название чата.\n6. Нажмите на кнопку 'Создать'.",
        "url": "https://company.com/messenger"},
    {"id": 5,
        "question": "Как оформить командировку?",
        "answer": "Пошаговая инструкция по оформлению командировки:\nЧтобы оформить командировку, выполните следующие действия:\n1. Получите согласование командировки у вашего руководителя.\n2. Заполните заявление на командировку.\n3. Предоставьте необходимые документы в отдел кадров.\n4. Получите командировочное удостоверение и другие необходимые документы.\n5. Оформите билеты и бронирование проживания.\n6. Предоставьте отчет о командировке по возвращении.\n\nНеобходимые документы для командировки:\nДля оформления командировки вам потребуются следующие документы:\n- Заявление на командировку.\n- Командировочное удостоверение.\n- Билеты на транспорт.\n- Документы, подтверждающие бронирование проживания.\n- Другие документы, предусмотренные внутренними правилами компании.\n\nКонтактные данные отдела кадров:\nДля связи с отделом кадров вы можете использовать следующие контактные данные:\nТелефон: +7 (495) 123-45-68\nЭлектронная почта: hr@company.com\nАдрес: Москва, ул. Ленина, д. 1, офис 101",
        "url": "https://company.com/hr"}
]

all_documents = documents1
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


@knowledge_base_router.message(FSMAdmin.input)
async def process_message(message: types.Message, state: FSMContext):
    if message.content_type == 'photo':
        # Если сообщение содержит изображение, получаем фото с наибольшим разрешением
        photo = message.photo[-1]
        
        # Получаем объект файла
        file = await bot.get_file(photo.file_id)
        
        # Определяем путь для сохранения фото
        photo_path = f"{file.file_id}.jpg"
        
        # Скачиваем файл в локальную систему
        await bot.download_file(file.file_path, destination=photo_path)
        
        # Открываем изображение и извлекаем текст
        img = Image.open(photo_path)
        text = pytesseract.image_to_string(img, lang='rus')
        

        if not text.strip():  # Проверка на пустой текст
            await message.reply("На фото плохо видно текст")
            return 
        

        query = text
        await message.reply(query)
    else:
        # Если это текстовое сообщение
        query = message.text

    # Далее ваш существующий код для поиска по базе данных
    results = db.store.similarity_search_with_score(query, k=1)
    if results:
        best_match = results[0]
        document_id = best_match[0].metadata['id']
        answer = next((doc['answer'] for doc in documents1 if doc['id'] == document_id), None)
        if answer is not None:
            await message.answer(answer)
        else:
            await message.answer("Извините, я не смог найти ответ на ваш вопрос.")
    else:
        await message.answer("Извините, я не смог найти ответ на ваш вопрос.")

