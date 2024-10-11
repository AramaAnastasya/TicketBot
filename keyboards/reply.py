from aiogram.types import KeyboardButtonPollType, ReplyKeyboardMarkup, KeyboardButton, ReplyKeyboardRemove
from aiogram.utils.keyboard import ReplyKeyboardBuilder
from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup

start_kb = ReplyKeyboardMarkup(
    keyboard=[
        [
            KeyboardButton(text="Ответ по базе знаний")
        ],
        [
            KeyboardButton(text="Ответ Ollama")
        ]
    ],
   resize_keyboard=True,
    one_time_keyboard=True
    # input_field_placeholder='Что Вас интересует?'
)