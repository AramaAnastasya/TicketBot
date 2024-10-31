import asyncio
import os
from aiogram import Router, types, F
from aiogram.filters import Command
from aiogram import Bot, Dispatcher
from aiogram.client.bot import DefaultBotProperties
from aiogram.enums import ParseMode
from keyboards import reply, inline
TOKEN = os.getenv('TOKEN')

if not TOKEN:
    raise ValueError("Токен бота не найден. Убедитесь, что он задан в файле .env")
# Создаем экземпляр маршрутизатора
router = Router()
bot = Bot(token=TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.HTML))
dp = Dispatcher()
# Обработчик команды /start
@router.message(Command("start"))
async def cmd_start(message: types.Message):
    chat_member = await bot.get_chat_member(message.chat.id, message.from_user.id)
    # Вывести имя пользователя
    await bot.send_message(message.chat.id, 
            f"Добрый день, <b>{chat_member.user.first_name}</b>! Я виртуальный помощник для ответов на ваши вопросы.\nЧто ваша проблема?",
            reply_markup=reply.start_kb        
            )

@router.message(F.text.lower() == "кнопки")
async def cmd(message: types.Message):
    await message.answer("Нажми кнопку", reply_markup=inline.start_keyboard)

@router.callback_query(F.data == "button1")
async def process_callback(callback_query: types.CallbackQuery):
    print("1")
    await callback_query.message.answer("Вы нажали на кнопку 1")
    await callback_query.answer()

@router.callback_query(F.data == "button2")
async def process_callback(callback_query: types.CallbackQuery):
    print("2")
    await callback_query.message.answer("Вы нажали на кнопку 2")
    await callback_query.answer()
