import os
from aiogram import F, types, Router, Bot,Dispatcher
from aiogram.filters import CommandStart, Command, or_f
from aiogram.enums import ParseMode
from aiogram.types import Message, CallbackQuery
from aiogram.fsm.context import FSMContext
from keyboards import reply

from dotenv import load_dotenv
load_dotenv()

ollama_router = Router()

TOKEN = os.getenv('TOKEN')
bot = Bot(token=TOKEN )
dp = Dispatcher()

@ollama_router.message(F.text.lower() == "ответ ollama")
async def cmd_cancel(message: Message):
    await message.answer(text="generation answer from Ollama", reply_markup=reply.start_kb)