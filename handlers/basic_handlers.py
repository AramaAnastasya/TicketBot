import asyncio
import os
from aiogram import Router, types
from aiogram.filters import Command
from aiogram import Bot, Dispatcher
from aiogram.client.bot import DefaultBotProperties
from aiogram.enums import ParseMode
from keyboards import reply
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


# # Обработчик текстовых сообщений
# @router.message()
# async def echo(message: types.Message):
#     await message.answer(f"Вы сказали: {message.text}")


# import os
# from aiogram import F, types, Router, Bot,Dispatcher
# from aiogram.enums import ParseMode
# from aiogram.filters import CommandStart, Command, or_f
# from aiogram.fsm.state import StatesGroup, State, default_state
# from aiogram.fsm.context import FSMContext
# from aiogram.utils.formatting import as_list, as_marked_section, Bold,Spoiler #Italic, as_numbered_list и тд 
# from aiogram.types import Message
# from keyboards import reply
# from handlers.utils.states import FShandlesr


# bot = Bot(token=os.getenv('TOKEN'))
# dp = Dispatcher()
# user_private_router = Router()

# @user_private_router.message(CommandStart())
# async def start_cmd(message:Message, state: FSMContext):
#     # Получить информацию о члене чата (пользователе, отправившем сообщение)
#     chat_member = await bot.get_chat_member(message.chat.id, message.from_user.id)
#     # Вывести имя пользователя
#     await bot.send_message(message.chat.id, 
#             f"Добрый день, <b>{chat_member.user.first_name}</b>! Я виртуальный помощник для ответов на ваши вопросы.\nЧто ваша проблема?",
#             reply_markup=reply.start_kb        
#             )
#     await state.set_state(FShandlesr.exeption)

# # @user_private_router.message((F.text.lower() == "Ответ по базе знаний"))
# # async def menu_cmd(message: types.Message):
# #     await message.answer(
# #         "generation answer from knowledge base",
# #         reply_markup=reply.start_kb
# #     )

# # @user_private_router.message(F.text.lower() == "Ответ Ollama")
# # async def cmd_cancel(message: Message, state: FSMContext):
# #     await state.clear()
# #     await message.answer(
# #         text="generation answer from Ollama",
# #         reply_markup=reply.start_kb
# #     )