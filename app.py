import asyncio
import os
from aiogram import Bot, Dispatcher
from aiogram.client.bot import DefaultBotProperties
from aiogram.enums import ParseMode
from dotenv import load_dotenv, find_dotenv

from KeywordFinder.handlers.keyWordFindler import knowledge_base_router
from UserPromptResponse.handlers.userHandlers import ollama_router
from handlers.basic_handlers import router

load_dotenv(find_dotenv())

TOKEN = os.getenv('TOKEN')

if not TOKEN:
    raise ValueError("Токен бота не найден. Убедитесь, что он задан в файле .env")

bot = Bot(token=TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.HTML))
dp = Dispatcher()

dp.include_router(router)
dp.include_router(knowledge_base_router)
dp.include_router(ollama_router)
ALLOWED_UPDATES = ['message', 'edited_message']

async def main():
    await bot.delete_webhook(drop_pending_updates=True)
    await dp.start_polling(bot, allowed_updates=ALLOWED_UPDATES)

# Запуск бота
if __name__ == '__main__':
    asyncio.run(main())