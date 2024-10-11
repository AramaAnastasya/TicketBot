from aiogram import Router, F, types
from aiogram.types import Message
from aiogram.fsm.context import FSMContext
from keyboards import reply

ollama_router = Router()

@ollama_router.message(F.text.lower() == "ответ ollama")
async def cmd_cancel(message: Message, state: FSMContext):
    await state.clear()
    await message.answer(text="generation answer from Ollama", reply_markup=reply.start_kb)