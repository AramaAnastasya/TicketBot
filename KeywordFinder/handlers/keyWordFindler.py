from aiogram import Router, F, types
from aiogram.types import Message
from keyboards import reply

knowledge_base_router = Router()

knowledge_base = {
    "doc1": ["Это документ 1. Он содержит информацию о...", "Это продолжение документа 1."],
    "doc2": ["Это документ 2. Он содержит информацию о...", "Это продолжение документа 2."],
    "doc3": ["Это документ 3. Он содержит информацию о..."]
}

@knowledge_base_router.message(F.text.lower() == "ответ по базе знаний")
async def menu_cmd(message: types.Message):
    await message.answer("generation answer from knowledge base", reply_markup=reply.start_kb)

@knowledge_base_router.message()
async def menu_cmd(message: types.Message):
    # Получаем текст, отправленный пользователем
    user_text = message.text.lower()
    
    # Ищем соответствующий ответ в базе знаний
    for doc_id, doc_text_list in knowledge_base.items():
        for doc_text in doc_text_list:
            if user_text in doc_text.lower():
                # Отправляем ответ пользователю
                await message.answer(f"Ответ из базы знаний:\n{doc_text}", reply_markup=reply.start_kb)
                return
    # Если не найден соответствующий ответ, отправляем сообщение об ошибке
    await message.answer("К сожалению, я не нашел ответа в базе знаний.", reply_markup=reply.start_kb)
# # Создаем базу знаний
# knowledge_base = {
#     "doc1": ["Это документ 1. Он содержит информацию о...", "Это продолжение документа 1."],
#     "doc2": ["Это документ 2. Он содержит информацию о...", "Это продолжение документа 2."],
#     "doc3": ["Это документ 3. Он содержит информацию о..."]
# }

# @knowledge_base_router.message(F.text.lower() == "ответ по базе знаний")
# async def menu_cmd(message: types.Message):
#     # Получаем случайный документ из базы знаний
#     import random
#     doc_id = random.choice(list(knowledge_base.keys()))
#     doc_text = "\n\n".join(knowledge_base[doc_id])
    
#     # Отправляем ответ пользователю
#     await message.answer(f"Ответ из базы знаний:\n{doc_text}", reply_markup=reply.start_kb)