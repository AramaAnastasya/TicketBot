from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton

start_keyboard = InlineKeyboardMarkup(
    inline_keyboard=[
        [
            InlineKeyboardButton(text="Кнопка 1", callback_data="button1"),
            InlineKeyboardButton(text="Кнопка 2", callback_data="button2")
        ]
    ]
)