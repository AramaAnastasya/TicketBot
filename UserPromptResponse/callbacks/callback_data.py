from aiogram.fsm.context import FSMContext
from aiogram import Router, F, Bot, types
from aiogram.types import Message
from aiogram.filters.callback_data import CallbackData
from UserPromptResponse.utils.states import Form
from keyboards.reply import cancel, main, start_kb
from handlers.keyboards.inline import init_gen, init_gen_d, init_quest, init_quest_d

router = Router()

