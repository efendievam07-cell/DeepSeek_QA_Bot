import asyncio
import os
import re

from aiogram import Bot, Dispatcher, Router
from aiogram.enums import ChatType, ParseMode
from aiogram.filters import CommandStart
from aiogram.types import Message
from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()

BOT_TOKEN = os.getenv("BOT_TOKEN")
if not BOT_TOKEN:
    raise ValueError("ОШИБКА: BOT_TOKEN не найден. Проверь файл .env!")

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Заполняются в main() через bot.get_me() до start_polling
BOT_USERNAME: str | None = None
BOT_ID: int | None = None

router = Router()
openai_client = AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

SYSTEM_PROMPT = (
    "Ты профессиональный ассистент. Всегда отвечай структурировано. "
    "Обязательно используй абзацы, маркированные списки и выделяй главные "
    "мысли жирным шрифтом. Пиши емко, без лишней воды. Добавляй подходящие "
    "по смыслу эмодзи, но не переборщи."
)


def _clean_question_for_model(raw: str, bot_username: str | None) -> str:
    """Убирает /ask, @username бота и слово DeepSeek из текста перед запросом к модели."""
    t = raw.strip()
    t = re.sub(r"^/ask(?:@[\w]+)?\s*", "", t, flags=re.IGNORECASE)
    if bot_username:
        t = re.sub(rf"@{re.escape(bot_username)}\b", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\bDeepSeek\b", "", t, flags=re.IGNORECASE)
    return " ".join(t.split()).strip()


def _mentions_bot(text: str, bot_username: str | None) -> bool:
    if not bot_username:
        return False
    u = bot_username.lower()
    low = text.lower()
    if f"@{u}" in low:
        return True
    return bool(re.match(rf"^/ask@{re.escape(u)}(\s|$)", text.strip(), re.IGNORECASE))


def _is_ask_command(text: str) -> bool:
    return bool(re.match(r"^/ask(?:@[\w]+)?(\s|$)", text.strip(), re.IGNORECASE))


def _should_handle_ai_message(message: Message) -> bool:
    chat_type = message.chat.type
    if chat_type == ChatType.PRIVATE:
        return True
    if chat_type not in (ChatType.GROUP, ChatType.SUPERGROUP):
        return False
    if BOT_ID is None:
        return False

    reply = message.reply_to_message
    if reply and reply.from_user and reply.from_user.id == BOT_ID:
        return True

    text = message.text or ""
    if not _is_ask_command(text):
        return False
    return _mentions_bot(text, BOT_USERNAME)


@router.message(CommandStart())
async def cmd_start(message: Message) -> None:
    await message.answer("Привет! ИИ-ассистент на базе DeepSeek.")


@router.message()
async def on_text(message: Message) -> None:
    if not message.text:
        return

    if re.match(r"^/start(?:@[\w]+)?(\s|$)", message.text.strip(), re.IGNORECASE):
        return

    if not _should_handle_ai_message(message):
        return

    if message.chat.type == ChatType.PRIVATE:
        user_content = message.text.strip()
    else:
        user_content = _clean_question_for_model(message.text, BOT_USERNAME)

    if message.chat.type in (ChatType.GROUP, ChatType.SUPERGROUP) and not user_content:
        return

    thinking = await message.answer("⏳ Думаю...")
    try:
        completion = await openai_client.chat.completions.create(
            model="deepseek/deepseek-chat",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
        )
        text = completion.choices[0].message.content or ""
        await thinking.edit_text(text, parse_mode=ParseMode.MARKDOWN)
    except Exception as exc:  # noqa: BLE001
        await thinking.edit_text(f"Не удалось получить ответ: {exc}")


async def main() -> None:
    if not OPENROUTER_API_KEY:
        raise SystemExit("Заполните OPENROUTER_API_KEY в файле .env")

    bot = Bot(token=BOT_TOKEN)
    bot_info = await bot.get_me()
    global BOT_USERNAME, BOT_ID
    BOT_USERNAME = bot_info.username
    BOT_ID = bot_info.id

    dp = Dispatcher()
    dp.include_router(router)
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
