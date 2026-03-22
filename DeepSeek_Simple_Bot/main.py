import asyncio
import base64
import os
import re
from datetime import datetime
from io import BytesIO

from aiogram import Bot, Dispatcher, F, Router
from aiogram.enums import ChatType, ParseMode
from aiogram.filters import Command, CommandStart
from aiogram.types import Message
from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()

BOT_TOKEN = os.getenv("BOT_TOKEN")
if not BOT_TOKEN:
    raise ValueError("ОШИБКА: BOT_TOKEN не найден. Проверь файл .env!")

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

ALLOWED_TOPIC_ID = 915
VISION_MODEL = os.getenv("OPENROUTER_VISION_MODEL", "google/gemini-2.0-flash-001")

router = Router()
openai_client = AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

SYSTEM_PROMPT = """Ты — умный и полезный ИИ-ассистент. Твоя задача — давать качественные ответы, опираясь на переданные данные из интернета.

ПРАВИЛА:

ДАТА И ВРЕМЯ: В начале каждого пользовательского сообщения передаётся строка «Сегодня …» — считай эту дату и время актуальными. Для вопросов вроде «что произошло сегодня» опирайся на неё и на блок «Данные из интернета». Не используй Markdown-звёздочки; жирный только через HTML <b>текст</b>.

АДАПТИВНОСТЬ: Если пользователь спрашивает точный факт (курс валют, число, погоду) — дай короткий и четкий ответ с цифрой. Если вопрос открытый (советы, инструкции, 'как сделать') — дай подробный, логичный и развернутый ответ.

ФОРМАТИРОВАНИЕ: КАТЕГОРИЧЕСКИ ЗАПРЕЩЕНО использовать Markdown (никаких ###, ##, **, *).

КАТЕГОРИЧЕСКИ ЗАПРЕЩЕНО использовать двойные звёздочки ** (как в **таком** оформлении) или любые другие символы Markdown. Для выделения — ТОЛЬКО HTML-теги: <b>текст</b>. Если ты используешь ** в ответе, сообщение не будет доставлено. Будь предельно внимателен!

Списки оформляй обычными дефисами (-).

БЕЗ ССЫЛОК: Никогда не выводи URL-адреса, теги <a> и названия сайтов-источников. Игнорируй их в поисковых данных. Просто дай ответ от себя."""

VISION_SYSTEM_PROMPT = """You are a helpful vision assistant. Answer clearly about what you see in the image.

RULES:
- Do not use Markdown (no ###, ##, **, or * for formatting).
- Use only HTML <b>text</b> for bold or key emphasis—never **.
- Do not output URLs, <a> tags, or source names unless the user explicitly asks for a link.
- If the user adds a caption with a question, answer that question using the image."""


def _clean_question_for_model(raw: str) -> str:
    """Убирает /ask и слово DeepSeek из текста перед запросом к модели."""
    t = raw.strip()
    t = re.sub(r"^/ask(?:@[\w]+)?\s*", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\bDeepSeek\b", "", t, flags=re.IGNORECASE)
    return " ".join(t.split()).strip()


async def get_web_search(query: str) -> str:
    try:
        from duckduckgo_search import DDGS

        with DDGS() as ddgs:
            results = ddgs.text(query, max_results=7)
            if not results:
                return ""
            return "\n".join([f"- {r.get('body', '')}" for r in results])
    except Exception:
        return ""


@router.message(CommandStart())
async def cmd_start(message: Message) -> None:
    await message.answer("Привет! ИИ-ассистент на базе DeepSeek.")


@router.message(Command("topic_id"))
async def cmd_topic_id(message: Message) -> None:
    await message.answer(
        f"Идентификатор этой темы: {message.message_thread_id}"
    )


@router.message(F.text)
async def on_text(message: Message) -> None:
    if re.match(r"^/start(?:@[\w]+)?(\s|$)", message.text.strip(), re.IGNORECASE):
        return

    if re.match(r"^/topic_id(?:@[\w]+)?(\s|$)", message.text.strip(), re.IGNORECASE):
        return

    if message.chat.type in (ChatType.GROUP, ChatType.SUPERGROUP):
        if message.message_thread_id != ALLOWED_TOPIC_ID:
            return

    user_content = _clean_question_for_model(message.text)
    if not user_content:
        return

    web_context = await get_web_search(user_content)

    today = datetime.now().strftime("%d %B %Y года, день недели: %A")
    prompt_with_date = (
        f"ВНИМАНИЕ: Сегодня {today}. Использовать эту дату как текущую.\n\n"
        f"Данные из интернета:\n{web_context}\n\n"
        f"Вопрос пользователя: {message.text}"
    )
    user_message = prompt_with_date

    thinking = await message.answer("⏳ Думаю...", parse_mode=ParseMode.HTML)
    # Один и тот же диалог для лички и для группы (после фильтра темы): system + user с вопросом и при необходимости web_context
    messages: list[dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]
    try:
        completion = await openai_client.chat.completions.create(
            model="deepseek/deepseek-chat",
            messages=messages,
        )
        text = completion.choices[0].message.content or ""
        await thinking.edit_text(text, parse_mode=ParseMode.HTML)
    except Exception as exc:  # noqa: BLE001
        await thinking.edit_text(f"Не удалось получить ответ: {exc}")


@router.message(F.photo)
async def on_photo(message: Message) -> None:
    if message.chat.type in (ChatType.GROUP, ChatType.SUPERGROUP):
        if message.message_thread_id != ALLOWED_TOPIC_ID:
            return

    photo = message.photo[-1]
    buf = BytesIO()
    try:
        file_info = await message.bot.get_file(photo.file_id)
        await message.bot.download_file(file_info.file_path, buf)
    except Exception:
        await message.answer("Не удалось скачать изображение.")
        return

    raw = buf.getvalue()
    if not raw:
        await message.answer("Пустой файл изображения.")
        return

    b64 = base64.b64encode(raw).decode("ascii")
    data_url = f"data:image/jpeg;base64,{b64}"

    user_text = (message.caption or "").strip()
    if not user_text:
        user_text = "Describe the image briefly. If there is text in the image, summarize it."

    thinking = await message.answer("⏳ Думаю...", parse_mode=ParseMode.HTML)
    vision_messages: list[dict] = [
        {"role": "system", "content": VISION_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_text},
                {"type": "image_url", "image_url": {"url": data_url}},
            ],
        },
    ]
    try:
        completion = await openai_client.chat.completions.create(
            model=VISION_MODEL,
            messages=vision_messages,
        )
        text = completion.choices[0].message.content or ""
        await thinking.edit_text(text, parse_mode=ParseMode.HTML)
    except Exception as exc:  # noqa: BLE001
        await thinking.edit_text(f"Не удалось получить ответ: {exc}")


async def main() -> None:
    if not OPENROUTER_API_KEY:
        raise SystemExit("Заполните OPENROUTER_API_KEY в файле .env")

    bot = Bot(token=BOT_TOKEN)
    dp = Dispatcher()
    dp.include_router(router)
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
