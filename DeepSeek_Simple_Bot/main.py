import asyncio
import os
import re

from aiogram import Bot, Dispatcher, Router
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

router = Router()
openai_client = AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

SYSTEM_PROMPT = """Ты — умный и полезный ИИ-ассистент. Твоя задача — давать качественные ответы, опираясь на переданные данные из интернета.

ПРАВИЛА:

АДАПТИВНОСТЬ: Если пользователь спрашивает точный факт (курс валют, число, погоду) — дай короткий и четкий ответ с цифрой. Если вопрос открытый (советы, инструкции, 'как сделать') — дай подробный, логичный и развернутый ответ.

ФОРМАТИРОВАНИЕ: КАТЕГОРИЧЕСКИ ЗАПРЕЩЕНО использовать Markdown (никаких ###, ##, **, *).

Для выделения главных мыслей и заголовков используй ТОЛЬКО HTML-теги: <b>текст</b>. Списки оформляй обычными дефисами (-).

БЕЗ ССЫЛОК: Никогда не выводи URL-адреса, теги <a> и названия сайтов-источников. Игнорируй их в поисковых данных. Просто дай ответ от себя."""


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


@router.message()
async def on_text(message: Message) -> None:
    if not message.text:
        return

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
    if web_context:
        user_message = (
            "Актуальные данные из поиска по запросу пользователя:\n"
            f"{web_context}\n\n"
            f"Вопрос пользователя:\n{user_content}"
        )
    else:
        user_message = user_content

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


async def main() -> None:
    if not OPENROUTER_API_KEY:
        raise SystemExit("Заполните OPENROUTER_API_KEY в файле .env")

    bot = Bot(token=BOT_TOKEN)
    dp = Dispatcher()
    dp.include_router(router)
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
