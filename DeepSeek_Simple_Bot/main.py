import asyncio
import os
import re

from aiogram import Bot, Dispatcher, Router
from aiogram.enums import ChatType, ParseMode
from aiogram.filters import Command, CommandStart
from aiogram.types import Message
from dotenv import load_dotenv
from duckduckgo_search import DDGS
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

SYSTEM_PROMPT = (
    "You are a strict, laconic AI assistant. Always give DIRECT, PRECISE, and SHORT answers.\n\n"
    "RULES:\n"
    "- NO FLUFF. Forbidden: introductory phrases, rambling, hedging, or excuses "
    '(e.g. "it is hard to predict", "however", "on the one hand").\n'
    "- Answer the question straight away; do not add meta-commentary.\n"
    "- Markdown is FORBIDDEN: no ###, ##, **, or * for formatting.\n"
    "- Use Telegram HTML only. For bold, use <b>text</b>.\n"
    "- Format lists with simple hyphen bullets (lines starting with \"- \").\n"
    "- STRICTLY FORBIDDEN in your reply: any links or URLs (including http/https), "
    "HTML link tags (<a>...</a>), site or publication names, or any named sources.\n"
    "- Web search snippets in the user message are for internal reasoning only. "
    "Output ONLY the answer itself—never echo, cite, or list sources.\n"
    "- If snippets contain no usable facts (only bare links or noise, no substantive "
    'content or numbers), reply with exactly: Нет данных.\n'
    "- Never output a list of sources or references.\n"
)


def _clean_question_for_model(raw: str) -> str:
    """Убирает /ask и слово DeepSeek из текста перед запросом к модели."""
    t = raw.strip()
    t = re.sub(r"^/ask(?:@[\w]+)?\s*", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\bDeepSeek\b", "", t, flags=re.IGNORECASE)
    return " ".join(t.split()).strip()


async def get_web_search(query: str) -> str:
    """DuckDuckGo search; empty string on failure. Ru-ru region; FX-related queries get a CB/news bias."""

    q_lower = query.lower()
    if any(k in q_lower for k in ("курс", "доллар", "евро")):
        search_query = f"{query} ЦБ РФ новости сводка"
    else:
        search_query = query

    def _sync_search() -> list[dict[str, str]]:
        with DDGS() as ddgs:
            return ddgs.text(search_query, region="ru-ru", max_results=7)

    try:
        results = await asyncio.to_thread(_sync_search)
        if not results:
            return ""
        chunks: list[str] = []
        for item in results:
            title = (item.get("title") or "").strip()
            body = (item.get("body") or "").strip()
            href = (item.get("href") or "").strip()
            block = "\n".join(x for x in (title, body, href) if x)
            if block:
                chunks.append(block)
        return "\n\n".join(chunks)
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
