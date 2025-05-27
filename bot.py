import os
import asyncio
import aiohttp
from aiogram import Bot, Dispatcher, types
from sympy import sympify, SympifyError
from dotenv import load_dotenv
from google.cloud import aiplatform_v1
from google.cloud.aiplatform_v1.types import PredictRequest

# --- Load environment variables ---
load_dotenv()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
PROJECT_ID = os.getenv("VERTEX_PROJECT")
LOCATION = os.getenv("VERTEX_LOCATION")
MODEL_ID = os.getenv("VERTEX_MODEL")
GOOGLE_APPLICATION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

# --- Telegram bot initialization ---
bot = Bot(token=TELEGRAM_TOKEN)
dp = Dispatcher()

# --- Vertex AI client setup ---
prediction_client = aiplatform_v1.PredictionServiceClient()
endpoint = prediction_client.endpoint_path(
    project=PROJECT_ID, location=LOCATION, endpoint=MODEL_ID
)

# --- Vertex AI: chat handler ---
async def chat_with_vertex_ai(prompt: str) -> str:
    try:
        instances = [{"prompt": prompt}]
        parameters = {"temperature": 0.7, "maxOutputTokens": 512}
        request = PredictRequest(
            endpoint=endpoint,
            instances=instances,
            parameters=parameters,
        )
        response = await asyncio.to_thread(prediction_client.predict, request=request)
        predictions = response.predictions
        return predictions[0].get("content", "🤖 No response received.")
    except Exception as e:
        return f"⚠️ Vertex AI error: {e}"

# --- Google Search handler ---
async def google_search(query: str) -> str:
    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "key": os.getenv("GOOGLE_API_KEY"),
        "cx": os.getenv("GOOGLE_CSE_ID"),
        "q": query,
        "num": 3
    }
    async with aiohttp.ClientSession() as session:
        async with session.get(url, params=params) as resp:
            if resp.status != 200:
                return f"Search error: HTTP {resp.status}"
            data = await resp.json()
            items = data.get("items", [])
            if not items:
                return "No results found."
            results = [f"{item['title']}\n{item['snippet']}\n{item['link']}" for item in items]
            return "\n\n".join(results)

# --- Joke API handler ---
async def fetch_joke() -> str:
    url = "https://official-joke-api.appspot.com/random_joke"
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as resp:
            if resp.status != 200:
                return "❌ Failed to fetch a joke."
            data = await resp.json()
            return f"{data.get('setup')}\n{data.get('punchline')}"

# --- Message handler ---
@dp.message()
async def handle_message(message: types.Message):
    text = message.text.strip()

    if text.lower() in ["/start", "hi", "hello"]:
        await message.answer(
            "👋 Hi! I'm a bot powered by Vertex AI (Gemini).\n\n"
            "Try:\n"
            "- search: Python → Google search\n"
            "- solve: 2+2*5 → Math calculation\n"
            "- joke → Get a random joke\n"
            "- chat: Tell me something about AI → Gemini response"
        )
        return

    if text.lower() == "/help":
        await message.answer("Available commands:\n- search:<query>\n- solve:<expression>\n- joke\n- chat:<message>")
        return

    if text.lower().startswith("search:"):
        query = text[7:].strip()
        await message.answer("🔍 Searching...")
        result = await google_search(query)
        await message.answer(result)
        return

    if text.lower().startswith("solve:"):
        expression = text[6:].strip()
        try:
            result = sympify(expression).evalf()
            await message.answer(f"Result: {result}")
        except SympifyError:
            await message.answer("❌ Invalid mathematical expression.")
        return

    if text.lower() == "joke":
        joke = await fetch_joke()
        await message.answer(joke)
        return

    if text.lower().startswith("chat:"):
        prompt = text[5:].strip()
        if not prompt:
            await message.answer("Please enter a message after 'chat:'.")
            return
        await message.answer("🤖 Thinking...")
        response = await chat_with_vertex_ai(prompt)
        await message.answer(response)
        return

    await message.answer("Unknown command. Type /help to see available options.")

# --- Main function ---
async def main():
    print("🚀 Telegram bot powered by Vertex AI is running...")
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())