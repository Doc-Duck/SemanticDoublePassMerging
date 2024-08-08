from openai import OpenAI, AsyncOpenAI

from src.embedders.embedder_base import BaseEmbedder
from src.config import EMBED_BASE_URL, EMBED_API_KEY, EMBED_MODEL, EMBED_BATCH_SIZE


sync_client = OpenAI(base_url=EMBED_BASE_URL, api_key=EMBED_API_KEY, max_retries=10)
async_client = AsyncOpenAI(
    base_url=EMBED_BASE_URL, api_key=EMBED_API_KEY, max_retries=10
)


EmbedderLocal = BaseEmbedder(sync_client, async_client, EMBED_MODEL, EMBED_BATCH_SIZE)
