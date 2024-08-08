from typing import List, Generator

from openai import AsyncOpenAI, OpenAI


class BaseEmbedder:
    def __init__(
        self,
        client: OpenAI,
        async_client: AsyncOpenAI,
        model_name: str,
        batch_size: int = 32,
    ) -> None:
        self.model_name = model_name
        self.client = client
        self.async_client = async_client
        self.batch_size = batch_size

    def batch_generator(self, data: List[str]) -> Generator[List[str], None, None]:
        for i in range(0, len(data), self.batch_size):
            yield data[i : i + self.batch_size]

    def get_embedding(self, text: str) -> List[float]:
        response = self.client.embeddings.create(
            model=self.model_name,
            input=[text],
        )
        return response.data[0].embedding

    async def get_embedding_async(self, text: str) -> List[float]:
        response = await self.async_client.embeddings.create(
            model=self.model_name,
            input=[text],
        )
        return response.data[0]

    def get_batch_embeddings(self, texts: List[str]) -> List[List[float]]:
        response = self.client.embeddings.create(
            model=self.model_name,
            input=texts,
        )
        return [embedding.embedding for embedding in response.data]

    async def get_batch_embeddings_async(self, texts: List[str]) -> List[List[float]]:
        response = await self.async_client.embeddings.create(
            model=self.model_name,
            input=texts,
        )
        return [embedding.embedding for embedding in response.data]
