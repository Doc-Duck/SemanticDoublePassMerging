from scipy import spatial
from nltk.tokenize import sent_tokenize
from langdetect import detect, DetectorFactory
from src.embedders.embedder_base import BaseEmbedder


class Semantic2PMSplitter:
    def __init__(self, embedder: BaseEmbedder) -> None:
        DetectorFactory.seed = 0
        self.embedder = embedder

    @staticmethod
    def detect_language(text: str) -> str:
        language = detect(text)
        if language == "ru":
            language = "russian"
        elif language == "en":
            language = "english"
        else:
            raise ValueError("Language not supported")
        return language

    @staticmethod
    def split_into_sentences(text: str, language: str) -> list[str]:
        return sent_tokenize(text, language)

    @staticmethod
    def calculate_cs(vector1: list[float], vector2: list[float]) -> float:
        return 1 - spatial.distance.cosine(vector1, vector2)

    async def first_pass(
        self,
        sentences: list[str],
        max_chunk_length: int,
        initial_threshold: float,
        appending_threshold: float,
    ) -> list[str]:
        chunks = []
        current_chunk = ""
        current_chunk_sent_count = 0

        for i, sentence in enumerate(sentences):
            if current_chunk:
                current_chunk_embed = await self.embedder.get_embedding_async(
                    current_chunk
                )
                sentence_embed = await self.embedder.get_embedding_async(sentence)
                distance = self.calculate_cs(current_chunk_embed, sentence_embed)

                if current_chunk_sent_count == 1:
                    threshold = initial_threshold
                else:
                    threshold = appending_threshold

                if distance > threshold:
                    current_chunk += " " + sentence
                    current_chunk_sent_count += 1
                else:
                    chunks.append(current_chunk)
                    current_chunk = sentence
                    current_chunk_sent_count = 1
            else:
                current_chunk = sentence
                current_chunk_sent_count = 1

            if len(current_chunk) >= max_chunk_length:
                chunks.append(current_chunk)
                current_chunk = ""
                current_chunk_sent_count = 0

        if current_chunk:
            chunks.append(current_chunk)
        elif i == len(sentences) - 1 and chunks:
            chunks[-1] += " " + sentence

        return chunks

    async def second_pass(
        self,
        fp_chunks: list[str],
        max_chunk_length: int,
        merging_threshold: float,
    ) -> list[str]:
        chunks = fp_chunks.copy()
        i = 0

        while i < len(chunks) - 1:
            first_chunk = chunks[i]
            second_chunk = chunks[i + 1]

            if len(first_chunk) >= max_chunk_length:
                i += 1
                continue

            first_embed = await self.embedder.get_embedding_async(first_chunk)
            second_embed = await self.embedder.get_embedding_async(second_chunk)

            cs_first_second = self.calculate_cs(first_embed, second_embed)

            if cs_first_second > merging_threshold:
                merged_chunk = first_chunk + " " + second_chunk
                if len(merged_chunk) <= max_chunk_length:
                    chunks[i] = merged_chunk
                    chunks.pop(i + 1)
                    continue

            if i + 2 <= len(chunks) - 1:
                third_chunk = chunks[i + 2]
                third_embed = await self.embedder.get_embedding_async(second_chunk)
                cs_first_third = self.calculate_cs(first_embed, third_embed)

                if cs_first_third > merging_threshold:
                    merged_chunk = first_chunk + " " + second_chunk + " " + third_chunk
                    if len(merged_chunk) <= max_chunk_length:
                        chunks[i] = merged_chunk
                        chunks.pop(i + 1)
                        chunks.pop(i + 1)
                        continue

            i += 1

        return chunks

    async def chunk(
        self,
        text: str,
        max_chunk_length: int,
        initial_threshold: float,
        merging_threshold: float,
        appending_threshold: float,
    ) -> list[str]:
        language = self.detect_language(text)
        sentences = self.split_into_sentences(text, language)
        print(f'Sentences: {len(sentences)}')

        fp_chunks = await self.first_pass(
            sentences, max_chunk_length, initial_threshold, appending_threshold
        )
        print(f'First pass chunks: {len(fp_chunks)}')
        sp_chunks = await self.second_pass(
            fp_chunks, max_chunk_length, merging_threshold
        )
        print(f'Second pass chunks: {len(sp_chunks)}')
        return sp_chunks
