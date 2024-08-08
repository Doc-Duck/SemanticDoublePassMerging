import pytest
from unittest.mock import AsyncMock, patch
from scipy import spatial
from nltk.tokenize import sent_tokenize
from langdetect import DetectorFactory
from src.embedders.embedder_base import BaseEmbedder
from src.semantic2pm_chunker import Semantic2PMSplitter

# Ensure langdetect's seed is set for consistent results
DetectorFactory.seed = 0


@pytest.fixture
def embedder_mock():
    embedder = AsyncMock(spec=BaseEmbedder)
    embedder.get_embedding_async.return_value = [0.1, 0.2, 0.3]
    return embedder


@pytest.fixture
def splitter(embedder_mock):
    return Semantic2PMSplitter(embedder_mock)


def test_detect_language():
    assert Semantic2PMSplitter.detect_language("This is a test.") == "english"
    assert Semantic2PMSplitter.detect_language("Это тест.") == "russian"
    with pytest.raises(ValueError):
        Semantic2PMSplitter.detect_language("これはテストです。")


def test_split_into_sentences():
    text = "This is a test. This is another test."
    assert Semantic2PMSplitter.split_into_sentences(text, "english") == [
        "This is a test.",
        "This is another test.",
    ]


def test_calculate_cs():
    vector1 = [0.1, 0.2, 0.3]
    vector2 = [0.1, 0.2, 0.3]
    dist = Semantic2PMSplitter.calculate_cs(vector1, vector2)
    assert dist == 1.0


@pytest.mark.asyncio
async def test_first_pass(splitter: Semantic2PMSplitter):
    sentences = ["This is a test.", "This is another test."]
    max_chunk_length = 50
    initial_threshold = 0.5
    appending_threshold = 0.3
    result = await splitter.first_pass(
        sentences, max_chunk_length, initial_threshold, appending_threshold
    )
    assert result == ["This is a test. This is another test."]


@pytest.mark.asyncio
async def test_second_pass(splitter: Semantic2PMSplitter):
    fp_chunks = ["This is a test.", "This is another test.", "This is another test 2"]
    max_chunk_length = 200
    merging_threshold = 0.1
    result = await splitter.second_pass(fp_chunks, max_chunk_length, merging_threshold)
    assert result == ["This is a test. This is another test. This is another test 2"]


@pytest.mark.asyncio
async def test_chunk(splitter: Semantic2PMSplitter):
    text = "This is a test. This is another test."
    max_chunk_length = 50
    initial_threshold = 0.5
    merging_threshold = 0.5
    appending_threshold = 0.3
    result = await splitter.chunk(
        text,
        max_chunk_length,
        initial_threshold,
        merging_threshold,
        appending_threshold,
    )
    assert result == ["This is a test. This is another test."]
