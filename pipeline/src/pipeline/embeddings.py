from tenacity import (
    retry,
    wait_exponential,
    stop_after_attempt,
    retry_if_exception_type,
    RetryError,
    RetryCallState,
)
from openai import OpenAI
from openai.types import Batch
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from rich.logging import RichHandler
from rich.console import Console
from rich.panel import Panel
from decouple import config
import polars as pl
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
import tiktoken
import logging
import json
import os
from typing import Iterable

from .schemas import BatchCreate, BatchUpdate
from .db.crud import JobRepository, BatchRepository
from .db.database import get_session

logging.basicConfig(
    level=logging.INFO, handlers=[RichHandler(rich_tracebacks=True, markup=True)]
)
logger = logging.getLogger(__name__)

console = Console()


MAX_INPUT_TOKENS = 1_000  # 200_000  # Max input tokens across all inputs in a single request. According to OpenAI docs is 300_000 but OpenAI counts different than tiktoken
MAX_REQUESTS_PER_BATCH = 100  # 50_000  # According to OpenAI batch inference API docs
MAX_REQUESTS_PER_MINUTE = 3_000  # According to OpenAI docs Tier 1 = 3_000 RPM
MAX_TOKENS_PER_MINUTE = 1_000_000  # According to OpenAI docs Tier 1 = 1_000_000 TPM
MAX_BATCH_FILE_SIZE = (
    200 * 1024 * 1024
)  # 10GB According to Nebius batch inference API docs (converted to bytes)


@dataclass
class EmbeddingRequest:
    request_id: int
    input: str | list[str]
    token_count: int

    def to_json_line(self, job_id: int, batch_id: str, model: str, datestr: str) -> str:
        """Serializes request into JSONL format required by the API"""
        payload = {
            "custom_id": f"batch-{job_id}-{datestr}_{batch_id}-request-{self.request_id}",
            "method": "POST",
            "url": "/v1/embeddings",
            "body": {"model": model, "input": self.input},
        }
        return json.dumps(payload, ensure_ascii=False) + "/n"


@dataclass
class BatchRequest:
    batch_id: int
    input: list[EmbeddingRequest]
    text_count: int
    token_count: int


# Utility functions
def tokenize(texts_chunk: list, model_name: str) -> list[int]:
    """Worker function to tokenize a chunk of texts."""
    try:
        encoding = tiktoken.encoding_for_model(model_name)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    return [len(encoding.encode(t, allowed_special="all")) for t in texts_chunk]


def chunk_list(lst: list, chunk_size: int) -> Iterable[list[str]]:
    """Split list into chunks of specified size."""
    for i in range(0, len(lst), chunk_size):
        yield lst[i : i + chunk_size]


def map_openai_batch_to_dict(batch_request: Batch) -> dict:
    """Maps an OpenAI Batch object to a flattened Python dictionary for DB schemas."""

    payload = batch_request.model_dump(exclude={"id", "request_counts"})
    payload["batch_id"] = batch_request.id

    if batch_request.request_counts:
        counts_dict = {
            f"request_{k}": v
            for k, v in batch_request.request_counts.model_dump().items()
        }
        payload.update(counts_dict)

    return payload


class VectorEmbeddings:
    def __init__(
        self,
        model: str,
        max_input_tokens: int = MAX_INPUT_TOKENS,
    ):
        self.model = model
        self.max_input_tokens = max_input_tokens

    def _post_batch_api(self, id: int, file: str) -> None:
        """Uploads the file and request batch embedding job to the API"""

        with OpenAI(
            base_url="https://api.tokenfactory.nebius.com/v1/",
            api_key=config("NEBIUS_API_KEY"),
        ) as client:

            # file_upload = client.files.create(
            #     file=open(file, "rb"), purpose="batch"
            # )

            # batch_request = client.batches.create(
            #     input_file_id=file_upload.id,
            #     endpoint="/v1/embeddings",
            #     completion_window="24h",
            # )

            batch_request = client.batches.retrieve(
                "batch_019c168e-6010-7a8b-96bf-5b30a3a40ee3"
            )  # TODO. Remove after tests

            self._log_batch_request(id, batch_request, file)

    def _get_batch_api(self, batch_id: str) -> Batch:

        with OpenAI(
            base_url="https://api.tokenfactory.nebius.com/v1/",
            api_key=config("NEBIUS_API_KEY"),
        ) as client:

            return client.batches.retrieve(batch_id)

    def _create_batch_file(
        self, id: int, batch: BatchRequest, max_file_size: int = MAX_BATCH_FILE_SIZE
    ) -> list[str]:
        """
        Creates multiple JSONL files that represent the batch requests.
        Each file has a maximum size allowed according to the API (OpenAI 200MB)
        """

        datestr = datetime.now().strftime("%Y%m%d-%H%M")
        base_name = f"{self.model.split("/")[-1].lower()}-{id}-{datestr}-batch-{batch.batch_id}.jsonl"
        base_path = f"./data/embeddings/{id}/batches"
        file = Path(f"{base_path}/{base_name}")
        file.parent.mkdir(parents=True, exist_ok=True)

        files = []
        current_file_size = 0
        split_num = 1

        f = open(file, "w+", encoding="utf-8")

        try:
            for req in batch.input:
                line = req.to_json_line(id, batch.batch_id, self.model, datestr)
                line_size = len(line.encode("utf-8"))

                if current_file_size + line_size > max_file_size:
                    f.close()
                    files.append(f.name)
                    logger.info(
                        f"[yellow]✨ Created file {f.name} for Batch #{batch.batch_id}.[/]"
                    )

                    file_split = file.parent / f"{file.stem}_{split_num}.jsonl"
                    f = open(file_split)
                    current_file_size = 0
                    split_num += 1

                f.write(line)
                current_file_size += line_size

        finally:
            if f:
                f.close()
                files.append(f.name)
                logger.info(
                    f"[yellow]✨ Created file {f.name} for Batch #{batch.batch_id}.[/]"
                )

        return files

    def _generate_embedding_request(
        self, texts: list[str], counts: list[int]
    ) -> Iterable[EmbeddingRequest]:
        """
        Generator function that create EmbeddingRequest objects according to the
        maximum limit of tokens allowed per request (TPR)
        """

        logger.info("Batching requests...")

        current_input, current_count = [], 0
        req_id = 1

        for text, count in zip(texts, counts):
            if current_count + count > self.max_input_tokens and current_input:
                yield EmbeddingRequest(
                    request_id=req_id, input=current_input, token_count=current_count
                )
                current_input, current_count = [], 0
                req_id += 1

            current_input.append(text)
            current_count += count

        if current_input:
            yield EmbeddingRequest(
                request_id=req_id, input=current_input, token_count=current_count
            )

    def _batch_texts(self, id: int, texts: list[str]) -> list[str]:
        """
        Takes a list of texts to embed and split them in multiple batches depending on
        the API limits: Tokens Per Request > Requests Per Batch > Batch File Size
        """

        console.print(
            Panel(
                f"[bold cyan]Model:[/bold cyan] {self.model}\n"
                f"[bold cyan]Total texts:[/bold cyan] {len(texts):,}\n"
                f"[bold cyan]Max tokens per request:[/bold cyan] {self.max_input_tokens:,}\n"
                f"[bold cyan]Max requests per batch:[/bold cyan] {MAX_REQUESTS_PER_BATCH:,}\n",
                title="[bold]Embedding Configuration[/bold]",
                border_style="cyan",
            )
        )

        logger.info(f"Pre-tokenizing texts for Rate Limits...")

        # Pre-tokenization (parallel workers)
        token_counts = self._tokenize_texts(texts)

        # Process texts into request objects
        requests = list(self._generate_embedding_request(texts, token_counts))

        # Process requests into batch files
        request_files = [
            self._create_batch_file(
                id,
                BatchRequest(
                    i,
                    batch,
                    sum(len(r.input) for r in batch),
                    sum(r.token_count for r in batch),
                ),
            )
            for i, batch in enumerate(
                chunk_list(requests, MAX_REQUESTS_PER_BATCH), start=1
            )
        ]

        return [f for batch_files in request_files for f in batch_files]

    def _log_batch_request(self, job_id: int, batch_request: Batch, local_file: str):
        """
        Takes a Batch request object from the OpenAI API and creates a new record in the local DB.
        It leverages that Batch is a pydantic model.
        """

        payload = map_openai_batch_to_dict(batch_request)
        batch_db = BatchCreate(job_id=job_id, local_file_id=local_file, **payload)

        with get_session() as db:
            batch = BatchRepository.create_batch(db, batch_db)

        logger.info(f"Created batch request {batch}")

    def _tokenize_texts(self, texts: list[str], max_workers: int = None) -> list[int]:
        """Tokenize texts in parallel using multiprocessing."""

        if max_workers is None:
            max_workers = os.cpu_count()

        # Chunk size: distribute work evenly across workers
        chunk_size = max(1, len(texts) // (max_workers * 4))
        chunks = list(chunk_list(texts, chunk_size))

        logger.info(
            f"Tokenizing {len(texts):,} texts using {max_workers} workers ({len(chunks)} chunks)..."
        )

        try:
            tiktoken.encoding_for_model(self.model)
            model_name = self.model
        except:
            logger.warning(
                f"Could not find a tokenizer in tiktoken for model {self.model}. Defaulting to `cl100k_base` tokenizer. Token count is approximated!"
            )
            # NOTE. Can use AutoTokenizer from HF but there's no need for exact matching tokens (only token count). HF very slow to download models.
            # encoding = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            model_name = "cl100k_base"

        # Use ProcessPoolExecutor for CPU-bound work
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            worker_fn = partial(tokenize, model_name=model_name)
            results = list(
                tqdm(
                    executor.map(worker_fn, chunks),
                    total=len(chunks),
                    desc="Tokenizing",
                )
            )

        # Flatten results
        token_counts = [count for chunk_result in results for count in chunk_result]

        return token_counts

    def _parse_texts(self, file: str, text_column: str | list[str]) -> list[str]:
        """
        Read texts from the .parquet file using polars and select columns
        to create the final texts to embed.
        """

        logger.info(
            f"Reading data from {file} 🔎\nThis may take a few seconds depending on the size of data..."
        )

        df = pl.read_parquet(file).head(2_000)  # TODO. Remove .head() in prod

        if isinstance(text_column, list):
            texts = (
                df.select(pl.concat_str(text_column, separator="\n\n"))
                .to_series()
                .to_list()
            )
        else:
            texts = df[text_column].to_list()

        return texts

    def _create_job(self):
        """Creates a new Job in DB"""

        with get_session() as db:
            job = JobRepository.create_job(db, self.model)
        return job

    def encode_batch(
        self,
        file: str,
        text_column: str | list[str],
    ):
        """
        Chunks texts from a Parquet file and submits them for batch embedding.

        Args:
            file: Path to the Parquet file containing the data.
            text_column: A single column name or a list of column names to embed.
                Multiple columns are concatenated with double newlines ('\n\n').

        Returns:
            A list of paths to the generated batch request files.
        """

        f = Path(file)

        assert (
            f.suffix == ".parquet"
        ), f"Expected 'file' to be a .parquet file, instead got {file}"
        assert f.exists(), f"Couldn't find the parquet file {file}"

        texts = self._parse_texts(file, text_column)
        logger.info(
            f"Found {len(texts)} texts! Creating new job to compute embeddings..."
        )

        job = self._create_job()
        logger.info(f"New Job {job.id} created for model '{job.model}'")

        batch_files = self._batch_texts(job.id, texts)

        for batch_f in batch_files:
            self._post_batch_api(job.id, batch_f)

        return job.id

    def check_status(self, job_id: int, refresh: bool = False) -> None:
        # 1. Get job_id from db
        # 2. Find all different batch_id from the job
        # 3. Query openai API
        # 4. Update jobs and batches in DB
        # 5. Print message with the batch summary
        # Repeat from 3. every minute if refresh = True

        with get_session() as db:
            batches = BatchRepository.get_batches(db, job_id)

            upd_batches = []
            for b in batches:
                api_batch = self._get_batch_api(b.batch_id)
                upd_batches.append(
                    BatchRepository.update_batch(
                        db, b.batch_id, BatchUpdate(map_openai_batch_to_dict(api_batch))
                    )
                )

            job = JobRepository.update_job_status(db, job_id)

            # TODO. Check CRUD operations work well
            # TODO.Print message with status summary

    def get_embeddings(self, job_id: int) -> str | None:
        # 1. self.check_status
        # 2. If job.status completed:
        #   2.1 Get output files
        #   2.2 Download files to local
        #   2.3 Parse files
        #   2.4 Create new file with all the data + embeddings
        #   2.5 Return file
        # 3. else: Print message and return None
        pass
