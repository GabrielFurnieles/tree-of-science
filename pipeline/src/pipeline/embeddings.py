from tenacity import (
    retry,
    wait_exponential,
    stop_after_attempt,
    retry_if_exception_type,
    RetryError,
    RetryCallState,
)
from openai import AsyncOpenAI, OpenAI, RateLimitError, APITimeoutError
from openai.types import CreateEmbeddingResponse
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
import asyncio
import logging
import string
import secrets
import json
import os
from typing import Generator

from .utils import AsyncRateLimiter
from .schemas import BatchCreate
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


@dataclass
class BatchRequest:
    batch_id: int
    input: list[EmbeddingRequest]
    text_count: int
    token_count: int


# Utility functions
def tokenize_batch(texts_chunk: list, model_name: str) -> list[int]:
    """Worker function to tokenize a chunk of texts."""
    try:
        encoding = tiktoken.encoding_for_model(model_name)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    return [len(encoding.encode(t, allowed_special="all")) for t in texts_chunk]


def chunk_list(lst: list, chunk_size: int) -> Generator[list[str]]:
    """Split list into chunks of specified size."""
    for i in range(0, len(lst), chunk_size):
        yield lst[i : i + chunk_size]


def log_retry_attempt(retry_state: RetryCallState):
    """Log retry attempts with relevant info"""
    exception = retry_state.outcome.exception()
    batch = (
        retry_state.args[1]
        if len(retry_state.args) > 1
        else retry_state.kwargs.get("batch")
    )
    wait_time = retry_state.next_action.sleep if retry_state.next_action else 0

    logger.warning(
        f"Batch {getattr(batch, 'batch_number', 'unknown')} with {len(batch.texts)} texts failed: {exception}. "
        f"Retrying in {wait_time:.2f} seconds... (Attempt {retry_state.attempt_number}/5)"
    )


class VectorEmbeddings:
    def __init__(
        self,
        model: str,
        max_input_tokens: int = MAX_INPUT_TOKENS,
        max_requests_per_minute: int = MAX_REQUESTS_PER_MINUTE,
        max_tokens_per_minute: int = MAX_TOKENS_PER_MINUTE,
    ):
        self.model = model
        self.max_input_tokens = max_input_tokens

        self.request_limiter = AsyncRateLimiter(max_requests_per_minute, time_period=60)
        self.tokens_limiter = AsyncRateLimiter(max_tokens_per_minute, time_period=60)

    @retry(
        retry=retry_if_exception_type((RateLimitError, APITimeoutError)),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        stop=stop_after_attempt(5),
        before_sleep=log_retry_attempt,
    )
    async def get_embeddings_api(
        self,
        req: EmbeddingRequest,
        client: AsyncOpenAI,
        semaphore: asyncio.Semaphore,
    ) -> tuple[int, CreateEmbeddingResponse]:

        await self.tokens_limiter.acquire(req.token_count)

        async with self.request_limiter:

            async with semaphore:

                logger.info(
                    f"[yellow][Request {req.request_id}] Requesting embeddings for {len(req.input) if isinstance(req.input, list) else 1} texts with {self.model} model...[/]"
                )

                try:
                    response = await client.embeddings.create(
                        input=req.input, model=self.model, encoding_format="float"
                    )

                except RetryError as e:
                    logger.error(
                        f"All retries exhausted for batch {req.request_id} "
                        f"with {len(req.input) if isinstance(req.input, list) else 1} texts. Final error: {e.last_attempt.exception()}"
                    )
                    response = e.last_attempt.exception()

                return (len(req.input) if isinstance(req.input, list) else 1, response)

    def post_embeddings_batch_api(self, id: int, files: list[str]) -> str:
        logger.info(f"POST v1/embeddings for files: {files}")

        datestr = datetime.now().strftime("%Y%m%d-%H%M")
        outputfile = Path(
            f"./data/embeddings/{id}/embedding-requests-{id}-{datestr}.jsonl"
        )
        outputfile.parent.mkdir(parents=True, exist_ok=True)

        with open(outputfile, "a+", encoding="utf-8") as fout:
            with OpenAI(
                base_url="https://api.tokenfactory.nebius.com/v1/",
                api_key=config("NEBIUS_API_KEY"),
            ) as client:

                for f in files:
                    file_upload = client.files.create(
                        file=open(f, "rb"), purpose="batch"
                    )

                    batch_request = client.batches.create(
                        input_file_id=file_upload.id,
                        endpoint="/v1/embeddings",
                        completion_window="24h",
                        metadata={"description": f},
                    )

                    d = batch_request.__dict__.copy()
                    d["batch_id"] = d.pop("id")
                    d.update(
                        {
                            f"request_{k}": v
                            for k, v in d.pop("request_counts").__dict__.items()
                        }
                    )

                    batch_db = BatchCreate(job_id=id, local_file_id=f, **d)

                    with get_session() as db:
                        BatchRepository.create_batch(db, batch_db)

                    logger.info(
                        f"Created batch request at https://api.tokenfactory.nebius.com/v1/ for {f}"
                    )

                    d = batch_request.__dict__.copy()
                    d["request_counts"] = d["request_counts"].__dict__
                    fout.write(json.dumps(d) + "\n")

        return outputfile

    def create_batch_file(
        self, id: int, batch: BatchRequest, max_file_size: int = MAX_BATCH_FILE_SIZE
    ) -> list[str]:
        datestr = datetime.now().strftime("%Y%m%d-%H%M")
        file = Path(
            f"./data/embeddings/{id}/batches/{self.model.split("/")[-1].lower()}-{id}-{datestr}-batch-{batch.batch_id}.jsonl"
        )
        file.parent.mkdir(parents=True, exist_ok=True)
        file_size = 0

        files = []
        file_handle = open(file, "w+", encoding="utf-8")
        split_num = 2

        subbatch_req_count = 0
        subbatch_text_count = 0
        subbatch_token_count = 0

        logger.info(
            f"Creating file/s for Batch #{batch.batch_id} with {len(batch.input)} requests... Total texts = {batch.text_count} tokens = {batch.token_count}."
        )

        for req in batch.input:
            request = {
                "custom_id": f"batch-{id}-{datestr}_{batch.batch_id}-request-{req.request_id}",
                "method": "POST",
                "url": "/v1/embeddings",
                "body": {"model": self.model, "input": req.input},
            }

            line = json.dumps(request, ensure_ascii=False) + "\n"
            line_size = len(line.encode("utf-8"))

            if file_size + line_size > max_file_size:
                if file_handle:
                    files.append(file_handle.name)
                    file_handle.close()

                    logger.info(
                        f"[yellow]✨ Created file {file_handle.name} for Batch #{batch.batch_id} with [bold]{subbatch_req_count}[/bold] requests...[/yellow] Total texts = {subbatch_text_count} tokens = {subbatch_token_count}."
                    )

                    subbatch_req_count = 0
                    subbatch_text_count = 0
                    subbatch_token_count = 0

                file_split = file.parent / f"{file.stem}_{split_num}.jsonl"
                file_handle = open(file_split, "w+", encoding="utf-8")

                file_size = 0
                split_num += 1

            file_handle.write(line)
            file_size += line_size

            subbatch_req_count += 1
            subbatch_text_count += len(req.input)
            subbatch_token_count += req.token_count

        if file_handle:
            files.append(file_handle.name)
            file_handle.close()
            logger.info(
                f"[yellow]✨ Created file {file_handle.name} for Batch #{batch.batch_id} with [bold]{subbatch_req_count}[/bold] requests...[/yellow] Total texts = {subbatch_text_count} tokens = {subbatch_token_count}."
            )

        return files

    async def compute_embeddings_async(
        self, texts: list[str], max_concurrent_requests: int
    ) -> list[list[float]]:
        semaphore = asyncio.Semaphore(max_concurrent_requests)

        try:
            encoding = tiktoken.encoding_for_model(self.model)
        except:
            encoding - tiktoken.get_encoding("cl100k_base")
            # encoding = AutoTokenizer.from_pretrained(self.model, trust_remote_code=True)

        console.print(
            Panel(
                f"[bold cyan]Model:[/bold cyan] {self.model}\n"
                f"[bold cyan]Total texts:[/bold cyan] {len(texts):,}\n"
                f"[bold cyan]Max concurrent requests:[/bold cyan] {max_concurrent_requests}\n"
                f"[bold cyan]Max tokens per request:[/bold cyan] {self.max_input_tokens:,}\n"
                f"[bold cyan]Max requests per minute (RPM):[/bold cyan] {int(self.request_limiter.max_rate)}\n"
                f"[bold cyan]Max tokens per minute (TPM):[/bold cyan] {int(self.tokens_limiter.max_rate)}\n",
                title="[bold]Embedding Configuration[/bold]",
                border_style="cyan",
            )
        )

        async with AsyncOpenAI(
            api_key=config("OPENAI_API_KEY"), max_retries=0
        ) as client:
            tasks = []
            token_count = 0
            model_input = []

            for i, t in enumerate(texts, start=1):
                n_tokens = self._token_len(t, encoding)
                token_count += n_tokens

                if token_count > self.max_input_tokens:
                    logger.info(
                        f"[yellow]Processing [bold]{i}/{len(texts)}[/bold] texts...[/yellow] Total texts = {len(model_input)} tokens = {token_count - n_tokens}"
                    )
                    tasks.append(
                        self.get_embeddings_api(
                            EmbeddingRequest(
                                request_id=len(tasks) + 1,
                                input=model_input,
                                token_count=token_count - n_tokens,
                            ),
                            client,
                            semaphore,
                        )
                    )

                    # Reset batch
                    token_count = n_tokens
                    model_input = [t]

                else:
                    model_input.append(t)

            if len(model_input):
                tasks.append(
                    self.get_embeddings_api(
                        EmbeddingRequest(len(tasks) + 1, model_input, token_count),
                        client,
                        semaphore,
                    )
                )
                console.print(
                    f"[yellow]Processing [bold]{i}/{len(texts)}[/bold] texts...[yellow] Total texts = {len(model_input)} tokens = {token_count}"
                )

            results: tuple[int, CreateEmbeddingResponse] = await asyncio.gather(
                *tasks, return_exceptions=True
            )

        embeddings = []
        for i, response in enumerate(results):
            n, embs = response
            if isinstance(embs, Exception):
                logging.error(
                    f"❌ Batch [{i}/{len(results)}]. {n} texts failed with error: {embs}"
                )
                result = [-1 for _ in range(n)]  # Expand error to the affected records
            else:
                result = [e.embedding for e in embs.data]  # unpack embeddings
            embeddings.extend(result)

        return embeddings

    def compute_embeddings_batch(self, id: int, texts: list[str]) -> str:
        token_count = 0
        request_input = []
        batch_id = 1
        batch_input = []
        batch_text_count = 0
        batch_token_count = 0

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

        # Pre-tokenization
        token_counts = self.parallel_tokenize(texts)

        logger.info(f"Batching requests...")

        for t, n_tokens in zip(texts, token_counts):
            token_count += n_tokens

            if token_count > self.max_input_tokens:
                batch_input.append(
                    EmbeddingRequest(
                        request_id=len(batch_input) + 1,
                        input=request_input,
                        token_count=token_count - n_tokens,
                    )
                )
                batch_text_count += len(request_input)
                batch_token_count += token_count - n_tokens

                # Reset request
                token_count = n_tokens
                request_input = [t]

            else:
                request_input.append(t)

            if len(batch_input) >= MAX_REQUESTS_PER_BATCH:
                batch_files = self.create_batch_file(
                    id,
                    BatchRequest(
                        batch_id, batch_input, batch_text_count, batch_token_count
                    ),
                )
                self.post_embeddings_batch_api(id, batch_files)

                # Reset batch
                batch_id += 1
                batch_input = []
                batch_text_count = 0
                batch_token_count = 0

        if len(request_input) > 0:
            batch_input.append(
                EmbeddingRequest(
                    request_id=len(batch_input) + 1,
                    input=request_input,
                    token_count=token_count,
                )
            )
            batch_text_count += len(request_input)
            batch_token_count += token_count

        if len(batch_input) > 0:
            batch_files = self.create_batch_file(
                id,
                BatchRequest(
                    batch_id, batch_input, batch_text_count, batch_token_count
                ),
            )
            self.post_embeddings_batch_api(id, batch_files)

    def get_embeddings(
        self,
        file: str,
        text_column: str | list[str],
        max_concurrent_requests: int = 5,
    ):
        file = Path(file)

        assert (
            file.suffix == ".parquet"
        ), f"Expected 'file' to be a .parquet file, instead got {file}"
        assert file.exists(), f"Couldn't find the parquet file {file}"

        logging.info(
            f"Reading data from {file} 🔎\nThis may take a few seconds depending on the size of data..."
        )

        df = pl.read_parquet(file).head(10_000)  # TODO. Remove .head() for production

        if isinstance(text_column, list):
            texts = (
                df.select(pl.concat_str(text_column, separator="\n\n"))
                .to_series()
                .to_list()
            )
        else:
            texts = df[text_column].to_list()

        emb: list[list[float]] = asyncio.run(
            self.compute_embeddings_async(texts, max_concurrent_requests)
        )

        df_emb = df.with_columns(pl.Series(name="embedding", values=emb))

        console.print("\n[bold]✨ Embeddings Completed[/bold]")

        return df_emb

    def post_embeddings_batch(
        self,
        file: str,
        text_column: str | list[str],
    ):
        file = Path(file)

        assert (
            file.suffix == ".parquet"
        ), f"Expected 'file' to be a .parquet file, instead got {file}"
        assert file.exists(), f"Couldn't find the parquet file {file}"

        logging.info(
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

        logger.info(f"Found {len(texts)} texts!")
        job = self._create_job()
        logger.info(f"New Embedding Job {job.id} created for model{job.model}")

        self.compute_embeddings_batch(job.id, texts)

        return job.id

    def get_embeddings_batch(self):
        pass

    def parallel_tokenize(self, texts: list[str], max_workers: int = None) -> list[int]:
        """
        Tokenize texts in parallel using multiprocessing.

        Args:
            texts: List of texts to tokenize
            model_name: Model name for tiktoken encoding
            max_workers: Number of worker processes (default: CPU count)

        Returns:
            List of token counts for each text
        """
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
            worker_fn = partial(tokenize_batch, model_name=model_name)
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

    def _create_job(self):
        with get_session() as db:
            job = JobRepository.create_job(db, self.model)
        return job

    def _generate_id(self, length: int = 8):
        alphabet = string.ascii_letters + string.digits
        return "".join(secrets.choice(alphabet) for _ in range(length))
