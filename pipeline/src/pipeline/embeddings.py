from concurrent.futures import ProcessPoolExecutor
from typing import Iterable, Optional
from dataclasses import dataclass
from functools import partial
from datetime import datetime
from decouple import config
from pathlib import Path
from tqdm import tqdm
import polars as pl
import tiktoken
import logging
import asyncio
import json
import os
import re

from openai.types import Batch
from openai import OpenAI

from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.console import Console, Group
from rich.logging import RichHandler
from rich.spinner import Spinner
from rich.columns import Columns
from rich.panel import Panel
from rich.table import Table
from rich.live import Live

from .schemas import JobRead, BatchCreate, BatchUpdate, BatchRead
from .db.crud import JobRepository, BatchRepository
from .db.models import JobStatus, BatchStatus
from .db.database import get_session

logging.basicConfig(
    level=logging.INFO, handlers=[RichHandler(rich_tracebacks=True, markup=True)]
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
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
    id: int
    input: str | list[str]
    model: str
    token_count: int

    def to_json_line(self, job_id: int, batch_id: str) -> str:
        """Serializes request into JSONL format required by the API"""
        payload = {
            "custom_id": f"batch-{str(job_id).zfill(4)}-{str(batch_id).zfill(5)}-request-{str(self.id).zfill(6)}",
            "method": "POST",
            "url": "/v1/embeddings",
            "body": {"model": self.model, "input": self.input},
        }
        return json.dumps(payload, ensure_ascii=False) + "\n"


@dataclass
class BatchRequest:
    id: int
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
    payload["batch_oaid"] = batch_request.id

    if batch_request.request_counts:
        counts_dict = {
            f"request_{k}": v
            for k, v in batch_request.request_counts.model_dump().items()
        }
        payload.update(counts_dict)

    return payload


class VectorEmbeddings:

    def encode_batch(
        self,
        file: str,
        text_column: str | list[str],
        model: str,
        max_input_tokens: int = MAX_INPUT_TOKENS,
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

        job = self._create_job(model)
        logger.info(f"New Job {job.id} created for model '{job.model}'")

        batch_files = self._batch_texts(job.id, texts, model, max_input_tokens)

        for batch_f in batch_files:
            self._post_batch_api(job.id, batch_f)

        return job.id

    def check_status(self, job_id: int, refresh: bool = False) -> None:
        """
        Displays a Table in the console with relevant info about all the
        batches linked to a job_id
        """

        job, batches = self._check_status_fetch_and_update(job_id)

        if not refresh:
            console.print(self._check_status_display(job, batches))
            return

        try:
            asyncio.run(self._check_status_async(job, batches))
        except KeyboardInterrupt:
            pass

    def get_embeddings(self, job_id: int) -> str | None:
        # 1. self.check_status
        # 2. If job.status completed:
        #   2.1 Get output files
        #   2.2 Download files to local
        #   2.3 Parse files NOTE Note that the output line order may not match the input line order.
        #       Instead of relying on order to process your results, use the custom_id field which will
        #       be present in each line of your output file and allow you to map requests in your input
        #       to results in your output (https://platform.openai.com/docs/guides/batch#:~:text=Note%20that%20the%20output%20line%20order%20may%20not%20match%20the%20input%20line%20order.%20Instead%20of%20relying%20on%20order%20to%20process%20your%20results%2C%20use%20the%20custom_id%20field%20which%20will%20be%20present%20in%20each%20line%20of%20your%20output%20file%20and%20allow%20you%20to%20map%20requests%20in%20your%20input%20to%20results%20in%20your%20output.)
        #   2.4 Create new file with all the data + embeddings
        #   2.5 Return file
        # 3. else: Print message and return None

        job, batches = self._check_status_fetch_and_update(job_id)

        if job.status != JobStatus.COMPLETED:
            return

        out_files = self._download_output_files(batches)

        self._parse_output_files(out_files)

        return

    def _post_batch_api(self, id: int, file: str) -> None:
        """Uploads the file and request batch embedding job to the API"""

        with OpenAI(
            base_url=config("OAI_URL"),
            api_key=config("OAI_API_KEY"),
        ) as client:

            file_upload = client.files.create(file=open(file, "rb"), purpose="batch")

            batch_request = client.batches.create(
                input_file_id=file_upload.id,
                endpoint="/v1/embeddings",
                completion_window="24h",
            )

            self._log_batch_request(id, batch_request, file)

    def _get_batch_api(self, batch_oaid: str) -> Batch:

        with OpenAI(
            base_url=config("OAI_URL"),
            api_key=config("OAI_API_KEY"),
        ) as client:

            return client.batches.retrieve(batch_oaid)

    def _create_batch_file(
        self,
        id: int,
        batch: BatchRequest,
        model: str,
        max_file_size: int = MAX_BATCH_FILE_SIZE,
    ) -> list[str]:
        """
        Creates multiple JSONL files that represent the batch requests.
        Each file has a maximum size allowed according to the API (OpenAI 200MB)
        """

        base_name = f"{model.split("/")[-1].lower()}-{str(id).zfill(4)}-batch-{str(batch.id).zfill(4)}.jsonl"
        base_path = f"./data/embeddings/{str(id).zfill(4)}/batches"
        file = Path(f"{base_path}/{base_name}")
        file.parent.mkdir(parents=True, exist_ok=True)

        files = []
        current_file_size = 0
        split_num = 1

        f = open(file, "w+", encoding="utf-8")

        try:
            for req in batch.input:
                line = req.to_json_line(id, batch.id)
                line_size = len(line.encode("utf-8"))

                if current_file_size + line_size > max_file_size:
                    f.close()
                    files.append(f.name)
                    logger.info(
                        f"[yellow]✨ Created file {f.name} for Batch #{batch.id}.[/]"
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
                    f"[yellow]✨ Created file {f.name} for Batch #{batch.id}.[/]"
                )

        return files

    def _generate_embedding_request(
        self, texts: list[str], model: str, counts: list[int], max_input_tokens: int
    ) -> Iterable[EmbeddingRequest]:
        """
        Generator function that create EmbeddingRequest objects according to the
        maximum limit of tokens allowed per request (TPR)
        """

        logger.info("Batching requests...")

        current_input, current_count = [], 0
        req_id = 1

        for text, count in zip(texts, counts):
            if current_count + count > max_input_tokens and current_input:
                yield EmbeddingRequest(
                    id=req_id,
                    input=current_input,
                    model=model,
                    token_count=current_count,
                )
                current_input, current_count = [], 0
                req_id += 1

            current_input.append(text)
            current_count += count

        if current_input:
            yield EmbeddingRequest(
                id=req_id, input=current_input, model=model, token_count=current_count
            )

    def _batch_texts(
        self, id: int, texts: list[str], model: str, max_input_tokens: int
    ) -> list[str]:
        """
        Takes a list of texts to embed and split them in multiple batches depending on
        the API limits: Tokens Per Request > Requests Per Batch > Batch File Size
        """

        console.print(
            Panel(
                f"[bold cyan]Model:[/bold cyan] {model}\n"
                f"[bold cyan]Total texts:[/bold cyan] {len(texts):,}\n"
                f"[bold cyan]Max tokens per request:[/bold cyan] {max_input_tokens:,}\n"
                f"[bold cyan]Max requests per batch:[/bold cyan] {MAX_REQUESTS_PER_BATCH:,}\n",
                title="[bold]Embedding Configuration[/bold]",
                border_style="cyan",
            )
        )

        logger.info(f"Pre-tokenizing texts for Rate Limits...")

        # Pre-tokenization (parallel workers)
        token_counts = self._tokenize_texts(texts, model)

        # Process texts into request objects
        requests = list(
            self._generate_embedding_request(
                texts, model, token_counts, max_input_tokens
            )
        )

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
                model,
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

    def _tokenize_texts(
        self, texts: list[str], model: str, max_workers: int = None
    ) -> list[int]:
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
            tiktoken.encoding_for_model(model)
            model_name = model
        except:
            logger.warning(
                f"Could not find a tokenizer in tiktoken for model {model}. Defaulting to `cl100k_base` tokenizer. Token count is approximated!"
            )
            # NOTE. Can use AutoTokenizer from HF but there's no need for exact matching tokens (only token count). HF very slow to download models.
            # encoding = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            model_name = "cl100k_base"

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

    def _create_job(self, model: str) -> JobRead:
        """Creates a new Job in DB"""

        with get_session() as db:
            job = JobRepository.create_job(db, model)
        return job

    def _check_status_display(
        self,
        job: JobRead,
        batches: list[BatchRead],
        upd_time: Optional[datetime] = None,
    ):
        status_colors = {
            JobStatus.PENDING: "cyan",
            JobStatus.PROCESSING: "yellow",
            JobStatus.COMPLETED: "green",
            JobStatus.FAILED: "red",
            BatchStatus.COMPLETED: "green",
            BatchStatus.VALIDATING: "cyan",
            BatchStatus.IN_PROGRESS: "yellow",
            BatchStatus.FAILED: "red",
            BatchStatus.EXPIRED: "magenta",
            BatchStatus.CANCELLED: "grey50",
        }

        job_color = status_colors.get(job.status, "white")

        table = Table(title=f"Batches for Job {job.id}", box=None, show_edge=False)
        table.add_column("ID", style="cyan")
        table.add_column("API ID", style="cyan")
        table.add_column("Status")
        table.add_column("Completed", style="green")
        table.add_column("Failed", style="magenta")
        table.add_column("Total", style="cyan")
        table.add_column("Errors", style="magenta")
        table.add_column("Created (UTC)", style="yellow")
        table.add_column("Updated (UTC)", style="yellow")

        for b in batches:
            b_color = status_colors.get(b.status, "white")

            table.add_row(
                str(b.id),
                b.batch_oaid,
                f"[{b_color}]{b.status.value}[/]",
                str(b.request_completed),
                str(b.request_failed),
                str(b.request_total),
                b.errors or "Null",
                b.created_at.strftime("%d/%m/%Y, %H:%M:%S"),
                b.calculated_update_at.strftime("%d/%m/%Y, %H:%M:%S"),
            )

        refresh_info = ""
        if upd_time is not None:
            refresh_info = Columns(
                [
                    f"[dim]Updating every 1 min. Last update: {upd_time.strftime('%H:%M:%S')}[/] ",
                    Spinner("point", style="yellow", speed=0.5),
                ]
            )

        return Panel(
            Group(
                f"Job Status: [bold {job_color}]{job.status.value.upper()}[/]",
                refresh_info,
                "",
                table,
                "",
                "[dim italic]Press Enter to exit...[/]",
            ),
            title="[bold]Job Monitor[/]",
            border_style="cyan",
            expand=False,
        )

    def _check_status_fetch_and_update(
        self, job_id: int
    ) -> tuple[JobRead, list[BatchRead]]:
        logger.info(f"Updating Batch statuses for [yellow]Job {job_id}[/]")

        with get_session() as db:
            batches = BatchRepository.get_batches(db, job_id)
            upd_batches = []
            for b in batches:
                api_batch = self._get_batch_api(b.batch_oaid)
                upd_data = BatchUpdate.model_validate(
                    map_openai_batch_to_dict(api_batch)
                )
                upd_batches.append(
                    BatchRepository.update_batch(
                        db,
                        b.id,
                        upd_data,
                    )
                )

            job = JobRepository.update_job_status(db, job_id)
            return job, upd_batches

    async def _check_status_monitor(self, job_id: int, live: Live) -> None:
        while True:
            await asyncio.sleep(60)
            job, upd_batches = self._check_status_fetch_and_update(job_id)
            live.update(self._check_status_display(job, upd_batches, datetime.now()))

    async def _check_status_async(self, job: JobRead, batches: list[BatchRead]):
        with Live(
            self._check_status_display(job, batches, datetime.now()),
            refresh_per_second=10,
        ) as live:
            monitor_task = asyncio.create_task(self._check_status_monitor(job.id, live))
            exit_task = asyncio.create_task(asyncio.to_thread(input))

            await asyncio.wait(
                [monitor_task, exit_task], return_when=asyncio.FIRST_COMPLETED
            )
            monitor_task.cancel()

    def _download_output_files(self, batches: list[BatchRead]) -> list[Path]:
        progress = Progress(
            TextColumn("🛻💨"),
            SpinnerColumn("point"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        )

        logger.info("Downloading Output files!")

        with progress:
            task = progress.add_task("download", total=len(batches))

            with OpenAI(
                base_url=config("OAI_URL"), api_key=config("OAI_API_KEY")
            ) as client:

                out_files = []

                for b in batches:
                    fout = b.output_file_id
                    localpath = Path(b.local_file_id)

                    fnew = localpath.parent / f"{localpath.stem}_result.jsonl"
                    fnew.parent.mkdir(parents=True, exist_ok=True)

                    content = client.files.content(fout)
                    fnew.write_text(content.text, encoding="utf-8")

                    out_files.append(fnew)
                    progress.advance(task)  # Update progress bar

        return out_files

    def _parse_output_files(self, files: list[Path]):
        for f in files:
            map_sorted = self._sort_output_file(f)
            self._stream_file_to_sorted_numpy(f, map_sorted)

    def _sort_output_file(self, file: Path) -> list[tuple[str, int]]:
        id_pattern = re.compile(rb'"custom_id"\s*:\s*"([^"]+)"')
        line_map = []  # Contains the custom_id and start line position

        with file.open("rb") as f:
            while True:
                offset = f.tell()
                chunk = f.read(1024)  # read only start of line

                if not chunk:
                    break

                match = id_pattern.search(chunk)
                if match:
                    line_map.append((match.group(1).decode("utf-8"), offset))

                f.seek(offset)
                f.readline()

        line_map.sort(key=lambda x: x[0])

        input(line_map)

        return line_map

    def _stream_file_to_sorted_numpy(self, f: Path, map_sorted: list[tuple[str, int]]):
        pass


Embeddings = VectorEmbeddings()
