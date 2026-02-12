from concurrent.futures import ProcessPoolExecutor
from typing import Iterable, Optional, Callable, TYPE_CHECKING
from dataclasses import dataclass
from openai.types import Batch
from functools import partial
from datetime import datetime
from pathlib import Path
import polars as pl
import numpy as np
import tiktoken
import asyncio
from tqdm import tqdm
import mmap
import json
import os
import re

from rich.logging import RichHandler
from rich.console import Console, Group
from rich.spinner import Spinner
from rich.columns import Columns
from rich.panel import Panel
from rich.table import Table
from rich.live import Live
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeRemainingColumn,
)
import logging


from .schemas import BatchCreate, BatchRead, BatchUpdate, JobRead
from .db.crud import BatchRepository, JobRepository
from .db.models import JobStatus, BatchStatus
from .db.database import get_session

# Type import that avoids circular import
if TYPE_CHECKING:
    from .embeddings import EmbeddingConfig

logging.basicConfig(
    level=logging.INFO, handlers=[RichHandler(rich_tracebacks=True, markup=True)]
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

console = Console()


# Dataclasses
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


class BatchManager:
    def __init__(self, config: "EmbeddingConfig"):
        self.cfg = config
        self.model = config.model

    def parse_texts(self, file: str, text_column: str | list[str]) -> list[str]:
        """
        Read texts from the .parquet file using polars and select columns
        to create the final texts to embed.
        """
        f = Path(file)

        assert (
            f.suffix == ".parquet"
        ), f"Expected 'file' to be a .parquet file, instead got {file}"
        assert f.exists(), f"Couldn't find the parquet file {file}"

        logger.info(
            f"Reading data from {file} 🔎\nThis may take a few seconds depending on the file size..."
        )

        df = pl.read_parquet(file)

        if isinstance(text_column, list):
            texts = (
                df.select(pl.concat_str(text_column, separator="\n\n"))
                .to_series()
                .to_list()
            )
        else:
            texts = df[text_column].to_list()

        logger.info(f"Found {len(texts)} texts!")

        return texts

    def batch_texts(self, job_id: int, texts: list[str]) -> list[str]:
        """
        Takes a list of texts to embed and split them in multiple batches depending on
        the API limits: Tokens Per Request > Requests Per Batch > Batch File Size
        """

        console.print(
            Panel(
                f"[bold cyan]Model:[/bold cyan] {self.model}\n"
                f"[bold cyan]Total texts:[/bold cyan] {len(texts):,}\n"
                f"[bold cyan]Max tokens per request:[/bold cyan] {self.cfg.max_input_tokens:,}\n"
                f"[bold cyan]Max requests per batch:[/bold cyan] {self.cfg.max_requests_per_batch:,}\n",
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
                job_id,
                BatchRequest(
                    i,
                    batch,
                    sum(len(r.input) for r in batch),
                    sum(r.token_count for r in batch),
                ),
            )
            for i, batch in enumerate(
                chunk_list(requests, self.cfg.max_requests_per_batch), start=1
            )
        ]

        return [f for batch_files in request_files for f in batch_files]

    def log_batch_request(self, job_id: int, batch_request: Batch, local_file: str):
        """
        Takes a Batch request object from the OpenAI API and creates a new record in the local DB.
        It leverages that Batch is a pydantic model.
        """

        payload = map_openai_batch_to_dict(batch_request)
        batch_db = BatchCreate(job_id=job_id, local_file_id=local_file, **payload)

        with get_session() as db:
            batch = BatchRepository.create_batch(db, batch_db)

        logger.info(f"Created batch request {batch}")

    def parse_output_files(self, files: list[Path]) -> Path:
        """
        Parses the OAI Batch API response files containing the mebddings
        into a single npy file.

        NOTE. This method doesn't know about the total number of vector embeddings
        requested accross all batches,having to stream the embeddings first into
        a .raw file and later copying them into a .npy file where the shape is
        pre-fixed.
        """

        out_raw = files[0].parents[1] / f'{"-".join(files[0].stem.split("-")[:-2])}.raw'
        out_npy = out_raw.with_suffix(".npy")

        vec_count = 0
        for f in files:
            map_sorted = self._sort_output_file(f)
            vec_count += self._stream_embs_to_sorted_raw(f, map_sorted, out_raw)

        out_file = self._stream_raw_to_numpy(
            fraw=out_raw, dims=(vec_count, self.cfg.embedding_dim), fout=out_npy
        )

        os.remove(out_raw)  # delete raw file
        return out_file

    def _create_batch_file(
        self,
        job_id: int,
        batch: BatchRequest,
    ) -> list[str]:
        """
        Creates multiple JSONL files that represent the batch requests.
        Each file has a maximum size allowed according to the API (OpenAI 200MB)
        """

        base_name = f"{self.model.split("/")[-1].lower()}-{str(job_id).zfill(4)}-batch-{str(batch.id).zfill(4)}.jsonl"
        base_path = Path(f"{self.cfg.output_path}/{str(job_id).zfill(4)}/batches")
        base_path.mkdir(parents=True, exist_ok=True)

        files = []
        current_file_size = 0
        split_num = 0

        def get_path(s_num: int):
            suffix = f"_{s_num}" if s_num > 0 else ""
            return base_path / f"{base_name}{suffix}.jsonl"

        current_path = get_path(split_num)
        f = open(current_path, "w", encoding="utf-8")

        try:
            for req in batch.input:
                line = req.to_json_line(job_id, batch.id)
                line_size = len(line.encode("utf-8"))

                if current_file_size + line_size > self.cfg.max_batch_file_size:
                    f.close()
                    files.append(f.name)

                    split_num += 1
                    current_path = get_path(split_num)

                    f = open(current_path)
                    current_file_size = 0

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
            if current_count + count > self.cfg.max_input_tokens and current_input:
                yield EmbeddingRequest(
                    id=req_id,
                    input=current_input,
                    model=self.model,
                    token_count=current_count,
                )
                current_input, current_count = [], 0
                req_id += 1

            current_input.append(text)
            current_count += count

        if current_input:
            yield EmbeddingRequest(
                id=req_id,
                input=current_input,
                model=self.model,
                token_count=current_count,
            )

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

        worker_fn = partial(tokenize, model_name=model_name)

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
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

    def _sort_output_file(self, file: Path) -> list[tuple[str, int]]:
        """
        Takes an Embedding OAI Batch API response file and sorts its responses
        based on the "custom_id" field as instructed by the API docs (https://platform.openai.com/docs/guides/batch#:~:text=Note%20that%20the%20output%20line%20order%20may%20not%20match%20the%20input%20line%20order.%20Instead%20of%20relying%20on%20order%20to%20process%20your%20results%2C%20use%20the%20custom_id%20field%20which%20will%20be%20present%20in%20each%20line%20of%20your%20output%20file%20and%20allow%20you%20to%20map%20requests%20in%20your%20input%20to%20results%20in%20your%20output.)

        Args:
            file: Path to the Batch API response file

        Returns:
            A list of tuples (custom_id, offset) organised by "custom_id". The offset
            represents the line position where the response with that ccustom_id starts.

        NOTE. Instead of explicitly sorting the file this function returns a sorted
        map object that maps the custom_id with the corresponding line in the file,
        avoiding extra computations.
        """

        id_pattern = re.compile(rb'"custom_id"\s*:\s*"([^"]+)"')
        line_map = []  # Contains the custom_id and start line position

        with file.open("rb") as f:
            while True:
                offset = f.tell()
                chunk = f.read(500)  # read only start of line

                if not chunk:
                    break

                match = id_pattern.search(chunk)
                if match:
                    line_map.append((match.group(1).decode("utf-8"), offset))

                f.seek(offset)
                f.readline()

        line_map.sort(key=lambda x: x[0])
        return line_map

    def _stream_embs_to_sorted_raw(
        self, file: Path, map_sorted: list[tuple[str, int]], output_raw: Path
    ) -> int:
        """
        Streams embeddings from the OAI Batch response file to a .raw file in the
        order they were requested.

        Args:
            file: Input Path OAI Batch response file
            map_sorted: Ordered list of tuples containing (custom_id, offset) where
                custom_id is the sorting key and offset the position of the line where
                the response for that custom_id start in the Batch response file.
            output_raw: Output Path to the .raw file (shared accross multiple files)

        Returns:
            The total count of vectors from the processed file
        """

        VECTOR_BATCH_SIZE = 1000
        vec_count = 0
        vec_batch = []

        # Define the rich progress bar columns
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
        )

        with progress:
            task_id = progress.add_task(f"Streaming {file.name}", total=len(map_sorted))

            with open(file, "rb") as f_in:
                # Use mmap to random access file faster
                with mmap.mmap(f_in.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                    with open(output_raw, "ab+") as f_out:
                        for _, offset in map_sorted:
                            mm.seek(offset)
                            line = mm.readline()

                            progress.update(task_id, advance=1)

                            if not line:
                                continue

                            try:
                                data = json.loads(line)
                                items = data["response"]["data"]

                                for i in items:
                                    vec_batch.append(i["embedding"])

                                    if len(vec_batch) >= VECTOR_BATCH_SIZE:
                                        arr = np.array(vec_batch, dtype="float32")
                                        f_out.write(arr.tobytes())
                                        vec_count += len(vec_batch)
                                        vec_batch = []

                            except Exception as e:
                                # Use rich's print to avoid breaking the progress bar layout
                                progress.console.print(
                                    f"[red]Error parsing line from {file.name} at offset: {offset} -> {e}[/red]"
                                )

                        # Final flush
                        if vec_batch:
                            arr = np.array(vec_batch, dtype="float32")
                            f_out.write(arr.tobytes())
                            vec_count += len(vec_batch)

        return vec_count

    def _stream_raw_to_numpy(self, fraw: Path, dims: tuple[int], fout: Path) -> Path:
        """
        Takes a .raw file containing the vector embeddings and streams
        it into a .npy file with pre-fixed shape. This way embeddings
        are stored in a more propper and efficient format, as .npy  files
        contain metadata about the shape and data type.
        """

        npy = np.lib.format.open_memmap(fout, mode="w+", dtype="float32", shape=dims)

        # Stream raw file in chunks of 100MB
        chunk_size = 100 * 1024 * 1024
        rows_per_chunk = chunk_size // (dims[1] * 4)  # float32 = 4 bytes

        with open(fraw, "rb") as f:
            for start_row in range(0, dims[0], rows_per_chunk):
                end_row = min(start_row + rows_per_chunk, dims[0])
                bytes_to_read = (end_row - start_row) * dims[1] * 4
                raw_chunk = f.read(bytes_to_read)
                array_chunk = np.frombuffer(raw_chunk, dtype="float32").reshape(
                    end_row - start_row, dims[1]
                )
                npy[start_row:end_row] = array_chunk

        npy.flush()
        del npy  # delete reference (mmap objects can sometimes keep a file "busy" even after the function ends)

        return fout


class BatchMonitor:
    def __init__(self, job_id: int, fetch_callback: Callable[[str], Batch]):
        self.job_id = job_id
        self.fetch_callback = fetch_callback

    def check_status_fetch_and_update(self) -> tuple[JobRead, list[BatchRead]]:
        logger.info(f"Updating Batch statuses for [yellow]Job {self.job_id}[/]")

        with get_session() as db:
            batches = BatchRepository.get_batches(db, self.job_id)
            upd_batches = []
            for b in batches:
                api_batch = self.fetch_callback(b.batch_oaid)
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

            job = JobRepository.update_job_status(db, self.job_id)
            return job, upd_batches

    def check_status_display(
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

    async def check_status_async(self, job: JobRead, batches: list[BatchRead]):
        with Live(
            self.check_status_display(job, batches, datetime.now()),
            refresh_per_second=10,
        ) as live:
            monitor_task = asyncio.create_task(self._check_status_monitor(job.id, live))
            exit_task = asyncio.create_task(asyncio.to_thread(input))

            await asyncio.wait(
                [monitor_task, exit_task], return_when=asyncio.FIRST_COMPLETED
            )
            monitor_task.cancel()

    async def _check_status_monitor(self, job_id: int, live: Live) -> None:
        while True:
            await asyncio.sleep(60)
            job, upd_batches = await asyncio.to_thread(
                self.check_status_fetch_and_update
            )
            live.update(self.check_status_display(job, upd_batches, datetime.now()))
