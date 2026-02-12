from dataclasses import dataclass, field
from decouple import config
from pathlib import Path
import numpy as np
import logging
import asyncio
import json
import mmap
import os
import re

from openai.types import Batch
from openai import OpenAI

from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeRemainingColumn,
)
from rich.console import Console
from rich.logging import RichHandler


from .utils import BatchManager, BatchMonitor
from .schemas import JobRead, BatchRead
from .db.crud import JobRepository
from .db.models import JobStatus
from .db.database import get_session

logging.basicConfig(
    level=logging.INFO, handlers=[RichHandler(rich_tracebacks=True, markup=True)]
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

console = Console()

MODEL_SPEC = {"Qwen/Qwen3-Embedding-8B": 4096}


@dataclass
class EmbeddingConfig:
    model: str
    embedding_dim: int = field(init=False)
    max_input_tokens: int = (
        1_000  # 200_000  # Max input tokens across all inputs in a single request. According to OpenAI docs is 300_000 but OpenAI counts different than tiktoken
    )
    max_requests_per_batch: int = (
        100  # 50_000  # According to OpenAI batch inference API docs
    )
    max_batch_file_size: int = 200 * 1024 * 1024  # 200MB
    output_path: str = "./data/embeddings"

    def __post_init__(self):
        if self.model in MODEL_SPEC:
            self.embedding_dim = MODEL_SPEC[self.model]
        else:
            raise ValueError(f"Unknown model {self.model}. Check the supported models.")


class VectorEmbeddings:
    def __init__(self, model: str = None, **overrides):

        self.config = None
        self.manager = None
        self.overrides = overrides

        if model is not None:
            self._set_up_manager(model)

        self.client = OpenAI(
            base_url=config("OAI_URL"),
            api_key=config("OAI_API_KEY"),
        )

    def encode_batch(self, file: str, text_column: str | list[str]):
        """
        Chunks texts from a Parquet file and submits them for batch embedding.

        Args:
            file: Path to the Parquet file containing the data.
            text_column: A single column name or a list of column names to embed.
                Multiple columns are concatenated with double newlines ('\n\n').

        Returns:
            A list of paths to the generated batch request files.
        """
        texts = self.manager.parse_texts(file, text_column)
        job = self._create_job(self.config.model)
        batch_files = self.manager.batch_texts(job.id, texts)

        for batch_f in batch_files:
            self._post_batch_api(job.id, batch_f)

        return job.id

    def check_status(self, job_id: int, refresh: bool = False) -> None:
        """
        Displays a Table in the console with relevant info about all the
        batches linked to a job_id

        Args:
            job_id: Id of the checking status Job
            refresh: Wether to keep refreshing the table and displaying it
                in the console every minute
        """
        monitor = BatchMonitor(job_id, fetch_callback=self._get_batch_api)

        job, batches = monitor.check_status_fetch_and_update()

        if job is None:
            logger.warning("Skipping Job monitoring")
            return

        if not refresh:
            console.print(monitor.check_status_display(job, batches))
            return

        try:
            asyncio.run(monitor.check_status_async(job, batches))
        except KeyboardInterrupt:
            pass

    def get_embeddings(self, job_id: int) -> str | None:
        """
        Download and parse the embeddings for a specific Job (only if all
        the batch requests have been completed)

        Args:
            job_id: Id of the Job

        Returns:
            The Path to the final file cotaining all the embeddings
        """
        monitor = BatchMonitor(job_id, fetch_callback=self._get_batch_api)
        job, batches = monitor.check_status_fetch_and_update()

        if job.status != JobStatus.COMPLETED:
            return

        model = self._fetch_model(job_id)
        if model is not None:
            self._set_up_manager(model)
        else:
            return

        out_files = self._download_output_files(batches)

        embs_file = self.manager.parse_output_files(out_files)

        return embs_file

    def _set_up_manager(self, model: str):
        self.config = EmbeddingConfig(model=model, **self.overrides)
        self.manager = BatchManager(self.config)

    def _post_batch_api(self, id: int, file: str) -> None:
        """Uploads the file and request batch embedding job to the API"""
        file_upload = self.client.files.create(file=open(file, "rb"), purpose="batch")

        batch_request = self.client.batches.create(
            input_file_id=file_upload.id,
            endpoint="/v1/embeddings",
            completion_window="24h",
        )

        self.manager.log_batch_request(id, batch_request, file)

    def _get_batch_api(self, batch_oaid: str) -> Batch:
        return self.client.batches.retrieve(batch_oaid)

    def _create_job(self, model: str) -> JobRead:
        """Creates a new Job in DB"""

        with get_session() as db:
            job = JobRepository.create_job(db, model)

        logger.info(f"New Job {job.id} created for model '{job.model}'")
        return job

    def _fetch_model(self, job_id: int) -> str:
        with get_session() as db:
            job = JobRepository.get_job(db, job_id)
        return job.model

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

            out_files = []

            for b in batches:
                fout = b.output_file_id
                localpath = Path(b.local_file_id)

                fnew = localpath.parent / f"{localpath.stem}_result.jsonl"
                fnew.parent.mkdir(parents=True, exist_ok=True)

                # Stream OAI file to local
                with self.client.files.with_streaming_response.content(fout) as content:
                    with open(fnew, "wb") as f:
                        for chunk in content.iter_bytes():
                            f.write(chunk)

                out_files.append(fnew)
                progress.advance(task)  # Update progress bar

        return out_files
