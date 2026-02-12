import typer
from rich import print
from decouple import config
from .etl import KaggleDatasetExtractor, ArxivDatasetTransformer
from .embeddings import VectorEmbeddings
from .db.crud import DBRepository
from .db.database import engine

# Initialize Typer app
app = typer.Typer()

# Create DB Tables if they don't exist
DBRepository.create_db(engine)


@app.command()
def recreate_db():
    """🔮 Recreates the entire Database."""

    DBRepository.recreate_db(engine)
    print("🔮 Database recreated!")


@app.command()
def get_dataset():
    """📄 Download and clean the latest version of the ArXiv Kaggle Dataset."""

    # Extract
    extractor = KaggleDatasetExtractor(config("KAGGLE_USERNAME"), config("KAGGLE_KEY"))
    path = extractor.get_dataset(dataset="Cornell-University/arxiv", path="./data")
    print(f"Files downloaded at: /{path}")

    # Transform
    transformer = ArxivDatasetTransformer(
        jsonfile="./data/arxiv-metadata-oai-snapshot.json"
    )
    clean_path = transformer.clean(outputfile="./data/clean-arxiv-metadata-oai.parquet")
    print(f"Files created at: /{clean_path}")


@app.command()
def compute_embeddings(model: str = "Qwen/Qwen3-Embedding-8B"):
    """⚒️  Create and Embedding Job and Submit a request to the Batch Embedding API form the cleaned dataset file."""

    Embeddings = VectorEmbeddings(model)
    job_id = Embeddings.encode_batch(
        file="./data/interim/clean-arxiv-metadata-oai.parquet",
        text_column=["title", "abstract"],
    )

    Embeddings.check_status(job_id)
    print(
        f"\n[bold]✨ Batch embeddings posted.[/bold] You can check the requests info at [bold yellow]check-embeddings-status --id {job_id}[/]"
    )


@app.command()
def check_embeddings_status(job_id: int):
    """🔎 Check the status for an Embedding Job. Info is updated every 1 minute."""

    Embeddings = VectorEmbeddings()
    Embeddings.check_status(job_id, refresh=True)


@app.command()
def get_embeddings(job_id: int):
    """✨ Download the embeddings for a given Job. Only if the job has status completed."""
    Embeddings = VectorEmbeddings()
    embs_file = Embeddings.get_embeddings(job_id)
    print(f"\n[bold]✨ Embeddings for job {job_id} downloaded at {embs_file}!")


if __name__ == "__main__":
    app()
