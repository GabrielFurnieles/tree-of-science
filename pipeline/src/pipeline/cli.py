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


# Commands
@app.command()
def hello_pipeline():
    print("Hello pipeline!")


@app.command()
def bye_pipeline():
    print("Bye pipeline!")


@app.command()
def recreate_db():
    DBRepository.recreate_db(engine)
    print("🔮 Database recreated!")


# TEST
@app.command()
def extract_dataset():
    extractor = KaggleDatasetExtractor(config("KAGGLE_USERNAME"), config("KAGGLE_KEY"))
    path = extractor.get_dataset(dataset="Cornell-University/arxiv", path="./data")
    print(f"Files downloaded at: /{path}")


# TEST
@app.command()
def clean_dataset():
    transformer = ArxivDatasetTransformer(
        jsonfile="./data/arxiv-metadata-oai-snapshot.json"
    )
    clean_path = transformer.clean(outputfile="./data/clean-arxiv-metadata-oai.parquet")
    print(f"Files created at: /{clean_path}")


# TEST
@app.command()
def post_embeddings_batch():
    embeddings = VectorEmbeddings(model="Qwen/Qwen3-Embedding-8B")
    job_id = embeddings.encode_batch(
        file="./data/interim/clean-arxiv-metadata-oai.parquet",
        text_column=["title", "abstract"],
    )
    print(
        f"\n[bold]✨ Batch embeddings posted.[/bold] You can check the requests info at get-job-status --id {job_id}"
    )


if __name__ == "__main__":
    app()
