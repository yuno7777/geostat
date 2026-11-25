"""Test local sample ingest in a throwaway database."""

from contextlib import closing
from pathlib import Path
from tempfile import TemporaryDirectory

from geostat import db, ingest


def main() -> None:
    with TemporaryDirectory(prefix="geostat-ingest-") as directory:
        with closing(db.connect(Path(directory) / "smoke.sqlite3")) as conn:
            db.init_db(conn)
            with db.begin_run(conn, "ingest", config={"mode": "sample"}) as run:
                result = ingest.ingest_dataset(conn, run, sample=True)
            print(f"Ingested scenes: {result['scenes']}")


if __name__ == "__main__":
    main()
