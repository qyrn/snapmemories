from datetime import datetime, timedelta, timezone
from pathlib import Path

from snapmemories.library import Library, LibraryRecord
from snapmemories.models import MediaKind

TAKEN = datetime(2026, 5, 8, 19, 1, 16, tzinfo=timezone(timedelta(hours=2)))


def test_reserved_paths_never_collide(tmp_path: Path) -> None:
    library = Library(tmp_path)

    first = library.reserve_path(TAKEN, "jpg")
    second = library.reserve_path(TAKEN, "jpg")

    assert first.name == "2026-05-08_19-01-16.jpg"
    assert second.name == "2026-05-08_19-01-16_2.jpg"
    assert first.parent == tmp_path / "2026" / "2026-05"


def test_records_pointing_outside_the_library_are_ignored(tmp_path: Path) -> None:
    library = Library(tmp_path / "Memories")
    (tmp_path / "secret.txt").write_text("x")
    library.add(
        LibraryRecord("evil", "../secret.txt", MediaKind.PHOTO, TAKEN.isoformat(), None, None)
    )

    assert library.existing_records() == {}
