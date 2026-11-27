"""Generate deterministic synthetic PIT fixtures into data/fixtures/."""

from __future__ import annotations

from pathlib import Path

from alpha_engine.data.fixtures import generate_synthetic_pit_dataset, save_dataset_fixtures


def main() -> None:
    fixtures_dir = Path(__file__).resolve().parent
    bundle = generate_synthetic_pit_dataset(seed=42)
    manifest = save_dataset_fixtures(bundle, fixtures_dir)
    print(
        f"Generated synthetic PIT fixtures in {fixtures_dir} "
        f"(dataset_id={manifest.dataset_id}, digest={manifest.manifest_digest[:12]}, "
        f"bars={bundle.bars.height}, fundamentals={bundle.fundamentals.height}, "
        f"actions={bundle.corporate_actions.height})"
    )


if __name__ == "__main__":
    main()
