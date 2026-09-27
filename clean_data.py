"""Read-only audit of the bundled school profile dataset."""

from data import configured_data_path, load_dataset, missingness_summary


def main() -> None:
    path = configured_data_path()
    frame = load_dataset(path)
    print(f"Source: {path}")
    print(f"Name-keyed school records: {len(frame):,}")
    print("Missing domain scores:")
    summary = missingness_summary(frame)
    for _, row in summary.iterrows():
        print(f"  {row['Domain']}: {row['Missing schools']:,}")
    print("NCESSCH is not read; source files are never modified.")


if __name__ == "__main__":
    main()
