from __future__ import annotations

import argparse
from pathlib import Path

import uproot


def find_root_files(
    input_dir: Path,
    output_path: Path,
    recursive: bool,
) -> list[Path]:
    pattern = "**/*.root" if recursive else "*.root"
    output_resolved = output_path.resolve()

    return sorted(
        path.resolve()
        for path in input_dir.glob(pattern)
        if path.is_file() and path.resolve() != output_resolved
    )


def find_tree_name(root_file: Path) -> str:
    """Return the first TTree found in the file."""

    with uproot.open(root_file) as source:
        tree_names = [
            key.split(";")[0]
            for key, class_name in source.classnames().items()
            if class_name == "TTree"
        ]

    if not tree_names:
        raise RuntimeError(f"No TTree found in {root_file}")

    if len(tree_names) > 1:
        raise RuntimeError(
            f"Multiple TTrees found in {root_file}: {tree_names}. "
            "Specify the tree explicitly with --tree."
        )

    return tree_names[0]


def verify_tree_compatibility(
    files: list[Path],
    tree_name: str,
) -> dict[str, str]:
    """Check that all input trees have the same branch schema."""

    reference_schema: dict[str, str] | None = None
    reference_file: Path | None = None

    for path in files:
        with uproot.open(path) as source:
            if tree_name not in source:
                raise RuntimeError(
                    f"Tree {tree_name!r} was not found in {path}"
                )

            tree = source[tree_name]
            schema = {
                name: branch.typename
                for name, branch in tree.items()
            }

        if reference_schema is None:
            reference_schema = schema
            reference_file = path
        elif schema != reference_schema:
            missing = sorted(set(reference_schema) - set(schema))
            extra = sorted(set(schema) - set(reference_schema))
            changed = sorted(
                name
                for name in set(reference_schema).intersection(schema)
                if reference_schema[name] != schema[name]
            )

            raise RuntimeError(
                f"Incompatible tree schema in {path}\n"
                f"Reference file: {reference_file}\n"
                f"Missing branches: {missing}\n"
                f"Extra branches: {extra}\n"
                f"Branches with changed types: {changed}"
            )

    assert reference_schema is not None
    return reference_schema


def merge_root_files(
    files: list[Path],
    output_path: Path,
    tree_name: str,
    step_size: str,
    overwrite: bool,
) -> None:
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists: {output_path}\n"
            "Use --overwrite to replace it."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        output_path.unlink()

    schema = verify_tree_compatibility(files, tree_name)

    # Uproot accepts NumPy/Awkward-compatible branch type descriptions.
    with uproot.recreate(output_path) as destination:
        output_tree = None
        total_entries = 0

        sources = {
            f"{path}:{tree_name}": tree_name
            for path in files
        }

        for batch in uproot.iterate(
            list(sources),
            step_size=step_size,
            library="ak",
        ):
            if output_tree is None:
                # Assigning the first batch creates the output TTree.
                destination[tree_name] = batch
                output_tree = destination[tree_name]
            else:
                output_tree.extend(batch)

            total_entries += len(batch)
            print(f"\rMerged entries: {total_entries:,}", end="")

    print()
    print(f"Created: {output_path}")
    print(f"Files merged: {len(files)}")
    print(f"Entries written: {total_entries:,}")
    print(f"Branches: {len(schema)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge compatible TTrees from ROOT files using Uproot."
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Directory containing the input ROOT files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path. Default: <input_dir>/all.root",
    )
    parser.add_argument(
        "--tree",
        type=str,
        default=None,
        help="TTree name. If omitted, it is inferred from the first file.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search recursively in subdirectories.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output file.",
    )
    parser.add_argument(
        "--step-size",
        default="100 MB",
        help='Chunk size used while merging. Default: "100 MB".',
    )

    args = parser.parse_args()

    input_dir = args.input_dir.resolve()
    if not input_dir.is_dir():
        parser.error(f"Directory does not exist: {input_dir}")

    output_path = (
        args.output.resolve()
        if args.output is not None
        else input_dir / "all.root"
    )

    files = find_root_files(
        input_dir=input_dir,
        output_path=output_path,
        recursive=args.recursive,
    )

    if not files:
        parser.error(f"No ROOT files found in {input_dir}")

    tree_name = args.tree or find_tree_name(files[0])

    print(f"Tree: {tree_name}")
    print(f"Input files: {len(files)}")

    merge_root_files(
        files=files,
        output_path=output_path,
        tree_name=tree_name,
        step_size=args.step_size,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()