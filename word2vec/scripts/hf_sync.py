"""Make a Hugging Face repo match a staged folder in one commit, except for kept path prefixes.

Every file under STAGE is added (files the Hub already holds, such as ONNX tables uploaded earlier, are not
sent again), and every repo file that is neither in STAGE nor under a --keep prefix is deleted.
Without --apply it only prints what it would do. Run with the Python that has huggingface_hub, logged in:
    python scripts/hf_sync.py mkkhavil/word2vec-wiki-cs-en-uk $W2V_STORAGE/hf_stage --keep e5-jepa-lemb0/
The deleted files stay in the repo's git history; HfApi.super_squash_history removes them for good.
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

ALWAYS_KEEP = (".gitattributes",)


def plan(repo_files: list[str], stage: Path, keep: tuple[str, ...]) -> tuple[list[str], list[str]]:
    """Returns (paths to add from stage, repo paths to delete)."""
    add = sorted(p.relative_to(stage).as_posix() for p in stage.rglob("*")
                 if p.is_file() and ".cache" not in p.relative_to(stage).parts)
    staged = set(add)
    delete = sorted(f for f in repo_files
                    if f not in staged and f not in ALWAYS_KEEP and not f.startswith(keep))
    return add, delete


def summary(paths: list[str]) -> str:
    """Counts per top-level folder (run directories grouped) and per file name."""
    top = Counter("<run>/" if p.count("/") == 1 and "-s" in p.split("/")[0] else p.split("/")[0] + "/"
                  if "/" in p else p for p in paths)
    names = Counter(p.rsplit("/", 1)[-1] for p in paths)
    return f"{len(paths)} files; by folder {dict(top)}; by name {dict(names)}"


def main(argv=None):
    from huggingface_hub import CommitOperationAdd, CommitOperationDelete, HfApi

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("repo")
    p.add_argument("stage", type=Path)
    p.add_argument("--keep", action="append", default=[], help="repo path prefix to leave untouched (repeatable)")
    p.add_argument("--message", default="Sync repo with staged folder")
    p.add_argument("--apply", action="store_true", help="create the commit (default: dry run)")
    a = p.parse_args(argv)
    if not a.stage.is_dir():
        raise SystemExit(f"{a.stage} is not a directory")
    api = HfApi()
    add, delete = plan(api.list_repo_files(a.repo), a.stage, tuple(a.keep))
    if not add:
        raise SystemExit(f"nothing staged under {a.stage}; refusing to delete the repo contents")
    print("add/update:", summary(add))
    print("delete:    ", summary(delete))
    if not a.apply:
        print("dry run; pass --apply to commit")
        return
    ops = [CommitOperationDelete(path_in_repo=f) for f in delete]
    ops += [CommitOperationAdd(path_in_repo=f, path_or_fileobj=str(a.stage / f)) for f in add]
    info = api.create_commit(a.repo, ops, commit_message=a.message, repo_type="model")
    print("committed:", info.commit_url)


if __name__ == "__main__":
    main()
