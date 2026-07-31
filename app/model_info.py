from dataclasses import dataclass, field
from pathlib import Path
from gguf import GGUFReader
from config import TENSOR_GROUPS

@dataclass
class GroupInfo:
    name: str
    size_mb: float
    count: int

@dataclass
class ModelInfo:
    path: Path
    n_layers: int
    arch: str
    file_size_mb: float
    groups: dict = field(default_factory=dict)

def inspect(path: Path) -> ModelInfo:
    reader = GGUFReader(str(path))

    arch, n_layers = "unknown", 0
    for field_name, f in reader.fields.items():
        if field_name == "general.architecture":
            arch = str(f.contents())
        if field_name.endswith(".block_count"):
            n_layers = int(f.contents())

    groups = {g: GroupInfo(g, 0.0, 0) for g in TENSOR_GROUPS}
    for t in reader.tensors:
        for g in TENSOR_GROUPS:
            if g in t.name:
                groups[g].size_mb += t.n_bytes / 1024**2
                groups[g].count += 1
                break

    return ModelInfo(
        path=path,
        n_layers=n_layers,
        arch=arch,
        file_size_mb=path.stat().st_size / 1024**2,
        groups={k: v for k, v in groups.items() if v.count > 0},
    )

if __name__ == "__main__":
    from config import MODEL_DIR
    info = inspect(MODEL_DIR / "qwen1.5b-q4.gguf")
    print(f"{info.arch}  {info.n_layers} layers  {info.file_size_mb:.0f} MB\n")
    for g in sorted(info.groups.values(), key=lambda x: -x.size_mb):
        print(f"  {g.name:14s} {g.size_mb:8.1f} MB  ({g.count} tensors)")