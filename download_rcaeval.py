from huggingface_hub import snapshot_download

# Download only RE1 first
snapshot_download(
    repo_id="phamquiluan/RCAEval",
    repo_type="dataset",
    allow_patterns="re1*",
    local_dir="data/rcaeval-data",
)

print("RE1 download complete")