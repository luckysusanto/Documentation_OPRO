import os
import sys
import time 

HF_HOME = os.environ.get("HF_HOME") # Set by `export HF_HOME=/path/to/your/cache`
if not HF_HOME:
    sys.exit("ERROR: HF_HOME is not set. Use `export HF_HOME=/path/to/your/cache` before running.")
if HF_HOME.startswith(os.path.expanduser("~")) and "scratch" not in HF_HOME.lower():
    print(f"WARNING: HF_HOME={HF_HOME} looks like it's under your home dir. Ensure you have enough space in your home dir before continuing.")
    print("Continuing in 3 seconds.")
    time.sleep(3)

from huggingface_hub import snapshot_download
from huggingface_hub.utils import HfHubHTTPError

MODELS = [
    "Qwen/Qwen3-8B",
    "Qwen/Qwen3-32B",
    "aisingapore/Qwen-SEA-LION-v4-32B-IT",
    "google/gemma-4-12b-it", 
    "google/gemma-4-31b-it", 
]

IGNORE = ["*.pt", "*.bin", "*.h5", "*.msgpack", "*.onnx", "*.gguf", "original/*"]

def main():
    print(f"HF_HOME = {HF_HOME}")
    print(f"caching {len(MODELS)} models\n")
    failed = []
    for i, repo in enumerate(MODELS, 1):
        print(f"[{i}/{len(MODELS)}] {repo}")
        try:
            path = snapshot_download(
                repo_id=repo,
                ignore_patterns=IGNORE,
                resume_download=True, 
                max_workers=4,
            )
            print(f"    OK -> {path}\n")
        except HfHubHTTPError as e:
            code = getattr(e.response, "status_code", "?")
            if code == 401 or code == 403:
                print(f"    AUTH ERROR ({code}): {repo} is likely gated. "
                      f"Accept its license on the HF page and run `hf auth login`.\n")
            else:
                print(f"    HTTP ERROR ({code}) on {repo}: {e}\n")
            failed.append(repo)
        except Exception as e:
            print(f"    FAILED {repo}: {type(e).__name__}: {e}\n")
            failed.append(repo)

    print("=" * 60)
    if failed:
        print(f"DONE with {len(failed)} failure(s):")
        for r in failed:
            print(f"  - {r}")
        sys.exit(1)
    print("DONE — all models cached successfully.")

if __name__ == "__main__":
    main()