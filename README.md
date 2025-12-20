

## 1. Create a kthcloud VM

1. Go to `https://cloud.cbh.kth.se` and log in with KTH SSO.[^1][^2]
2. In **Profile**, add your SSH public key so new VMs can use it.[^2]
3. Go to **Virtual Machines** → **Create VM**:[^3]
    - OS template: Ubuntu Server 24.04 (default).
    - CPU: 2 cores.
    - RAM: 4 GB.
    - Disk: 20 GB (default).
    - Select your SSH key.
4. Wait for the VM to be ready, note the hostname/port (e.g. `vm.cloud.cbh.kth.se` and port).

## 2. SSH into the VM

From your local machine:

```bash
ssh -p <PORT> root@vm.cloud.cbh.kth.se
```

No extra port forwarding is needed; kthcloud already exposes SSH.[^4][^3]

On the VM you should see:

```bash
lsb_release -a
# Ubuntu 24.04.3 LTS (noble)
```


## 3. Base system setup

On the VM:

```bash
apt update && apt upgrade -y

apt install -y \
  python3 python3-venv python3-pip \
  git curl
```

This gives you a recent Python and basic tools on Ubuntu 24.04.[^5]

## 4. Create Python virtual env and install DuckDB stack

```bash
python3 -m venv ~/venv
. ~/venv/bin/activate

pip install --upgrade pip
pip install duckdb requests numpy pandas
```

- `duckdb` is the DB engine; `requests` talks to the LLM HTTP API; `numpy`/`pandas` are used by `fetchdf()`.[^6][^7]


## 5. Copy the DuckDB database to the VM

From your local machine:

```bash
scp -P <PORT> /path/to/kth_metadata.duckdb root@vm.cloud.cbh.kth.se:/root/kth_metadata.duckdb
```

Test from inside the VM:

```bash
. ~/venv/bin/activate

python -c "import duckdb; con=duckdb.connect('kth_metadata.duckdb'); print(con.execute('SHOW TABLES').fetchall())"
```

You should see tables like `pub`, `titleInfo`, `abstract`, etc.[^7]

## 6. Install Ollama (open LLM runtime)

On the VM (still as root is fine here):

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

This will:

- Install Ollama binaries to `/usr/local`.
- Create an `ollama` user and systemd service.
- Start the service and expose the API at `127.0.0.1:11434`.[^8][^9]

Verify:

```bash
ollama --help
systemctl status ollama
```


## 7. Pull a small open model that fits 4 GB RAM

On 4 GB RAM, use a small Llama 3.2 variant:

```bash
ollama pull llama3.2:1b
```

Test interactively:

```bash
ollama run llama3.2:1b
# type a short question, Ctrl+D to exit
```

The 1B variant is designed for constrained environments and runs on low RAM systems.[^10][^11]

If you previously pulled a larger model (e.g. `mistral`) and want to free space:

```bash
ollama list
ollama rm mistral
```


## 8. Test DuckDB → Llama 3.2:1b integration

Create `/root/kth_llama_pub.py`:

```python
import json
import duckdb
import requests

DB_PATH = "kth_metadata.duckdb"
MODEL = "llama3.2:1b"

con = duckdb.connect(DB_PATH)

# Fetch a few publications from the 'pub' table
df = con.execute("""
    SELECT
      PID,
      Title,
      Abstract,
      PublicationType,
      ContentType,
      Language,
      Year,
      Publisher,
      Journal,
      DOI,
      FreeFulltext
    FROM pub
    LIMIT 5
""").fetchdf()

for idx, row in df.iterrows():
    record = row.to_dict()

    prompt = f"""You are an expert on KTH DiVA publication metadata.

Here is one publication record from the 'pub' table as JSON:
{json.dumps(record, indent=2)}

1. Explain in plain English what this publication is about.
2. Briefly describe the meaning of each field shown.
3. Infer whether this looks like an article, conference paper, thesis, etc., and state your reasoning.
"""

    resp = requests.post(
        "http://127.0.0.1:11434/api/generate",
        json={"model": MODEL, "prompt": prompt, "stream": False},
    )
    resp.raise_for_status()
    data = resp.json()

    print(f"\n=== PID {record['PID']} (row {idx}) ===\n")
    print(data["response"])
```

Run:

```bash
. ~/venv/bin/activate
python kth_llama_pub.py
```

You should see one explanation per row; `stream: false` ensures Ollama returns a single JSON object per request instead of NDJSON chunks.[^9][^12]

## 9. Optional: using MinIO later

If later you want to fetch the DuckDB file from MinIO instead of copying it manually:

1. Install MinIO server on some VM or on this one (standalone mode):[^13][^14]

```bash
wget https://dl.min.io/server/minio/release/linux-amd64/minio
chmod +x minio
./minio server /mnt/data
```

2. Use `mc` (MinIO client) or S3 libraries from Python to sync the `.duckdb` file to `/root/kth_metadata.duckdb` before running your scripts.

The rest of the DuckDB + Llama pipeline remains identical.

***

Following these steps in order on a fresh kthcloud VM will reproduce your current environment: Ubuntu 24.04 VM, Python venv with DuckDB + helpers, Ollama with `llama3.2:1b`, and a working script that feeds KTH DiVA metadata from DuckDB into an open LLM.

<div align="center">⁂</div>

[^1]: https://cloud.cbh.kth.se

[^2]: https://docs.cloud.cbh.kth.se/usage/profile/

[^3]: https://docs.cloud.cbh.kth.se/usage/virtualMachines/

[^4]: https://intra.kth.se/en/it/arbeta-pa-distans/unix/ssh-1.971102

[^5]: https://idroot.us/install-duckdb-ubuntu-24-04/

[^6]: https://duckdb.org/docs/installation/

[^7]: https://motherduck.com/blog/duckdb-tutorial-for-beginners/

[^8]: https://ollama.com/install.sh

[^9]: https://docs.ollama.com/linux

[^10]: https://ollama.com/blog/llama3.2

[^11]: https://registry.ollama.ai/library/llama3.2

[^12]: https://github.com/ollama/ollama

[^13]: https://www.atlantic.net/dedicated-server-hosting/how-to-deploy-minio-on-ubuntu-24-04-an-open-source-object-storage-application/

[^14]: https://www.digitalocean.com/community/tutorials/how-to-set-up-minio-object-storage-server-in-standalone-mode-on-ubuntu-20-04

