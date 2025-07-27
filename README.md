# LoRA Dataset Tools 🛠️

All tools are based on a workflow using https://huggingface.co/deepghs datasets.

Scripts that help you **collect, clean and prepare image datasets** for training
LoRA / DreamBooth / SDXL finetunes.

| Pipeline stage | Script | What it does |
| -------------- | ------ | ------------ |
| 1️⃣ Filter & download | `Pull images 2.py` | Query the **Danbooru 2024** metadata parquet, apply powerful tag/size/score filters, and (optionally) download the resulting images via [`cheesechaser`](https://github.com/deepghs/cheesechaser). It also writes per‑image JSON side‑cars or one master metadata file. :contentReference[oaicite:0]{index=0} |
| 2️⃣ Sort by aspect | `sizesorter.py` | Moves *non‑square* images (or those below a size threshold) into a separate folder for follow‑up processing. Uses multithreading for speed. :contentReference[oaicite:1]{index=1} |
| 3️⃣ Auto‑crop | `crop.py` | Generates one or many centered **1:1 crops** from large images, with optional controlled overlap and minimum‑size enforcement. :contentReference[oaicite:2]{index=2} |
| 4️⃣ Clean orphan files | `cleanup.py` | Removes images that lost their JSON (or JSONs that lost their image) after manual editing. Multithreaded for large folders. :contentReference[oaicite:3]{index=3} |
| 5️⃣ Purge unwanted tags | `tag cleaner.py` | Recursively deletes any image + JSON pair that contains an *exact* keyword (case‑insensitive). :contentReference[oaicite:4]{index=4} |
| (Optional) Schema peek | `import pyarrow.py` | Quick one‑liner to print the column names of a Danbooru parquet with **PyArrow**. |

---

## Quick start

```bash
# 0. Clone repo & enter it
git clone https://github.com/grio43/LORA-tools.git
cd LORA-tools

# 1. Create & activate a Python 3.10+ venv (recommended)
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
