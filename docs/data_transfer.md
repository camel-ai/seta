# Data Transfer

How to move evaluation artifacts, datasets, and other large files between
the eval machine and external storage.

## Rclone + Google Drive

Use [rclone](https://rclone.org) to push eval result tarballs (or any other
artifacts) to Google Drive without browser uploads.

### Install

```bash
sudo apt install rclone
# or
curl https://rclone.org/install.sh | sudo bash
```

### One-time setup

```bash
rclone config
```

Walkthrough:

- `n` → new remote
- Name: `gdrive`
- Storage type: **24** (Google Drive) — **NOT 23** (that's Google Cloud Storage)
- `client_id` → leave blank (Enter)
- `client_secret` → leave blank (Enter)
- `scope` → `1` (full access)
- Everything else → keep pressing Enter (leave blank/default)
- `Edit advanced config?` → `n`
- `Use web browser to authenticate?` → `y`
- Authorize in browser
- `Configure as Shared Drive?` → `n`
- Confirm with `y`, then `q` to quit

### Usage

Upload a single file to a Drive folder:

```bash
rclone copy myfile.tar.gz gdrive:"folder-name"/ --progress
```

Upload an entire folder (parallel transfers):

```bash
rclone copy /path/to/local-folder gdrive:"remote-folder"/local-folder \
    --progress --transfers 8
```

List files in a Drive folder:

```bash
rclone ls gdrive:"folder-name"/
```

Download from Drive to local:

```bash
rclone copy gdrive:"folder-name"/file.tar.gz /path/to/local/ --progress
```

Mirror a local folder to Drive (deletes extras on the remote):

```bash
rclone sync /path/to/local-folder gdrive:"remote-folder"/ --progress
```

### Tips

- `--progress` → show real-time transfer stats.
- `--transfers 8` → upload 8 files in parallel; faster for folders with many small files.
- `--dry-run` → preview what would happen without actually transferring.
- Prefer `copy` (safe, no deletes) over `sync` (mirrors, deletes extras) unless you specifically want an exact mirror.
