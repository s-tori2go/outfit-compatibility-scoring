import os
import requests

# 0.1 Check if required directories exist
def download_file(url, path):
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        print(f"\n⬇️ Downloading {os.path.basename(path)}...")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        try:
            with requests.get(url, stream=True) as r:
                r.raise_for_status()
                total_size = int(r.headers.get('content-length', 0))
                downloaded = 0
                with open(path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            print(f"\rProgress: {downloaded / total_size * 100:.1f}%", end='')
                print(f"\n✅ Download of {os.path.basename(path)} complete: {path}")
        except Exception as e:
            print(f"\n❌ Exception during download: {e}")
    else:
        print(f"✅ {os.path.basename(path)} exists")
