from pathlib import Path

IMAGE_DIR = Path("data/raw/images")

SUPPORTED_EXTS = {".jpg"}

files = sorted(
    [f for f in IMAGE_DIR.iterdir() if f.suffix.lower() in SUPPORTED_EXTS]
)

# First rename to temporary names to avoid filename collisions
temp_files = []
for i, file in enumerate(files):
    temp_name = IMAGE_DIR / f"temp_{i}{file.suffix.lower()}"
    file.rename(temp_name)
    temp_files.append(temp_name)

# Rename to sequential filenames
for i, file in enumerate(sorted(temp_files), start=1):
    new_name = IMAGE_DIR / f"{i:06d}.jpg"
    file.rename(new_name)

print(f"Renamed {len(temp_files)} images.")