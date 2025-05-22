from typing import Iterable
from pathlib import Path
from bing_image_downloader import downloader


# Work around a bug in bing_image_downloader where it calls ``Path.isdir``
# instead of ``Path.is_dir``. Add the alias if missing so downloads work.
if not hasattr(Path, "isdir"):
    Path.isdir = Path.is_dir


import os
import shutil
from bing_image_downloader import downloader


def download_images(search_terms: str, num_images: int, output_dir: str) -> None:
    """Download images for the given comma-separated search terms."""
    terms = [term.strip() for term in search_terms.split(',') if term.strip()]
    for term in terms:
        term_dir = os.path.join(output_dir, term)
        if os.path.exists(term_dir):
            shutil.rmtree(term_dir)
        downloader.download(
            term,
            limit=num_images,
            output_dir=output_dir,
            adult_filter_off=True,
            force_replace=True,
            timeout=60,
        )
        downloader.download(term, limit=num_images, output_dir=output_dir,
                            adult_filter_off=True, force_replace=False, timeout=60)
