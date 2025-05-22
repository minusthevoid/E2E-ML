from typing import Iterable
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
