from typing import Iterable
from bing_image_downloader import downloader


def download_images(search_terms: str, num_images: int, output_dir: str) -> None:
    """Download images for the given comma-separated search terms."""
    terms = [term.strip() for term in search_terms.split(',') if term.strip()]
    for term in terms:
        downloader.download(term, limit=num_images, output_dir=output_dir,
                            adult_filter_off=True, force_replace=False, timeout=60)
