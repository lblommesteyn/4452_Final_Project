from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Iterable
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
from PIL import Image


CASTLES_MODELS_URL = "https://www.cfa.harvard.edu/castles/models.html"
CASTLES_INDIVIDUAL_BASE = "https://lweb.cfa.harvard.edu/castles/Individual/"
GZH_NEGATIVE_PAGES = [
    "https://data.galaxyzoo.org/data/gzh/samples/low_redshift_disks.html",
    "https://data.galaxyzoo.org/data/gzh/samples/intermediate_redshift_disks.html",
    "https://data.galaxyzoo.org/data/gzh/samples/high_redshift_disks.html",
]

REQUEST_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; strong-lens-project/1.0; +https://data.galaxyzoo.org/)",
}


@dataclass
class DownloadedRecord:
    image_path: str
    label: int
    split: str
    source: str
    group_id: str
    original_url: str


def _request_text(url: str, timeout: int = 60) -> str:
    response = requests.get(url, headers=REQUEST_HEADERS, timeout=timeout)
    response.raise_for_status()
    return response.text


def _request_bytes(url: str, timeout: int = 60) -> bytes:
    response = requests.get(url, headers=REQUEST_HEADERS, timeout=timeout)
    response.raise_for_status()
    return response.content


def scrape_castles_image_urls() -> list[dict[str, str]]:
    html = _request_text(CASTLES_MODELS_URL)
    soup = BeautifulSoup(html, "html.parser")

    systems: dict[str, str] = {}
    for anchor in soup.find_all("a"):
        href = anchor.get("href")
        text = " ".join(anchor.get_text(" ", strip=True).split())
        if href and href.endswith(".html") and text:
            systems[href] = text

    image_entries: list[dict[str, str]] = []
    for href, system_name in sorted(systems.items()):
        page_url = urljoin(CASTLES_INDIVIDUAL_BASE, href)
        try:
            page_html = _request_text(page_url)
        except requests.HTTPError:
            continue
        page_soup = BeautifulSoup(page_html, "html.parser")
        image_links = []
        for anchor in page_soup.find_all("a"):
            image_href = anchor.get("href")
            if not image_href:
                continue
            lowered = image_href.lower()
            if "postagestamps/gifs/fullsize" not in lowered or not lowered.endswith(".gif"):
                continue
            if "/animate/" in lowered:
                continue
            image_links.append(urljoin(page_url, image_href))

        # Keep all available full-size static images; splitting is done by lens system to avoid leakage.
        for index, image_url in enumerate(sorted(set(image_links))):
            image_entries.append(
                {
                    "group_id": Path(href).stem,
                    "system_name": system_name,
                    "image_url": image_url,
                    "variant_id": f"{Path(href).stem}_{index:02d}",
                }
            )
    return image_entries


def scrape_gzh_negative_urls(max_images: int, seed: int) -> list[str]:
    image_urls: list[str] = []
    for page_url in GZH_NEGATIVE_PAGES:
        html = _request_text(page_url)
        soup = BeautifulSoup(html, "html.parser")
        for image in soup.find_all("img"):
            src = image.get("src")
            if not src:
                continue
            normalized = src.replace("http://", "https://")
            if "zoo-hst.s3.amazonaws.com" in normalized and normalized.lower().endswith(".jpg"):
                image_urls.append(normalized)

    deduped = sorted(set(image_urls))
    rng = random.Random(seed)
    rng.shuffle(deduped)
    return deduped[:max_images]


def _split_groups(group_ids: Iterable[str], seed: int) -> dict[str, str]:
    group_ids = sorted(set(group_ids))
    rng = random.Random(seed)
    rng.shuffle(group_ids)

    total = len(group_ids)
    train_end = max(1, int(round(total * 0.7)))
    val_end = max(train_end + 1, int(round(total * 0.85)))
    if val_end >= total:
        val_end = total - 1

    mapping: dict[str, str] = {}
    for group_id in group_ids[:train_end]:
        mapping[group_id] = "train"
    for group_id in group_ids[train_end:val_end]:
        mapping[group_id] = "val"
    for group_id in group_ids[val_end:]:
        mapping[group_id] = "test"
    return mapping


def _save_image_as_grayscale_png(image_bytes: bytes, output_path: Path) -> None:
    with Image.open(BytesIO(image_bytes)) as image:
        image = image.convert("L").convert("RGB")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(output_path, format="PNG")


def build_real_dataset(
    output_dir: str | Path,
    seed: int = 42,
    test_negative_ratio: int = 4,
) -> dict[str, object]:
    output_dir = Path(output_dir)
    images_dir = output_dir / "images"
    manifest_path = output_dir / "manifest.csv"
    images_dir.mkdir(parents=True, exist_ok=True)

    positive_entries = scrape_castles_image_urls()
    positive_split_map = _split_groups([entry["group_id"] for entry in positive_entries], seed=seed)

    positive_counts = {"train": 0, "val": 0, "test": 0}
    for entry in positive_entries:
        positive_counts[positive_split_map[entry["group_id"]]] += 1

    negative_target_counts = {
        "train": positive_counts["train"],
        "val": positive_counts["val"],
        "test": positive_counts["test"] * test_negative_ratio,
    }
    negative_urls = scrape_gzh_negative_urls(
        max_images=sum(negative_target_counts.values()),
        seed=seed,
    )
    if len(negative_urls) < sum(negative_target_counts.values()):
        raise RuntimeError("Not enough Galaxy Zoo: Hubble negatives were collected for the requested split sizes.")

    downloaded_rows: list[DownloadedRecord] = []

    for index, entry in enumerate(positive_entries):
        split = positive_split_map[entry["group_id"]]
        image_bytes = _request_bytes(entry["image_url"])
        output_path = images_dir / split / "lens" / f"{entry['variant_id']}.png"
        _save_image_as_grayscale_png(image_bytes, output_path)
        downloaded_rows.append(
            DownloadedRecord(
                image_path=str(output_path.relative_to(output_dir)),
                label=1,
                split=split,
                source="castles_hst",
                group_id=entry["group_id"],
                original_url=entry["image_url"],
            )
        )

    cursor = 0
    for split in ("train", "val", "test"):
        split_count = negative_target_counts[split]
        for split_index in range(split_count):
            image_url = negative_urls[cursor]
            cursor += 1
            image_bytes = _request_bytes(image_url)
            output_path = images_dir / split / "non_lens" / f"gzh_{split}_{split_index:04d}.png"
            _save_image_as_grayscale_png(image_bytes, output_path)
            downloaded_rows.append(
                DownloadedRecord(
                    image_path=str(output_path.relative_to(output_dir)),
                    label=0,
                    split=split,
                    source="galaxy_zoo_hubble",
                    group_id=f"gzh_{split}_{split_index:04d}",
                    original_url=image_url,
                )
            )

    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["image_path", "label", "split", "source", "group_id", "original_url"],
        )
        writer.writeheader()
        for row in downloaded_rows:
            writer.writerow(
                {
                    "image_path": row.image_path,
                    "label": row.label,
                    "split": row.split,
                    "source": row.source,
                    "group_id": row.group_id,
                    "original_url": row.original_url,
                }
            )

    return {
        "manifest_path": str(manifest_path),
        "positive_counts": positive_counts,
        "negative_counts": negative_target_counts,
        "num_rows": len(downloaded_rows),
    }
