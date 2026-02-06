"""
Image loading utilities for A-LLMRec + SmolVLM.

Provides robust image loading from URLs with retry logic, disk caching,
and fallback handling for recommendation system inference.

Usage:
    from utils_image import ImageLoader
    
    loader = ImageLoader(timeout=10, max_retries=2, cache_dir="./image_cache")
    image, error = loader.load_image("https://example.com/image.jpg")
    if image:
        # Use PIL Image
        pass
"""

import io
import time
import hashlib
import logging
from typing import Optional, Tuple, List
from pathlib import Path

import requests
from PIL import Image

logger = logging.getLogger(__name__)


class ImageLoader:
    """
    Robust image loader with retry logic and disk caching for Amazon product images.
    
    Features:
    - Disk-based image cache (avoids re-downloading across epochs)
    - Retry with exponential backoff
    - User-Agent headers for Amazon CDN compatibility
    - Timeout handling
    - RGB conversion
    - Error tracking
    
    Args:
        timeout: Request timeout in seconds (default: 10)
        max_retries: Maximum retry attempts (default: 2)
        user_agent: Custom user agent string (optional)
        cache_dir: Path to disk cache directory (optional, enables caching)
    """
    
    DEFAULT_USER_AGENT = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
    
    def __init__(
        self,
        timeout: int = 10,
        max_retries: int = 2,
        user_agent: str = None,
        cache_dir: str = None,
    ):
        self.timeout = timeout
        self.max_retries = max_retries
        self.user_agent = user_agent or self.DEFAULT_USER_AGENT
        
        # Disk cache setup
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Image cache directory: {self.cache_dir}")
        
        # Statistics
        self.stats = {
            'total_requests': 0,
            'successful': 0,
            'failed': 0,
            'retries': 0,
            'cache_hits': 0,
        }
        
        logger.info(
            f"ImageLoader initialized: timeout={timeout}s, max_retries={max_retries}, "
            f"cache={'enabled' if cache_dir else 'disabled'}"
        )
    
    def _download_url(self, url: str) -> bytes:
        """
        Download image from URL with retry logic.
        
        Args:
            url: Image URL
            
        Returns:
            Raw image bytes
            
        Raises:
            requests.RequestException: If all retries fail
        """
        last_error = None
        
        for attempt in range(self.max_retries + 1):
            try:
                response = requests.get(
                    url,
                    timeout=self.timeout,
                    headers={'User-Agent': self.user_agent},
                    stream=True
                )
                response.raise_for_status()
                return response.content
                
            except requests.RequestException as e:
                last_error = e
                if attempt < self.max_retries:
                    wait_time = 2 ** attempt  # Exponential backoff: 1, 2, 4...
                    logger.debug(f"Retry {attempt + 1}/{self.max_retries} for {url} after {wait_time}s")
                    self.stats['retries'] += 1
                    time.sleep(wait_time)
                    continue
                raise last_error
    
    def _get_cache_path(self, url: str) -> Optional[Path]:
        """Get the cache file path for a URL using MD5 hash."""
        if self.cache_dir is None:
            return None
        cache_key = hashlib.md5(url.encode()).hexdigest()
        return self.cache_dir / f"{cache_key}.jpg"

    def _load_from_cache(self, url: str) -> Optional[Image.Image]:
        """Try to load an image from disk cache."""
        cache_path = self._get_cache_path(url)
        if cache_path is None or not cache_path.exists():
            return None
        try:
            img = Image.open(cache_path).convert("RGB")
            self.stats['cache_hits'] += 1
            logger.debug(f"Cache hit: {url[:60]}...")
            return img
        except Exception:
            # Corrupted cache file — delete it
            cache_path.unlink(missing_ok=True)
            return None

    def _save_to_cache(self, url: str, image_bytes: bytes):
        """Save downloaded image bytes to disk cache."""
        cache_path = self._get_cache_path(url)
        if cache_path is None:
            return
        try:
            cache_path.write_bytes(image_bytes)
        except Exception as e:
            logger.debug(f"Failed to cache image: {e}")

    def load_image(self, path_or_url: str) -> Tuple[Optional[Image.Image], Optional[str]]:
        """
        Load image from path or URL.
        
        Handles both local files and remote URLs. For URLs, checks disk cache
        first, then downloads with retry logic and exponential backoff.
        Images are converted to RGB format.
        
        Args:
            path_or_url: Local file path or HTTP(S) URL
            
        Returns:
            Tuple of (image, error_message):
            - If successful: (PIL.Image, None)
            - If failed: (None, error_string)
        """
        self.stats['total_requests'] += 1
        
        try:
            # Check if URL or local path
            is_url = path_or_url.startswith("http://") or path_or_url.startswith("https://")
            
            if is_url:
                # Check disk cache first
                cached_img = self._load_from_cache(path_or_url)
                if cached_img is not None:
                    self.stats['successful'] += 1
                    return cached_img, None

                # Download from URL
                logger.debug(f"Downloading: {path_or_url[:80]}...")
                data = self._download_url(path_or_url)
                
                # Open and verify image
                img = Image.open(io.BytesIO(data))
                img.verify()  # Verify image integrity
                
                # Re-open after verify (verify() closes the file)
                img = Image.open(io.BytesIO(data)).convert("RGB")

                # Save to disk cache
                self._save_to_cache(path_or_url, data)
                
            else:
                # Local file
                path = Path(path_or_url)
                if not path.exists():
                    self.stats['failed'] += 1
                    return None, f"File not found: {path_or_url}"
                
                img = Image.open(path_or_url).convert("RGB")
            
            self.stats['successful'] += 1
            logger.debug(f"Loaded image: {img.size}")
            return img, None
            
        except requests.RequestException as e:
            error_msg = f"Download failed: {type(e).__name__}: {str(e)}"
            logger.warning(f"Failed to load {path_or_url[:50]}...: {error_msg}")
            self.stats['failed'] += 1
            return None, error_msg
            
        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}"
            logger.warning(f"Failed to load {path_or_url[:50]}...: {error_msg}")
            self.stats['failed'] += 1
            return None, error_msg
    
    def load_images_for_history(
        self,
        item_ids: List[int],
        image_url_dict: dict,
        max_images: int = 3
    ) -> List[Tuple[int, Image.Image]]:
        """
        Load images for user history items, looking back if images are missing.
        
        This implements the fallback logic: if an item doesn't have an image,
        try the previous item in history, and so on.
        
        Args:
            item_ids: List of item IDs in chronological order (oldest first)
            image_url_dict: Dictionary mapping item_id -> image_url
            max_images: Maximum number of images to load (default: 3)
            
        Returns:
            List of (item_id, PIL.Image) tuples for successfully loaded images,
            in chronological order. May be fewer than max_images if not enough
            images are available.
        """
        images = []
        
        # Iterate from most recent to oldest
        for item_id in reversed(item_ids):
            if len(images) >= max_images:
                break
            
            # Check if item has an image URL
            image_url = image_url_dict.get(item_id)
            if not image_url:
                logger.debug(f"Item {item_id}: No image URL in dict")
                continue
            
            # Try to load the image
            img, error = self.load_image(image_url)
            if img is not None:
                images.append((item_id, img))
                logger.debug(f"Item {item_id}: Image loaded successfully")
            else:
                logger.debug(f"Item {item_id}: Failed to load - {error}")
        
        # Return in chronological order (oldest first among selected)
        images = list(reversed(images))
        
        logger.info(f"Loaded {len(images)}/{max_images} images from history of {len(item_ids)} items")
        return images
    
    def get_stats(self) -> dict:
        """Get loading statistics."""
        stats = self.stats.copy()
        if stats['total_requests'] > 0:
            stats['success_rate'] = stats['successful'] / stats['total_requests']
        else:
            stats['success_rate'] = 0.0
        return stats
    
    def reset_stats(self):
        """Reset loading statistics."""
        self.stats = {
            'total_requests': 0,
            'successful': 0,
            'failed': 0,
            'retries': 0,
            'cache_hits': 0,
        }


def test_image_loader():
    """Test the image loader with a sample Amazon image."""
    import json
    import gzip
    
    logging.basicConfig(level=logging.DEBUG)
    
    # Load image URL dict
    dict_path = './data/amazon/All_Beauty_image_url_dict.json.gz'
    try:
        with gzip.open(dict_path, 'rt', encoding='utf-8') as f:
            image_url_dict = json.load(f)
        image_url_dict = {int(k): v for k, v in image_url_dict.items()}
        print(f"Loaded {len(image_url_dict)} image URLs")
    except FileNotFoundError:
        print(f"Image URL dict not found at {dict_path}")
        return
    
    # Test single image load
    loader = ImageLoader(timeout=10, max_retries=2)
    
    # Get first item with an image
    sample_id = list(image_url_dict.keys())[0]
    sample_url = image_url_dict[sample_id]
    
    print(f"\nTesting single image load:")
    print(f"  Item ID: {sample_id}")
    print(f"  URL: {sample_url}")
    
    img, error = loader.load_image(sample_url)
    if img:
        print(f"  SUCCESS: {img.size} {img.mode}")
    else:
        print(f"  FAILED: {error}")
    
    # Test history loading
    print(f"\nTesting history image loading:")
    sample_history = list(image_url_dict.keys())[:10]  # First 10 items
    images = loader.load_images_for_history(sample_history, image_url_dict, max_images=3)
    
    print(f"  Requested: 3 images from 10 items")
    print(f"  Loaded: {len(images)} images")
    for item_id, img in images:
        print(f"    Item {item_id}: {img.size}")
    
    # Print stats
    print(f"\nLoader statistics:")
    for k, v in loader.get_stats().items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    test_image_loader()
