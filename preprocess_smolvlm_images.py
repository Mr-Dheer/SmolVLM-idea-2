"""
Preprocess Amazon data for A-LLMRec + SmolVLM integration.

This script creates all necessary mappings from raw Amazon data:
1. itemid_to_asin.pkl - Item ID to ASIN mapping
2. {dataset}_text_name_dict.json.gz - Title/description mapping (for A-LLMRec)
3. {dataset}_image_url_dict.json.gz - Image URL mapping (for SmolVLM)

Usage:
    python preprocess_smolvlm_images.py \
        --review-path ./data/amazon/All_Beauty.json.gz \
        --meta-path ./data/amazon/meta_All_Beauty.json \
        --output-dir ./data/amazon \
        --dataset-name All_Beauty

Output:
    - All_Beauty_itemid_to_asin.pkl: {item_id: asin} mapping
    - All_Beauty_text_name_dict.json.gz: {title: {item_id: title}, description: {item_id: desc}}
    - All_Beauty_image_url_dict.json.gz: {item_id: image_url} mapping
    - All_Beauty_preprocess_stats.json: Statistics
"""

import os
import json
import gzip
import pickle
import argparse
import logging
from tqdm import tqdm
from datetime import datetime
from collections import defaultdict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def parse_gzip_json(path: str):
    """
    Generator to parse gzipped JSON lines file.
    
    Args:
        path: Path to .json.gz file
        
    Yields:
        Parsed JSON objects
    """
    logger.info(f"Parsing: {path}")
    with gzip.open(path, 'rb') as f:
        for line in tqdm(f, desc="Reading reviews"):
            try:
                yield json.loads(line.decode('utf-8'))
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse line: {e}")
                continue


def parse_json(path: str):
    """
    Generator to parse JSON lines file (not gzipped).
    
    Args:
        path: Path to .json file
        
    Yields:
        Parsed JSON objects
    """
    logger.info(f"Parsing: {path}")
    with open(path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Reading file"):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse line: {e}")
                continue


def load_reviews(review_path: str):
    """
    Load review data from JSON file (handles both .json and .json.gz).
    
    Returns:
        Generator of review objects
    """
    if review_path.endswith('.gz'):
        return parse_gzip_json(review_path)
    else:
        return parse_json(review_path)


def load_metadata(meta_path: str) -> dict:
    """
    Load metadata JSON file into dictionary.
    
    Args:
        meta_path: Path to meta_*.json or meta_*.json.gz
        
    Returns:
        Dictionary mapping asin -> metadata dict
    """
    logger.info(f"Loading metadata from: {meta_path}")
    
    meta_dict = {}
    
    if meta_path.endswith('.gz'):
        parser = parse_gzip_json(meta_path)
    else:
        parser = parse_json(meta_path)
    
    for item in parser:
        asin = item.get('asin')
        if asin:
            meta_dict[asin] = item
    
    logger.info(f"Loaded metadata for {len(meta_dict):,} items")
    return meta_dict


def extract_image_url(meta: dict) -> str:
    """
    Extract the best available image URL from metadata.
    
    Priority order:
    1. imageURLHighRes (highest quality)
    2. imUrl
    3. imageURL
    
    Args:
        meta: Item metadata dictionary
        
    Returns:
        Image URL string or None if no image available
    """
    image_url = None
    
    # Try imageURLHighRes first (highest quality)
    high_res = meta.get('imageURLHighRes')
    if high_res:
        if isinstance(high_res, list) and high_res:
            if isinstance(high_res[0], str) and high_res[0].strip():
                image_url = high_res[0].strip()
        elif isinstance(high_res, str) and high_res.strip():
            image_url = high_res.strip()
    
    # Try imUrl if no high-res
    if not image_url:
        im_url = meta.get('imUrl')
        if isinstance(im_url, str) and im_url.strip():
            image_url = im_url.strip()
    
    # Try imageURL as fallback
    if not image_url:
        img_url = meta.get('imageURL')
        if img_url:
            if isinstance(img_url, list) and img_url:
                if isinstance(img_url[0], str) and img_url[0].strip():
                    image_url = img_url[0].strip()
            elif isinstance(img_url, str) and img_url.strip():
                image_url = img_url.strip()
    
    return image_url


def extract_description(meta: dict) -> str:
    """
    Extract description from metadata, handling various formats.
    
    Args:
        meta: Item metadata dictionary
        
    Returns:
        Description string or 'Empty description' if not found
    """
    desc = meta.get('description')
    
    if desc:
        if isinstance(desc, list) and desc:
            # Join all descriptions
            return " ".join(str(d).strip() for d in desc if d)
        elif isinstance(desc, str) and desc.strip():
            return desc.strip()
    
    # Try feature field as fallback
    features = meta.get('feature', [])
    if isinstance(features, list) and features:
        return ". ".join(str(f).strip() for f in features if f)
    
    return 'Empty description'


def preprocess_data(
    review_path: str,
    meta_path: str,
    output_dir: str,
    dataset_name: str
) -> dict:
    """
    Main preprocessing function - creates all necessary mappings.
    
    This follows the same logic as A-LLMRec's data_preprocess.py to ensure
    consistent item IDs.
    
    Args:
        review_path: Path to review JSON file
        meta_path: Path to metadata JSON file
        output_dir: Output directory
        dataset_name: Dataset name (e.g., 'All_Beauty')
        
    Returns:
        Statistics dictionary
    """
    logger.info("=" * 70)
    logger.info("Starting preprocessing")
    logger.info("=" * 70)
    
    # Statistics
    stats = {
        'dataset_name': dataset_name,
        'timestamp': datetime.now().isoformat(),
        'review_path': review_path,
        'meta_path': meta_path,
    }
    
    # Determine threshold based on dataset name (same as A-LLMRec)
    if ('Beauty' in dataset_name) or ('Toys' in dataset_name):
        threshold = 4
    else:
        threshold = 5
    logger.info(f"Using interaction threshold: {threshold}")
    
    # =========================================================================
    # Step 1: Count interactions per user and item
    # =========================================================================
    logger.info("\n[Step 1/5] Counting interactions...")
    
    countU = defaultdict(int)
    countP = defaultdict(int)
    
    # First pass: count interactions
    for review in load_reviews(review_path):
        asin = review.get('asin')
        rev = review.get('reviewerID')
        if asin and rev:
            countU[rev] += 1
            countP[asin] += 1
    
    logger.info(f"Found {len(countU):,} users and {len(countP):,} items before filtering")
    stats['users_before_filter'] = len(countU)
    stats['items_before_filter'] = len(countP)
    
    # =========================================================================
    # Step 2: Load metadata
    # =========================================================================
    logger.info("\n[Step 2/5] Loading metadata...")
    meta_dict = load_metadata(meta_path)
    stats['metadata_items'] = len(meta_dict)
    
    # =========================================================================
    # Step 3: Build mappings (second pass through reviews)
    # =========================================================================
    logger.info("\n[Step 3/5] Building item mappings...")
    
    usermap = {}
    usernum = 0
    itemmap = {}  # asin -> item_id
    itemnum = 0
    User = {}  # user_id -> list of (timestamp, item_id)
    
    # Data structures for output
    name_dict = {'title': {}, 'description': {}}
    image_url_dict = {}
    itemid_to_asin = {}
    
    # Track statistics
    items_with_title = 0
    items_with_description = 0
    items_with_image = 0
    items_missing_metadata = 0
    
    # Second pass: build mappings
    for review in load_reviews(review_path):
        asin = review.get('asin')
        rev = review.get('reviewerID')
        time = review.get('unixReviewTime', 0)
        
        if not asin or not rev:
            continue
        
        # Apply threshold filter (same as A-LLMRec)
        if countU[rev] < threshold or countP[asin] < threshold:
            continue
        
        # Create user mapping
        if rev in usermap:
            userid = usermap[rev]
        else:
            usernum += 1
            userid = usernum
            usermap[rev] = userid
            User[userid] = []
        
        # Create item mapping
        if asin in itemmap:
            itemid = itemmap[asin]
        else:
            itemnum += 1
            itemid = itemnum
            itemmap[asin] = itemid
            itemid_to_asin[itemid] = asin
            
            # Extract metadata for this item
            if asin in meta_dict:
                meta = meta_dict[asin]
                
                # Title
                title = meta.get('title', f'Product {asin}')
                if title:
                    name_dict['title'][itemid] = title
                    items_with_title += 1
                
                # Description
                description = extract_description(meta)
                name_dict['description'][itemid] = description
                if description != 'Empty description':
                    items_with_description += 1
                
                # Image URL
                image_url = extract_image_url(meta)
                if image_url:
                    image_url_dict[itemid] = image_url
                    items_with_image += 1
            else:
                items_missing_metadata += 1
                name_dict['title'][itemid] = f'Product {asin}'
                name_dict['description'][itemid] = 'Empty description'
        
        # Add interaction
        User[userid].append([time, itemid])
    
    # Sort user interactions by time
    for userid in User.keys():
        User[userid].sort(key=lambda x: x[0])
    
    logger.info(f"After filtering: {usernum:,} users, {itemnum:,} items")
    stats['users_after_filter'] = usernum
    stats['items_after_filter'] = itemnum
    stats['items_with_title'] = items_with_title
    stats['items_with_description'] = items_with_description
    stats['items_with_image'] = items_with_image
    stats['items_missing_metadata'] = items_missing_metadata
    
    # =========================================================================
    # Step 4: Save all outputs
    # =========================================================================
    logger.info("\n[Step 4/5] Saving outputs...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save itemid_to_asin.pkl
    itemmap_path = os.path.join(output_dir, f'{dataset_name}_itemid_to_asin.pkl')
    with open(itemmap_path, 'wb') as f:
        pickle.dump(itemid_to_asin, f)
    logger.info(f"  Saved: {itemmap_path} ({len(itemid_to_asin):,} items)")
    
    # Save text_name_dict.json.gz (pickle format, matching A-LLMRec)
    text_dict_path = os.path.join(output_dir, f'{dataset_name}_text_name_dict.json.gz')
    with open(text_dict_path, 'wb') as f:
        pickle.dump(name_dict, f)
    logger.info(f"  Saved: {text_dict_path}")
    
    # Save image_url_dict.json.gz (JSON format for easy inspection)
    image_dict_path = os.path.join(output_dir, f'{dataset_name}_image_url_dict.json.gz')
    with gzip.open(image_dict_path, 'wt', encoding='utf-8') as f:
        json.dump(image_url_dict, f)
    logger.info(f"  Saved: {image_dict_path} ({len(image_url_dict):,} items with images)")
    
    # Save interaction file (All_Beauty.txt format)
    interaction_path = os.path.join(output_dir, f'{dataset_name}.txt')
    with open(interaction_path, 'w') as f:
        for userid in User.keys():
            for time, itemid in User[userid]:
                f.write(f'{userid} {itemid}\n')
    logger.info(f"  Saved: {interaction_path}")
    
    # =========================================================================
    # Step 5: Save statistics
    # =========================================================================
    logger.info("\n[Step 5/5] Saving statistics...")
    
    stats['output_files'] = {
        'itemid_to_asin': itemmap_path,
        'text_name_dict': text_dict_path,
        'image_url_dict': image_dict_path,
        'interactions': interaction_path
    }
    
    stats_path = os.path.join(output_dir, f'{dataset_name}_preprocess_stats.json')
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    logger.info(f"  Saved: {stats_path}")
    
    return stats


def verify_outputs(output_dir: str, dataset_name: str):
    """
    Verify the created files by loading and sampling.
    """
    logger.info("\n" + "=" * 70)
    logger.info("Verifying outputs")
    logger.info("=" * 70)
    
    # Verify itemid_to_asin
    itemmap_path = os.path.join(output_dir, f'{dataset_name}_itemid_to_asin.pkl')
    with open(itemmap_path, 'rb') as f:
        itemid_to_asin = pickle.load(f)
    logger.info(f"\nitemid_to_asin.pkl:")
    logger.info(f"  Total items: {len(itemid_to_asin):,}")
    logger.info(f"  ID range: {min(itemid_to_asin.keys())} to {max(itemid_to_asin.keys())}")
    logger.info(f"  Sample: {dict(list(itemid_to_asin.items())[:3])}")
    
    # Verify text_name_dict
    text_dict_path = os.path.join(output_dir, f'{dataset_name}_text_name_dict.json.gz')
    with open(text_dict_path, 'rb') as f:
        name_dict = pickle.load(f)
    logger.info(f"\ntext_name_dict.json.gz:")
    logger.info(f"  Titles: {len(name_dict['title']):,}")
    logger.info(f"  Descriptions: {len(name_dict['description']):,}")
    sample_id = list(name_dict['title'].keys())[0]
    logger.info(f"  Sample (id={sample_id}):")
    logger.info(f"    Title: {name_dict['title'][sample_id][:60]}...")
    logger.info(f"    Desc: {name_dict['description'][sample_id][:60]}...")
    
    # Verify image_url_dict
    image_dict_path = os.path.join(output_dir, f'{dataset_name}_image_url_dict.json.gz')
    with gzip.open(image_dict_path, 'rt', encoding='utf-8') as f:
        image_url_dict = json.load(f)
    # Convert keys to int
    image_url_dict = {int(k): v for k, v in image_url_dict.items()}
    logger.info(f"\nimage_url_dict.json.gz:")
    logger.info(f"  Items with images: {len(image_url_dict):,}")
    if image_url_dict:
        sample_id = list(image_url_dict.keys())[0]
        sample_url = image_url_dict[sample_id]
        logger.info(f"  Sample (id={sample_id}): {sample_url[:70]}...")


def main():
    parser = argparse.ArgumentParser(
        description='Preprocess Amazon data for A-LLMRec + SmolVLM'
    )
    
    parser.add_argument(
        '--review-path',
        type=str,
        required=True,
        help='Path to review data (All_Beauty.json or All_Beauty.json.gz)'
    )
    parser.add_argument(
        '--meta-path',
        type=str,
        required=True,
        help='Path to metadata file (meta_All_Beauty.json or .json.gz)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./data/amazon',
        help='Output directory (default: ./data/amazon)'
    )
    parser.add_argument(
        '--dataset-name',
        type=str,
        default='All_Beauty',
        help='Dataset name for output files (default: All_Beauty)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose (DEBUG) logging'
    )
    
    args = parser.parse_args()
    
    # Set log level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    logger.info("=" * 70)
    logger.info("A-LLMRec + SmolVLM Data Preprocessing")
    logger.info("=" * 70)
    logger.info(f"\nConfiguration:")
    logger.info(f"  Review data:   {args.review_path}")
    logger.info(f"  Metadata:      {args.meta_path}")
    logger.info(f"  Output dir:    {args.output_dir}")
    logger.info(f"  Dataset name:  {args.dataset_name}")
    
    # Run preprocessing
    stats = preprocess_data(
        review_path=args.review_path,
        meta_path=args.meta_path,
        output_dir=args.output_dir,
        dataset_name=args.dataset_name
    )
    
    # Verify outputs
    verify_outputs(args.output_dir, args.dataset_name)
    
    # Print summary
    logger.info("\n" + "=" * 70)
    logger.info("PREPROCESSING COMPLETE")
    logger.info("=" * 70)
    logger.info(f"\nSummary:")
    logger.info(f"  Users:              {stats['users_after_filter']:,}")
    logger.info(f"  Items:              {stats['items_after_filter']:,}")
    logger.info(f"  Items with images:  {stats['items_with_image']:,} ({100*stats['items_with_image']/stats['items_after_filter']:.1f}%)")
    logger.info(f"\nOutput files:")
    for name, path in stats['output_files'].items():
        logger.info(f"  - {path}")
    logger.info("\nNext step: Run Stage-1 training or create utils_image.py")


if __name__ == "__main__":
    main()
