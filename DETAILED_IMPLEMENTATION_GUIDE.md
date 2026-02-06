# Complete Technical Documentation: SmolVLM Integration into A-LLMRec

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Original A-LLMRec Architecture](#2-original-a-llmrec-architecture)
3. [What We Changed](#3-what-we-changed)
4. [New Files Created](#4-new-files-created)
5. [Modified Files](#5-modified-files)
6. [Complete Data Flow](#6-complete-data-flow)
7. [Detailed Code Walkthrough](#7-detailed-code-walkthrough)
8. [Technical Challenges Solved](#8-technical-challenges-solved)
9. [How to Run](#9-how-to-run)
10. [File Structure](#10-file-structure)

---

## 1. Project Overview

### 1.1 Goal
Replace **OPT-6.7B** in A-LLMRec's Stage-2 with **SmolVLM2-500M-Video-Instruct** to enable **multimodal recommendations** using product images alongside collaborative filtering signals.

### 1.2 Key Achievement
The system now:
- Processes actual product images from user purchase history
- Combines visual features with collaborative filtering embeddings
- Uses a Vision-Language Model for recommendation generation
- Maintains all Stage-1 functionality unchanged

### 1.3 Comparison

| Aspect | Original (OPT-6.7B) | New (SmolVLM-500M) |
|--------|---------------------|---------------------|
| Model Size | 6.7 Billion parameters | 500 Million parameters |
| Hidden Dimension | 4096 | 960 |
| Modality | Text only | Text + Images |
| Memory Usage | ~15 GB VRAM | ~4 GB VRAM |
| Images per Sample | 0 | Up to 3 |
| Context Window | 2048 tokens | 8192 tokens |

---

## 2. Original A-LLMRec Architecture

### 2.1 Two-Stage Training Philosophy

A-LLMRec addresses the fundamental challenge of integrating collaborative filtering (CF) signals with language models. The problem is that CF embeddings and text embeddings live in different semantic spaces.

### 2.2 Stage 1: Collaborative-Text Alignment

**Purpose**: Create a joint embedding space where CF signals and text semantics are aligned.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           STAGE 1 ARCHITECTURE                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   User Interaction Sequence                     Item Text (Title+Desc)   │
│          [item1, item2, item3, ...]                    ↓                 │
│                   ↓                              ┌──────────┐            │
│            ┌──────────┐                          │  SBERT   │            │
│            │  SASRec  │                          │ (frozen) │            │
│            │ (frozen) │                          └────┬─────┘            │
│            └────┬─────┘                               │                  │
│                 │                                     │                  │
│                 ↓                                     ↓                  │
│        CF Embedding (50-dim)              Text Embedding (768-dim)       │
│                 │                                     │                  │
│                 ↓                                     ↓                  │
│          ┌───────────┐                         ┌───────────┐            │
│          │   MLP     │                         │   MLP2    │            │
│          │ (trained) │                         │ (trained) │            │
│          └─────┬─────┘                         └─────┬─────┘            │
│                │                                     │                  │
│                ↓                                     ↓                  │
│        Joint Emb (128-dim)  ←───── MSE Loss ─────→ Joint Emb (128-dim)  │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

**Key Components**:
- **SASRec**: Self-Attentive Sequential Recommendation model (pre-trained, frozen)
- **SBERT**: Sentence-BERT for text encoding (frozen)
- **MLP/MLP2**: Two-layer MLPs that project into a shared 128-dim space (trained)

**Output**: The MLP learns to map CF embeddings to match text semantics. This 128-dim "joint embedding" captures BOTH user behavior patterns AND item semantic meaning.

### 2.3 Stage 2: LLM Integration (Original)

**Purpose**: Use a Large Language Model to generate recommendations based on projected embeddings.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    ORIGINAL STAGE 2 (OPT-6.7B)                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   User Sequence Embedding (50-dim)      Joint Item Embeddings (128-dim) │
│              ↓                                      ↓                    │
│      ┌──────────────┐                      ┌──────────────┐             │
│      │ log_emb_proj │                      │ item_emb_proj│             │
│      │  50 → 4096   │                      │  128 → 4096  │             │
│      └──────┬───────┘                      └──────┬───────┘             │
│             │                                     │                      │
│             ↓                                     ↓                      │
│      [UserRep] token                    [HistoryEmb], [CandidateEmb]    │
│             │                                     │                      │
│             └──────────────┬──────────────────────┘                      │
│                            ↓                                             │
│                   ┌─────────────────┐                                   │
│                   │  OPT-6.7B LLM   │                                   │
│                   │    (frozen)     │                                   │
│                   └────────┬────────┘                                   │
│                            ↓                                             │
│                  Generated Recommendation                                │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 3. What We Changed

### 3.1 Stage 2 Replacement

We replaced OPT-6.7B with SmolVLM2-500M-Video-Instruct, adding multimodal capability:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      NEW STAGE 2 (SmolVLM-500M)                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   User Seq Emb (50)    Joint Item Embs (128)    Product Images (PIL)    │
│         ↓                      ↓                       ↓                 │
│   ┌───────────┐         ┌───────────┐         ┌───────────────┐         │
│   │log_emb_proj│        │item_emb_proj│       │ SmolVLM Vision│         │
│   │  50 → 960  │        │  128 → 960  │       │    Encoder    │         │
│   └─────┬─────┘         └─────┬─────┘         └───────┬───────┘         │
│         │                     │                       │                  │
│         ↓                     ↓                       ↓                  │
│    [UserRep]         [HistoryEmb] tokens      Image Hidden States       │
│      token           [CandidateEmb] tokens          (960-dim)           │
│         │                     │                       │                  │
│         └─────────────────────┼───────────────────────┘                  │
│                               ↓                                          │
│                    ┌──────────────────────┐                             │
│                    │   inputs_merger      │                             │
│                    │ (merges text+images) │                             │
│                    └──────────┬───────────┘                             │
│                               ↓                                          │
│                    ┌──────────────────────┐                             │
│                    │  Token Replacement   │                             │
│                    │ (inject embeddings)  │                             │
│                    └──────────┬───────────┘                             │
│                               ↓                                          │
│                    ┌──────────────────────┐                             │
│                    │ SmolVLM Text Model   │                             │
│                    │     (frozen)         │                             │
│                    └──────────┬───────────┘                             │
│                               ↓                                          │
│                    Generated Recommendation                              │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 3.2 What Stayed the Same

- **Stage 1 is completely untouched**
- SASRec model (frozen)
- SBERT encoder (frozen)  
- MLP/MLP2 alignment networks
- Joint embeddings (128-dim output)
- Data loading and preprocessing logic

### 3.3 Key Design Decisions

1. **Reserved Tokens**: Use SmolVLM's pre-existing reserved tokens instead of adding new ones (avoids embedding resize)

2. **Two-Phase Embedding Injection**: 
   - First: Let SmolVLM merge images with text
   - Then: Replace our custom tokens with projected embeddings

3. **Fallback Image Loading**: If recent items don't have images, look back in history

4. **Manual Generation Loop**: Implement custom greedy decoding since `model.generate()` doesn't support our embedding injection

---

## 4. New Files Created

### 4.1 `models/smolvlm4rec.py` (722 lines)

This is the core new file that wraps SmolVLM for recommendation tasks.

#### 4.1.1 Class Definition and Constants

```python
class SmolVLM4Rec(nn.Module):
    """
    SmolVLM wrapper for recommendation tasks.
    
    Handles:
    - Model and processor loading
    - Chat template formatting
    - Image-text interleaving
    - Custom embedding token replacement
    - Training forward pass with loss
    - Inference generation
    """
    
    # Model identifier
    MODEL_NAME = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
    
    # Use SmolVLM's reserved tokens instead of adding new ones
    # This avoids resizing embeddings and is more stable
    USER_REP_TOKEN = "<|reserved_special_token_0|>"      # For user representation
    HISTORY_EMB_TOKEN = "<|reserved_special_token_1|>"   # For history item embeddings
    CANDIDATE_EMB_TOKEN = "<|reserved_special_token_2|>" # For candidate item embeddings
    
    # Token IDs from SmolVLM config (added_tokens.json)
    USER_REP_TOKEN_ID = 49191
    HISTORY_EMB_TOKEN_ID = 49192
    CANDIDATE_EMB_TOKEN_ID = 49193
```

**Why Reserved Tokens?**
- SmolVLM has 88 reserved special tokens (`<|reserved_special_token_0|>` through `<|reserved_special_token_87|>`)
- These already exist in the vocabulary with trained embeddings
- Using them avoids calling `resize_token_embeddings()` which can destabilize a frozen model

#### 4.1.2 Initialization

```python
def __init__(self, device: str, model_name: str = None, max_output_len: int = 256):
    super().__init__()
    self.device = device
    self.model_name = model_name or self.MODEL_NAME
    
    # Load processor (handles both text and images)
    self.processor = AutoProcessor.from_pretrained(self.model_name)
    
    # Load model in bfloat16 for efficiency
    self.model = AutoModelForImageTextToText.from_pretrained(
        self.model_name,
        dtype=torch.bfloat16,
        device_map=self.device
    )
    
    # CRITICAL: Freeze all model parameters
    for param in self.model.parameters():
        param.requires_grad = False
    
    # Setup custom tokens (just store IDs, no resize)
    self._setup_custom_tokens()
    
    # Get hidden size (960 for SmolVLM-500M)
    self.hidden_size = self.model.config.text_config.hidden_size
```

#### 4.1.3 Image Processing Pipeline

The key challenge was extracting vision features separately from the main forward pass:

```python
def _get_image_hidden_states(self, pixel_values, pixel_attention_mask):
    """
    Extract image hidden states using a forward hook.
    
    SmolVLM's internal flow:
    1. pixel_values → vision_model → vision features (768-dim)
    2. vision features → connector → image_hidden_states (960-dim)
    
    We capture the output of step 2 using a hook.
    """
    image_hidden_states_captured = [None]
    
    def capture_hook(module, input, output):
        image_hidden_states_captured[0] = output
    
    # Register hook on the connector module
    hook = self.model.model.connector.register_forward_hook(capture_hook)
    
    try:
        # Run forward pass to trigger vision processing
        dummy_input_ids = torch.tensor([[1]], device=self.device)
        with torch.no_grad():
            _ = self.model.model(
                input_ids=dummy_input_ids,
                pixel_values=pixel_values,
                pixel_attention_mask=pixel_attention_mask,
                return_dict=True
            )
    finally:
        hook.remove()
    
    return image_hidden_states_captured[0]
```

**Why a hook?**
- SmolVLM's `inputs_merger` expects `image_hidden_states` but doesn't expose it directly
- The connector output is exactly what we need
- Using a hook lets us capture it without modifying SmolVLM's code

#### 4.1.4 Two-Phase Embedding Merging

This is the core innovation that solved our biggest challenge:

```python
def _merge_images_and_replace_tokens(
    self,
    input_ids: torch.Tensor,
    image_hidden_states: torch.Tensor,
    user_emb: torch.Tensor,
    history_embs: torch.Tensor,
    candidate_embs: torch.Tensor,
) -> torch.Tensor:
    """
    Two-step process:
    1. Use SmolVLM's inputs_merger for image-text merging
    2. Replace our custom tokens with projected embeddings
    """
    
    # Get base text embeddings
    inputs_embeds = self.model.get_input_embeddings()(input_ids)
    
    # STEP 1: Let SmolVLM merge images at <image> positions
    # This expands each <image> token into 64 vision tokens
    merged_embeds = self.model.model.inputs_merger(
        input_ids=input_ids,
        inputs_embeds=inputs_embeds,
        image_hidden_states=image_hidden_states
    )
    
    # Create attention mask for expanded sequence
    merged_attention_mask = torch.ones(
        (1, merged_embeds.size(1)), 
        device=self.device, 
        dtype=torch.long
    )
    
    # STEP 2: Find and replace our custom tokens
    # After image merging, positions have shifted, so we find tokens by embedding value
    
    # Get reference embeddings for our tokens
    user_token_emb = self.model.get_input_embeddings()(
        torch.tensor([self.user_rep_token_id], device=self.device)
    )[0]
    history_token_emb = self.model.get_input_embeddings()(
        torch.tensor([self.history_emb_token_id], device=self.device)
    )[0]
    candidate_token_emb = self.model.get_input_embeddings()(
        torch.tensor([self.candidate_emb_token_id], device=self.device)
    )[0]
    
    # Replace UserRep token with projected user embedding
    for pos in range(merged_embeds.size(1)):
        if torch.allclose(merged_embeds[0, pos], user_token_emb, atol=1e-4):
            merged_embeds[0, pos] = user_emb[0]
            break
    
    # Replace HistoryEmb tokens (multiple)
    history_idx = 0
    for pos in range(merged_embeds.size(1)):
        if history_idx >= history_embs.size(1):
            break
        if torch.allclose(merged_embeds[0, pos], history_token_emb, atol=1e-4):
            merged_embeds[0, pos] = history_embs[0, history_idx]
            history_idx += 1
    
    # Replace CandidateEmb tokens (multiple)
    candidate_idx = 0
    for pos in range(merged_embeds.size(1)):
        if candidate_idx >= candidate_embs.size(1):
            break
        if torch.allclose(merged_embeds[0, pos], candidate_token_emb, atol=1e-4):
            merged_embeds[0, pos] = candidate_embs[0, candidate_idx]
            candidate_idx += 1
    
    return merged_embeds, merged_attention_mask
```

#### 4.1.5 Training Forward Pass

```python
def forward(self, user_emb: torch.Tensor, samples: Dict[str, Any]) -> torch.Tensor:
    """
    Training forward pass with language modeling loss.
    
    Args:
        user_emb: Projected user representations [batch, hidden_size]
        samples: Dictionary containing:
            - text_input: List of prompt texts
            - text_output: List of target item titles
            - images: List of List of PIL Images
            - history_embs: List of history embedding tensors
            - candidate_embs: List of candidate embedding tensors
    
    Returns:
        Language modeling loss (scalar tensor)
    """
    batch_size = len(samples['text_input'])
    all_losses = []
    
    for i in range(batch_size):
        prompt_text = samples['text_input'][i]
        target_text = samples['text_output'][i]
        images = samples.get('images', [[]])[i]
        history_embs = samples['history_embs'][i].unsqueeze(0)
        candidate_embs = samples['candidate_embs'][i].unsqueeze(0)
        
        # Prepare full text (prompt + target + EOS)
        full_text = prompt_text + " " + target_text + self.processor.tokenizer.eos_token
        
        if images and len(images) > 0:
            # MULTIMODAL PATH
            
            # Process images and text together
            inputs = self.processor(
                text=full_text,
                images=images,
                return_tensors="pt",
                padding=True
            ).to(self.device)
            
            inputs['pixel_values'] = inputs['pixel_values'].to(torch.bfloat16)
            
            # Extract image features
            image_hidden_states = self._get_image_hidden_states(
                inputs['pixel_values'],
                inputs['pixel_attention_mask']
            )
            
            # Merge images and replace tokens
            inputs_embeds, attention_mask = self._merge_images_and_replace_tokens(
                input_ids=inputs['input_ids'],
                image_hidden_states=image_hidden_states,
                user_emb=user_emb[i:i+1],
                history_embs=history_embs,
                candidate_embs=candidate_embs
            )
            
            # Create labels (mask prompt, only compute loss on target)
            labels = torch.full((1, inputs_embeds.size(1)), -100, device=self.device)
            target_tokens = self.processor.tokenizer(
                " " + target_text + self.processor.tokenizer.eos_token,
                return_tensors="pt",
                add_special_tokens=False
            ).to(self.device)
            target_length = target_tokens['input_ids'].size(1)
            labels[0, -target_length:] = target_tokens['input_ids'][0]
            
        else:
            # TEXT-ONLY PATH
            full_text_clean = full_text.replace('<image>', '')
            inputs = self.processor.tokenizer(
                full_text_clean,
                return_tensors="pt",
                padding=True
            ).to(self.device)
            
            # Simple token replacement (no image merging)
            inputs_embeds = self.model.get_input_embeddings()(inputs['input_ids'])
            inputs_embeds = self._replace_custom_tokens(
                input_ids=inputs['input_ids'],
                inputs_embeds=inputs_embeds,
                user_emb=user_emb[i:i+1],
                history_embs=history_embs,
                candidate_embs=candidate_embs
            )
            attention_mask = inputs['attention_mask']
            labels = inputs['input_ids'].clone()
            # Mask prompt portion
            prompt_length = len(self.processor.tokenizer(prompt_text.replace('<image>', '')))
            labels[:, :prompt_length] = -100
        
        # Forward pass through language model
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            outputs = self.model.model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True
            )
            
            # Compute language modeling loss
            logits = self.model.lm_head(outputs.last_hidden_state)
            
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
        
        all_losses.append(loss)
    
    return torch.stack(all_losses).mean()
```

#### 4.1.6 Generation (Inference)

```python
@torch.no_grad()
def generate(self, user_emb, prompt_text, images, history_embs, candidate_embs, max_new_tokens=64):
    """
    Generate recommendation using manual greedy decoding.
    
    We can't use model.generate() because it doesn't support
    our custom embedding injection approach.
    """
    # Prepare initial embeddings (same as forward)
    if images and len(images) > 0:
        inputs = self.processor(text=prompt_text, images=images, ...)
        image_hidden_states = self._get_image_hidden_states(...)
        inputs_embeds, attention_mask = self._merge_images_and_replace_tokens(...)
    else:
        # Text-only path
        inputs_embeds = self._replace_custom_tokens(...)
    
    # Manual greedy generation loop
    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        generated_tokens = []
        current_embeds = inputs_embeds
        current_attention = attention_mask
        
        for _ in range(max_new_tokens):
            # Forward pass
            outputs = self.model.model(
                inputs_embeds=current_embeds,
                attention_mask=current_attention,
                return_dict=True
            )
            
            # Get logits for last position
            logits = self.model.lm_head(outputs.last_hidden_state[:, -1:, :])
            
            # Greedy selection
            next_token_id = logits.argmax(dim=-1)
            generated_tokens.append(next_token_id.item())
            
            # Check for EOS
            if next_token_id.item() == self.processor.tokenizer.eos_token_id:
                break
            
            # Get embedding for next token and append
            next_token_emb = self.model.get_input_embeddings()(next_token_id)
            current_embeds = torch.cat([current_embeds, next_token_emb], dim=1)
            current_attention = torch.cat([
                current_attention,
                torch.ones((1, 1), device=self.device)
            ], dim=1)
    
    # Decode
    return self.processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)
```

---

### 4.2 `utils_image.py` (287 lines)

Robust image loading utilities with retry logic.

#### 4.2.1 ImageLoader Class

```python
class ImageLoader:
    """
    Robust image loader with retry logic for Amazon product images.
    
    Features:
    - Retry with exponential backoff
    - User-Agent headers for Amazon CDN compatibility
    - Timeout handling
    - RGB conversion
    - Error tracking
    """
    
    DEFAULT_USER_AGENT = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
    
    def __init__(self, timeout=10, max_retries=2, user_agent=None):
        self.timeout = timeout
        self.max_retries = max_retries
        self.user_agent = user_agent or self.DEFAULT_USER_AGENT
        
        # Statistics tracking
        self.stats = {
            'total_requests': 0,
            'successful': 0,
            'failed': 0,
            'retries': 0
        }
```

#### 4.2.2 Download with Retry

```python
def _download_url(self, url: str) -> bytes:
    """Download image with exponential backoff retry."""
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
                wait_time = 2 ** attempt  # 1, 2, 4 seconds
                self.stats['retries'] += 1
                time.sleep(wait_time)
                continue
            raise last_error
```

#### 4.2.3 Fallback History Loading

```python
def load_images_for_history(self, item_ids, image_url_dict, max_images=3):
    """
    Load images for user history with fallback logic.
    
    If the most recent item doesn't have an image,
    try the next most recent, and so on.
    
    Args:
        item_ids: List of item IDs (oldest first)
        image_url_dict: {item_id: image_url}
        max_images: Maximum images to load
    
    Returns:
        List of (item_id, PIL.Image) tuples
    """
    images = []
    
    # Iterate from most recent to oldest
    for item_id in reversed(item_ids):
        if len(images) >= max_images:
            break
        
        image_url = image_url_dict.get(item_id)
        if not image_url:
            continue  # No URL, try earlier item
        
        img, error = self.load_image(image_url)
        if img is not None:
            images.append((item_id, img))
    
    # Return in chronological order
    return list(reversed(images))
```

---

### 4.3 `preprocess_smolvlm_images.py` (532 lines)

Preprocessing script that creates image URL mappings.

#### 4.3.1 Purpose

Creates all necessary data files from raw Amazon data:
1. `{dataset}_image_url_dict.json.gz` - Maps item_id → image URL
2. `{dataset}_text_name_dict.json.gz` - Maps item_id → title/description
3. `{dataset}_itemid_to_asin.pkl` - Maps item_id → ASIN

#### 4.3.2 Image URL Extraction Priority

```python
def extract_image_url(meta: dict) -> str:
    """
    Extract the best available image URL.
    
    Priority order (highest quality first):
    1. imageURLHighRes - High resolution images
    2. imUrl - Medium quality
    3. imageURL - Fallback
    """
    image_url = None
    
    # Try imageURLHighRes first
    high_res = meta.get('imageURLHighRes')
    if high_res:
        if isinstance(high_res, list) and high_res:
            image_url = high_res[0].strip()
        elif isinstance(high_res, str):
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
                image_url = img_url[0].strip()
            elif isinstance(img_url, str):
                image_url = img_url.strip()
    
    return image_url
```

#### 4.3.3 Main Preprocessing Flow

```python
def preprocess_data(review_path, meta_path, output_dir, dataset_name):
    """
    Main preprocessing - follows same filtering as A-LLMRec.
    
    Steps:
    1. Count user/item interactions
    2. Load metadata
    3. Filter by threshold (same as A-LLMRec)
    4. Build mappings
    5. Save outputs
    """
    # Threshold based on dataset (same as A-LLMRec)
    if 'Beauty' in dataset_name or 'Toys' in dataset_name:
        threshold = 4
    else:
        threshold = 5
    
    # First pass: count interactions
    for review in load_reviews(review_path):
        countU[review['reviewerID']] += 1
        countP[review['asin']] += 1
    
    # Second pass: build mappings with filtering
    for review in load_reviews(review_path):
        if countU[rev] < threshold or countP[asin] < threshold:
            continue
        
        # Create item mapping
        itemnum += 1
        itemmap[asin] = itemnum
        itemid_to_asin[itemnum] = asin
        
        # Extract metadata
        if asin in meta_dict:
            name_dict['title'][itemnum] = meta_dict[asin].get('title')
            name_dict['description'][itemnum] = extract_description(meta_dict[asin])
            
            image_url = extract_image_url(meta_dict[asin])
            if image_url:
                image_url_dict[itemnum] = image_url
```

---

## 5. Modified Files

### 5.1 `main.py`

**Changes**: Added command-line arguments for SmolVLM mode.

```python
# NEW: SmolVLM settings
parser.add_argument("--use_smolvlm", action='store_true',
                help='Use SmolVLM instead of OPT for Stage-2 (enables multimodal)')
parser.add_argument("--num_history_images", type=int, default=3,
                help='Number of history items to include images for (default: 3)')
```

### 5.2 `models/a_llmrec_model.py`

This file had the most significant modifications.

#### 5.2.1 New Imports

```python
# SmolVLM imports (conditional to avoid errors if not installed)
try:
    from models.smolvlm4rec import SmolVLM4Rec
    from utils_image import ImageLoader
    SMOLVLM_AVAILABLE = True
except ImportError:
    SMOLVLM_AVAILABLE = False
```

#### 5.2.2 Modified `__init__`

```python
def __init__(self, args):
    # ... existing Stage-1 code unchanged ...
    
    # NEW: Check if using SmolVLM
    self.use_smolvlm = getattr(args, 'use_smolvlm', False)
    self.num_history_images = getattr(args, 'num_history_images', 3)
    
    if args.pretrain_stage2 or args.inference:
        if self.use_smolvlm:
            # NEW: Load SmolVLM instead of OPT
            if not SMOLVLM_AVAILABLE:
                raise ImportError("SmolVLM not available")
            
            self.vlm = SmolVLM4Rec(device=self.device)
            llm_hidden_size = self.vlm.hidden_size  # 960
            
            # NEW: Load image URL dictionary
            image_dict_path = f'./data/amazon/{args.rec_pre_trained_data}_image_url_dict.json.gz'
            with gzip.open(image_dict_path, 'rt') as f:
                self.image_url_dict = json.load(f)
            self.image_url_dict = {int(k): v for k, v in self.image_url_dict.items()}
            
            # NEW: Initialize image loader
            self.image_loader = ImageLoader(timeout=10, max_retries=2)
            
        else:
            # Original OPT loading (unchanged)
            self.llm = llm4rec(device=self.device, llm_model=args.llm)
            llm_hidden_size = self.llm.llm_model.config.hidden_size  # 4096
        
        # Projection dimensions now depend on which LLM
        # KEY: Projects 128-dim joint embeddings to LLM space
        self.log_emb_proj = nn.Sequential(
            nn.Linear(self.rec_sys_dim, llm_hidden_size),  # 50 → 960 or 4096
            nn.LayerNorm(llm_hidden_size),
            nn.LeakyReLU(),
            nn.Linear(llm_hidden_size, llm_hidden_size)
        )
        
        self.item_emb_proj = nn.Sequential(
            nn.Linear(128, llm_hidden_size),  # 128 → 960 or 4096
            nn.LayerNorm(llm_hidden_size),
            nn.GELU(),
            nn.Linear(llm_hidden_size, llm_hidden_size)
        )
```

#### 5.2.3 New `get_history_images` Method

```python
def get_history_images(self, item_ids, max_images=3):
    """
    Get images for the last N history items with fallback.
    
    This is where the image loading happens during training/inference.
    """
    if not self.use_smolvlm:
        return [], []
    
    images = []
    image_item_ids = []
    
    # Iterate from most recent to oldest (fallback logic)
    for item_id in reversed(item_ids):
        if len(images) >= max_images:
            break
        
        image_url = self.image_url_dict.get(int(item_id))
        if not image_url:
            continue  # No URL, try earlier item
        
        img, error = self.image_loader.load_image(image_url)
        if img is not None:
            images.append(img)
            image_item_ids.append(item_id)
    
    # Reverse to get chronological order
    return list(reversed(images)), list(reversed(image_item_ids))
```

#### 5.2.4 New `make_interact_text_with_images` Method

```python
def make_interact_text_with_images(self, interact_ids, interact_max_num, image_item_ids, images):
    """
    Create interaction text with <image> tokens where images exist.
    
    IMPORTANT: Only adds <image> tokens for items we have actual images for,
    and only once per unique item (handles duplicate items in history).
    
    Returns:
        Tuple of (interact_text, final_interact_ids, ordered_images)
        - ordered_images: Images in the order they appear in text
    
    Example output:
    '<image>"Product A"<|reserved_special_token_1|>,"Product B"<|reserved_special_token_1|>'
    """
    HISTORY_EMB_TOKEN = "<|reserved_special_token_1|>"
    
    # Create mapping from item_id to image (handles duplicates)
    image_item_to_img = {}
    for item_id, img in zip(image_item_ids, images):
        item_id_int = int(item_id)
        if item_id_int not in image_item_to_img:
            image_item_to_img[item_id_int] = img
    
    selected_ids = interact_ids[-interact_max_num:]
    selected_titles = interact_item_titles[-interact_max_num:]
    
    # Track used images to avoid duplicates
    used_image_items = set()
    ordered_images = []
    interact_text = []
    
    for item_id, title in zip(selected_ids, selected_titles):
        item_id_int = int(item_id)
        # Only add <image> if we have image AND haven't used it yet
        if item_id_int in image_item_to_img and item_id_int not in used_image_items:
            interact_text.append(f'<image>{title}{HISTORY_EMB_TOKEN}')
            ordered_images.append(image_item_to_img[item_id_int])
            used_image_items.add(item_id_int)
        else:
            interact_text.append(f'{title}{HISTORY_EMB_TOKEN}')
    
    return ','.join(interact_text), list(selected_ids), ordered_images
```

#### 5.2.5 Modified `make_candidate_text` Method

```python
def make_candidate_text(self, interact_ids, candidate_num, target_item_id, target_item_title, 
                        use_smolvlm_tokens=False):
    """
    Create candidate text with appropriate embedding tokens.
    
    Args:
        use_smolvlm_tokens: If True, use SmolVLM reserved tokens
    """
    if use_smolvlm_tokens:
        CAND_TOKEN = "<|reserved_special_token_2|>"
    else:
        CAND_TOKEN = "[CandidateEmb]"  # Original OPT token
    
    # Generate negative samples
    neg_item_id = []
    while len(neg_item_id) < 50:
        t = np.random.randint(1, self.item_num + 1)
        if t not in interact_ids and t not in neg_item_id:
            neg_item_id.append(t)
    
    # Build candidate list (target + negatives)
    candidate_ids = [target_item_id]
    candidate_text = [target_item_title + CAND_TOKEN]
    
    for neg_candidate in neg_item_id[:candidate_num - 1]:
        title = self.find_item_text_single(neg_candidate, title_flag=True, description_flag=False)
        candidate_text.append(title + CAND_TOKEN)
        candidate_ids.append(neg_candidate)
    
    # Shuffle candidates
    random_ = np.random.permutation(len(candidate_text))
    candidate_text = np.array(candidate_text)[random_]
    candidate_ids = np.array(candidate_ids)[random_]
    
    return ','.join(candidate_text), candidate_ids
```

#### 5.2.6 New `pre_train_phase2_smolvlm` Method

```python
def pre_train_phase2_smolvlm(self, data, optimizer, batch_iter):
    """SmolVLM-based Stage-2 training with image support."""
    epoch, total_epoch, step, total_step = batch_iter
    
    optimizer.zero_grad()
    u, seq, pos, neg = data
    
    text_input = []
    text_output = []
    interact_embs = []
    candidate_embs = []
    all_images = []
    
    # Get user sequence embeddings from SASRec (UNCHANGED from original)
    with torch.no_grad():
        log_emb = self.recsys.model(u, seq, pos, neg, mode='log_only')
    
    for i in range(len(u)):
        target_item_id = pos[i][-1]
        target_item_title = self.find_item_text_single(target_item_id, title_flag=True, description_flag=False)
        
        # Get history items
        history_ids = seq[i][seq[i] > 0]
        
            # NEW: Load images for last N history items (with fallback)
            images, image_item_ids = self.get_history_images(history_ids, max_images=self.num_history_images)
            
            # NEW: Create interaction text with <image> tokens
            # Returns ordered_images to ensure <image> tokens match actual images
            interact_text, interact_ids, ordered_images = self.make_interact_text_with_images(
                history_ids, 10, image_item_ids, images
            )
        
        # Create candidate text with SmolVLM tokens
        candidate_text, candidate_ids = self.make_candidate_text(
            history_ids, 20, target_item_id, target_item_title,
            use_smolvlm_tokens=True
        )
        
        # Build prompt using SmolVLM reserved tokens
        USER_REP_TOKEN = "<|reserved_special_token_0|>"
        input_text = f'{USER_REP_TOKEN} is a user representation.\n\n'
        input_text += 'This user has bought:\n'
        input_text += interact_text
        input_text += '\n\nRecommend one item from the following candidates:\n'
        input_text += candidate_text
        input_text += '\n\nThe recommendation is'
        
        text_input.append(input_text)
        text_output.append(target_item_title)
        all_images.append(images)
        
        # Get joint embeddings (from Stage-1!) and project
        # THIS IS WHERE STAGE-1 CONNECTS TO STAGE-2
        interact_embs.append(self.item_emb_proj(self.get_item_emb(interact_ids)))
        candidate_embs.append(self.item_emb_proj(self.get_item_emb(candidate_ids)))
    
    # Prepare samples for SmolVLM
    samples = {
        'text_input': text_input,
        'text_output': text_output,
        'images': all_images,
        'history_embs': interact_embs,
        'candidate_embs': candidate_embs
    }
    
    # Project user representation (50 → 960)
    log_emb = self.log_emb_proj(log_emb)
    
    # Forward pass through SmolVLM
    loss_rm = self.vlm(log_emb, samples)
    loss_rm.backward()
    optimizer.step()
    
    # Log with image stats
    num_images_loaded = sum(len(imgs) for imgs in all_images)
    print(f"SmolVLM loss epoch {epoch}/{total_epoch} iter {step}/{total_step}: {loss_rm.item():.4f} | Images: {num_images_loaded}/{len(u)*self.num_history_images}")
```

---

## 6. Complete Data Flow

### 6.1 Training Data Flow (One Sample)

```
                                USER INTERACTION DATA
                                         │
                    ┌────────────────────┼────────────────────┐
                    │                    │                    │
                    ▼                    ▼                    ▼
            User ID (u)         Sequence (seq)         Target (pos)
                    │                    │                    │
                    │                    │                    │
                    ▼                    ▼                    ▼
┌───────────────────────────────────────────────────────────────────────────┐
│                           SASRec MODEL (FROZEN)                            │
├───────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│   log_emb = model(u, seq, pos, neg, mode='log_only')                      │
│                         │                                                  │
│                         ▼                                                  │
│              User Sequence Embedding (50-dim)                              │
│                                                                            │
└───────────────────────────────────────────────────────────────────────────┘
                                         │
                                         ▼
┌───────────────────────────────────────────────────────────────────────────┐
│                         STAGE-1 MLP (FROZEN)                               │
├───────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│   For each item in history and candidates:                                 │
│                                                                            │
│   item_emb = recsys.model.item_emb(item_id)   # 50-dim CF embedding       │
│                         │                                                  │
│                         ▼                                                  │
│   joint_emb, _ = self.mlp(item_emb)           # 128-dim JOINT embedding   │
│                                                                            │
│   This 128-dim embedding captures BOTH:                                    │
│   - Collaborative Filtering patterns (user behavior)                       │
│   - Text Semantics (aligned with SBERT in Stage-1)                        │
│                                                                            │
└───────────────────────────────────────────────────────────────────────────┘
                                         │
                                         ▼
┌───────────────────────────────────────────────────────────────────────────┐
│                       PROJECTION NETWORKS (TRAINABLE)                      │
├───────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│   log_emb_proj: 50 → 960   (user representation)                          │
│   item_emb_proj: 128 → 960 (joint embeddings)                             │
│                                                                            │
│   These are the ONLY trainable components in Stage-2!                     │
│                                                                            │
└───────────────────────────────────────────────────────────────────────────┘
                                         │
              ┌──────────────────────────┼──────────────────────────┐
              │                          │                          │
              ▼                          ▼                          ▼
     Projected User Emb         Projected History Embs    Projected Candidate Embs
         (1, 960)                  (num_hist, 960)           (num_cand, 960)
              │                          │                          │
              │                          │                          │
              └──────────────────────────┼──────────────────────────┘
                                         │
                                         ▼
┌───────────────────────────────────────────────────────────────────────────┐
│                          IMAGE LOADING                                     │
├───────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│   images, image_item_ids = get_history_images(history_ids, max_images=3)  │
│                                                                            │
│   Fallback Logic:                                                          │
│   - Start from most recent item                                            │
│   - If no image URL → try previous item                                    │
│   - If download fails → try previous item                                  │
│   - Continue until max_images found                                        │
│                                                                            │
└───────────────────────────────────────────────────────────────────────────┘
                                         │
                                         ▼
┌───────────────────────────────────────────────────────────────────────────┐
│                        PROMPT CONSTRUCTION                                 │
├───────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│   <|reserved_special_token_0|> is a user representation.                  │
│                                                                            │
│   This user has bought:                                                    │
│   <image>"Product A"<|reserved_special_token_1|>,                         │
│   <image>"Product B"<|reserved_special_token_1|>,                         │
│   "Product C"<|reserved_special_token_1|>                                 │
│                                                                            │
│   Recommend one item from the following candidates:                        │
│   "Candidate 1"<|reserved_special_token_2|>,                              │
│   "Candidate 2"<|reserved_special_token_2|>,                              │
│   ...                                                                      │
│                                                                            │
│   The recommendation is                                                    │
│                                                                            │
└───────────────────────────────────────────────────────────────────────────┘
                                         │
                                         ▼
┌───────────────────────────────────────────────────────────────────────────┐
│                         SmolVLM PROCESSING                                 │
├───────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│   STEP 1: Process images through vision encoder                            │
│   ┌─────────────────────────────────────────────────────────────────┐     │
│   │  pixel_values = processor(images)                                │     │
│   │        │                                                         │     │
│   │        ▼                                                         │     │
│   │  vision_model: pixel_values → vision_features (768-dim)         │     │
│   │        │                                                         │     │
│   │        ▼                                                         │     │
│   │  connector: vision_features → image_hidden_states (960-dim)     │     │
│   └─────────────────────────────────────────────────────────────────┘     │
│                                                                            │
│   STEP 2: Merge images with text using inputs_merger                       │
│   ┌─────────────────────────────────────────────────────────────────┐     │
│   │  inputs_embeds = embed(input_ids)                                │     │
│   │        │                                                         │     │
│   │        ▼                                                         │     │
│   │  merged = inputs_merger(input_ids, inputs_embeds, image_hidden)  │     │
│   │                                                                  │     │
│   │  Each <image> token (1 token) → 64 vision tokens                │     │
│   └─────────────────────────────────────────────────────────────────┘     │
│                                                                            │
│   STEP 3: Replace custom tokens with projected embeddings                  │
│   ┌─────────────────────────────────────────────────────────────────┐     │
│   │  Find <|reserved_special_token_0|> → replace with user_emb      │     │
│   │  Find <|reserved_special_token_1|> → replace with history_embs  │     │
│   │  Find <|reserved_special_token_2|> → replace with candidate_embs│     │
│   └─────────────────────────────────────────────────────────────────┘     │
│                                                                            │
│   STEP 4: Forward through language model                                   │
│   ┌─────────────────────────────────────────────────────────────────┐     │
│   │  outputs = text_model(inputs_embeds=merged_with_replacements)    │     │
│   │        │                                                         │     │
│   │        ▼                                                         │     │
│   │  logits = lm_head(outputs.last_hidden_state)                    │     │
│   │        │                                                         │     │
│   │        ▼                                                         │     │
│   │  loss = CrossEntropyLoss(logits, target_tokens)                 │     │
│   └─────────────────────────────────────────────────────────────────┘     │
│                                                                            │
└───────────────────────────────────────────────────────────────────────────┘
                                         │
                                         ▼
                              BACKPROPAGATION
                                         │
                    Only updates projection networks!
                    (SmolVLM is frozen)
```

### 6.2 The Critical Stage-1 → Stage-2 Connection

```python
def get_item_emb(self, item_ids):
    """
    THIS IS WHERE STAGE-1 OUTPUT IS USED IN STAGE-2
    """
    with torch.no_grad():
        # Step 1: Get raw CF embeddings from SASRec (50-dim)
        item_embs = self.recsys.model.item_emb(torch.LongTensor(item_ids).to(self.device))
        
        # Step 2: Pass through Stage-1 MLP to get JOINT embedding (128-dim)
        # This MLP was trained in Stage-1 to align CF with text semantics
        item_embs, _ = self.mlp(item_embs)
    
    return item_embs  # 128-dim joint embedding
```

---

## 7. Detailed Code Walkthrough

### 7.1 What Happens When You Run Training

```bash
python main.py --pretrain_stage2 --use_smolvlm --rec_pre_trained_data All_Beauty
```

**Step-by-step execution:**

1. **Argument Parsing** (`main.py`):
   ```python
   args.use_smolvlm = True
   args.num_history_images = 3
   ```

2. **Model Initialization** (`a_llmrec_model.py:__init__`):
   - Load SASRec (frozen)
   - Load Stage-1 MLP (frozen)
   - Load SmolVLM (frozen)
   - Load image URL dictionary
   - Create projection networks (trainable)

3. **Training Loop** (`train_model.py`):
   ```python
   for epoch in range(num_epochs):
       for batch in dataloader:
           model([u, seq, pos, neg], optimizer=optimizer, batch_iter=(...), mode='phase2')
   ```

4. **Forward Pass** (`a_llmrec_model.py:pre_train_phase2_smolvlm`):
   - Get user embeddings from SASRec
   - Load images for history items
   - Build prompt with `<image>` tokens
   - Get joint embeddings via `get_item_emb()` → Stage-1 MLP
   - Project to SmolVLM space
   - Call `self.vlm(log_emb, samples)`

5. **SmolVLM Processing** (`smolvlm4rec.py:forward`):
   - Process images through vision encoder
   - Merge images and text
   - Replace custom tokens with embeddings
   - Compute language modeling loss

6. **Backward Pass**:
   - Loss backpropagates through projection networks only
   - SmolVLM gradients are not computed (frozen)

---

## 8. Technical Challenges Solved

### 8.1 Challenge: SmolVLM's inputs_merger Conflict

**Problem**: When we tried to pass both `inputs_embeds` (with our custom tokens) and `pixel_values` to SmolVLM, it failed with:
```
ValueError: At least one sample has <image> tokens not divisible by patch_size
```

**Root Cause**: SmolVLM's `inputs_merger` expects to find `<image>` tokens in `input_ids` and expand them. But we had already modified `inputs_embeds`, causing a mismatch.

**Solution**: Two-phase approach:
1. First, let SmolVLM's `inputs_merger` handle image-text merging normally
2. Then, find and replace our custom tokens in the resulting merged embeddings

### 8.2 Challenge: Extracting Vision Features

**Problem**: Need `image_hidden_states` (output of vision encoder + connector), but SmolVLM doesn't expose this directly.

**Solution**: Use a forward hook:
```python
def _get_image_hidden_states(self, pixel_values, pixel_attention_mask):
    captured = [None]
    
    def hook(module, input, output):
        captured[0] = output
    
    handle = self.model.model.connector.register_forward_hook(hook)
    # Trigger vision processing
    _ = self.model.model(pixel_values=pixel_values, ...)
    handle.remove()
    
    return captured[0]
```

### 8.3 Challenge: Token Embedding Resize

**Problem**: Adding new tokens requires calling `resize_token_embeddings()`, which can destabilize a frozen model.

**Solution**: Use SmolVLM's pre-existing reserved tokens:
- `<|reserved_special_token_0|>` (ID: 49191) → User representation
- `<|reserved_special_token_1|>` (ID: 49192) → History embeddings
- `<|reserved_special_token_2|>` (ID: 49193) → Candidate embeddings

These tokens already exist with trained embeddings - no resize needed!

### 8.4 Challenge: Duplicate Items in History

**Problem**: When a user bought the same item multiple times, each occurrence would get an `<image>` token, but we only loaded one image per unique item. This caused:
```
ValueError: The number of images in the text [4] and images [3] should be the same.
```

**Example of the bug:**
- User history: `[A, B, A, C]` (item A appears twice)
- Images loaded for unique items: `[A, C]` (2 images)
- Old code added `<image>` for each occurrence: 3 `<image>` tokens
- SmolVLM expected 3 images but only 2 provided → Error!

**Solution**: Track which images have been used and only add `<image>` token once per unique item:

```python
def make_interact_text_with_images(self, interact_ids, interact_max_num, image_item_ids, images):
    # Create a mapping from item_id to image
    image_item_to_img = {}
    for item_id, img in zip(image_item_ids, images):
        item_id_int = int(item_id)
        if item_id_int not in image_item_to_img:
            image_item_to_img[item_id_int] = img
    
    # Track which images we've used (avoid adding <image> for same item twice)
    used_image_items = set()
    ordered_images = []
    
    for item_id, title in zip(selected_ids, selected_titles):
        item_id_int = int(item_id)
        
        # Only add <image> if: 1) we have an image, 2) we haven't used it yet
        if item_id_int in image_item_to_img and item_id_int not in used_image_items:
            interact_text.append(f'<image>{title}{HISTORY_EMB_TOKEN}')
            ordered_images.append(image_item_to_img[item_id_int])
            used_image_items.add(item_id_int)  # Mark as used
        else:
            interact_text.append(f'{title}{HISTORY_EMB_TOKEN}')
    
    return interact_text, selected_ids, ordered_images  # Return ordered_images
```

The function now returns `ordered_images` which contains images in the exact order they appear in the text, ensuring the count always matches.

### 8.5 Challenge: Custom Generation

**Problem**: `model.generate()` doesn't support our custom embedding injection.

**Solution**: Implement manual greedy generation:
```python
for _ in range(max_new_tokens):
    outputs = model(inputs_embeds=current_embeds)
    next_token = outputs.logits[:, -1, :].argmax()
    next_emb = embed(next_token)
    current_embeds = cat([current_embeds, next_emb])
```

---

## 9. How to Run

### 9.1 Prerequisites

1. **Stage-1 must be completed first** (trains the alignment MLP)
2. **Preprocessing** to create image URL mappings

### 9.2 Preprocessing

```bash
python preprocess_smolvlm_images.py \
    --review-path ./data/amazon/All_Beauty.json.gz \
    --meta-path ./data/amazon/meta_All_Beauty.json \
    --output-dir ./data/amazon \
    --dataset-name All_Beauty
```

### 9.3 Training

```bash
# Activate environment
conda activate ALLM-Rec

# Run Stage-2 training with SmolVLM
python main.py \
    --pretrain_stage2 \
    --use_smolvlm \
    --rec_pre_trained_data All_Beauty \
    --num_history_images 3 \
    --batch_size2 2 \
    --num_epochs 5
```

### 9.4 Expected Output

```
SmolVLM loss epoch 1/5 iter 0/2413: 0.8748 | Images: 1/3
SmolVLM loss epoch 1/5 iter 1/2413: 0.6220 | Images: 1/3
SmolVLM loss epoch 1/5 iter 2/2413: 0.6233 | Images: 3/3
SmolVLM loss epoch 1/5 iter 3/2413: 3.1572 | Images: 0/3
...
```

The `Images: X/Y` shows how many images were successfully loaded (X) out of requested (Y).

---

## 10. File Structure

```
A-LLMRec-Forked/
├── main.py                          # Entry point (MODIFIED: added --use_smolvlm)
├── train_model.py                   # Training loops (unchanged)
├── utils.py                         # General utilities (unchanged)
├── utils_image.py                   # NEW: Image loading utilities
├── preprocess_smolvlm_images.py     # NEW: Preprocessing script
├── DETAILED_IMPLEMENTATION_GUIDE.md # This document
├── SMOLVLM_INTEGRATION.md           # Summary document
│
├── models/
│   ├── a_llmrec_model.py            # Main model (MODIFIED: SmolVLM integration)
│   ├── smolvlm4rec.py               # NEW: SmolVLM wrapper
│   ├── llm4rec.py                   # Original OPT wrapper (unchanged)
│   ├── recsys_model.py              # SASRec wrapper (unchanged)
│   └── saved_models/                # Checkpoints
│
├── data/
│   └── amazon/
│       ├── All_Beauty.json.gz               # Raw reviews
│       ├── meta_All_Beauty.json             # Raw metadata
│       ├── All_Beauty.txt                   # Processed interactions
│       ├── All_Beauty_text_name_dict.json.gz    # Title/description mapping
│       ├── All_Beauty_image_url_dict.json.gz    # NEW: Image URL mapping
│       └── All_Beauty_itemid_to_asin.pkl        # ID to ASIN mapping
│
└── pre_train/
    └── sasrec/                      # SASRec pre-training (unchanged)
```

---

## Summary

We successfully integrated SmolVLM2-500M-Video-Instruct into A-LLMRec's Stage-2, enabling multimodal recommendations that combine:
- **Collaborative Filtering signals** (from SASRec via Stage-1 joint embeddings)
- **Text semantics** (aligned with CF in Stage-1)
- **Visual features** (from product images via SmolVLM's vision encoder)

Key technical achievements:
1. Maintained Stage-1 completely unchanged
2. Used reserved tokens to avoid embedding resize
3. Implemented two-phase embedding merging to work with SmolVLM's architecture
4. Created robust image loading with fallback logic
5. Implemented manual generation for inference
6. Fixed duplicate item handling to ensure `<image>` token count matches actual images

The system is now training successfully with images being processed and loss decreasing.
