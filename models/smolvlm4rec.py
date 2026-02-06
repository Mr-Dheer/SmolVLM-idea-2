"""
SmolVLM wrapper for A-LLMRec multimodal recommendations.

This module provides the SmolVLM4Rec class that:
1. Loads SmolVLM2-500M-Video-Instruct model (frozen)
2. Applies chat template for proper instruction-tuned behavior
3. Handles interleaved image-text inputs
4. Replaces custom tokens with projected embeddings using analytical position tracking
5. Provides training (forward) and inference (generate) methods

Key design decisions:
- Uses SmolVLM's reserved tokens to avoid embedding resize
- Two-phase embedding merging: (1) SmolVLM merges images, (2) custom tokens replaced
- Analytical position tracking after image expansion (no torch.allclose)
- Chat template wrapping for proper instruction following
"""

import torch
import torch.nn as nn
import logging
from typing import List, Optional, Dict, Any

from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image

logger = logging.getLogger(__name__)


class SmolVLM4Rec(nn.Module):
    """
    SmolVLM wrapper for recommendation tasks.

    Handles model loading, chat template formatting, image-text interleaving,
    custom embedding token replacement, training, and inference.
    """

    # Model identifier
    MODEL_NAME = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"

    # Reserved tokens for embedding injection (already in vocabulary, no resize needed)
    USER_REP_TOKEN = "<|reserved_special_token_0|>"
    HISTORY_EMB_TOKEN = "<|reserved_special_token_1|>"
    CANDIDATE_EMB_TOKEN = "<|reserved_special_token_2|>"

    # System prompt following SmolVLM paper Finding 6
    SYSTEM_PROMPT = (
        "You are a recommendation assistant. Based on the user's purchase history "
        "and visual preferences, recommend the most suitable product from the candidates."
    )

    def __init__(
        self,
        device: str,
        model_name: str = None,
        max_output_len: int = 256,
    ):
        super().__init__()
        self.device = device
        self.model_name = model_name or self.MODEL_NAME
        self.max_output_len = max_output_len

        logger.info(f"Loading SmolVLM from: {self.model_name}")
        logger.info(f"Device: {self.device}")

        # Load processor and model
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            device_map=self.device,
        )

        # Freeze all model parameters
        for param in self.model.parameters():
            param.requires_grad = False

        # Setup custom tokens with runtime verification
        self._setup_custom_tokens()

        # Cache image token ID
        self.image_token_id = self.processor.tokenizer.convert_tokens_to_ids("<image>")

        # Get model hidden size
        self.hidden_size = self.model.config.text_config.hidden_size

        logger.info(f"Model hidden size: {self.hidden_size}")
        logger.info(f"Image token ID: {self.image_token_id}")
        logger.info("SmolVLM loaded successfully")

    # -------------------------------------------------------------------------
    # Setup
    # -------------------------------------------------------------------------

    def _setup_custom_tokens(self):
        """Resolve custom token IDs at runtime with bidirectional verification.

        This avoids hardcoded IDs that silently break if the model/tokenizer changes.
        """
        tokenizer = self.processor.tokenizer

        self.user_rep_token_id = tokenizer.convert_tokens_to_ids(self.USER_REP_TOKEN)
        self.history_emb_token_id = tokenizer.convert_tokens_to_ids(self.HISTORY_EMB_TOKEN)
        self.candidate_emb_token_id = tokenizer.convert_tokens_to_ids(self.CANDIDATE_EMB_TOKEN)

        # Bidirectional verification: token_str -> id -> token_str must round-trip
        for name, token_str, token_id in [
            ("UserRep", self.USER_REP_TOKEN, self.user_rep_token_id),
            ("HistoryEmb", self.HISTORY_EMB_TOKEN, self.history_emb_token_id),
            ("CandidateEmb", self.CANDIDATE_EMB_TOKEN, self.candidate_emb_token_id),
        ]:
            assert token_id != tokenizer.unk_token_id, (
                f"{name} token '{token_str}' resolved to UNK — not in vocabulary! "
                f"Check that the model has reserved special tokens."
            )
            roundtrip = tokenizer.convert_ids_to_tokens(token_id)
            assert roundtrip == token_str, (
                f"{name} round-trip failed: '{token_str}' -> ID {token_id} -> '{roundtrip}'"
            )
            logger.info(f"  {name}: '{token_str}' -> ID {token_id} (verified)")

    # -------------------------------------------------------------------------
    # Chat template
    # -------------------------------------------------------------------------

    def _wrap_in_chat_template(self, raw_text: str) -> str:
        """Convert raw prompt text (with <image> markers) to chat template format.

        Splits the raw text by <image> markers, creates structured content blocks,
        and applies the processor's chat template. This ensures SmolVLM receives
        inputs in the format it was instruction-tuned on (Finding 6 from SmolVLM paper).

        Args:
            raw_text: Raw prompt text possibly containing <image> markers

        Returns:
            Formatted text string with chat template applied.
            Falls back to raw_text if chat template is unavailable.
        """
        # Split by <image> to create interleaved content blocks
        parts = raw_text.split("<image>")
        content = []

        # First part: text before any image
        if parts[0]:
            content.append({"type": "text", "text": parts[0]})

        # Remaining parts: each starts after an <image>
        for part in parts[1:]:
            content.append({"type": "image"})
            if part:
                content.append({"type": "text", "text": part})

        messages = [
            {"role": "user", "content": content},
        ]

        try:
            # Prefer processor.apply_chat_template (handles multimodal content)
            if hasattr(self.processor, "apply_chat_template"):
                formatted = self.processor.apply_chat_template(
                    messages, add_generation_prompt=True
                )
            elif hasattr(self.processor.tokenizer, "apply_chat_template"):
                formatted = self.processor.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            else:
                logger.warning("No apply_chat_template found. Using raw text.")
                return raw_text

            # Handle case where template returns token IDs instead of string
            if isinstance(formatted, list):
                formatted = self.processor.tokenizer.decode(
                    formatted, skip_special_tokens=False
                )

            return formatted

        except Exception as e:
            logger.warning(f"Chat template failed ({e}). Falling back to raw text.")
            return raw_text

    # -------------------------------------------------------------------------
    # Vision processing
    # -------------------------------------------------------------------------

    def _get_image_hidden_states(
        self,
        pixel_values: torch.Tensor,
        pixel_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Process images through vision encoder and connector.

        Uses a forward hook on the connector to capture image_hidden_states
        without running the full LM forward pass unnecessarily.

        NOTE: This depends on SmolVLM's internal module naming (model.model.connector).
        If HuggingFace refactors internals, this will need updating. A future
        optimization could call vision_model + connector directly.

        Args:
            pixel_values: Image tensor from processor
            pixel_attention_mask: Attention mask for images

        Returns:
            image_hidden_states from the connector
        """
        captured = [None]

        def hook(module, input, output):
            captured[0] = output

        handle = self.model.model.connector.register_forward_hook(hook)
        try:
            dummy_ids = torch.tensor([[1]], device=self.device)
            with torch.no_grad():
                self.model.model(
                    input_ids=dummy_ids,
                    pixel_values=pixel_values,
                    pixel_attention_mask=pixel_attention_mask,
                    return_dict=True,
                )
        finally:
            handle.remove()

        return captured[0]

    # -------------------------------------------------------------------------
    # Position tracking and embedding replacement
    # -------------------------------------------------------------------------

    def _compute_post_merge_positions(
        self, input_ids: torch.Tensor, post_merge_len: int
    ) -> Dict[str, List[int]]:
        """Compute where custom tokens end up after image token expansion.

        Each <image> token (1 token) expands to N vision tokens during merging.
        This method analytically computes the post-merge position of each custom
        token based on how many <image> tokens precede it.

        Args:
            input_ids: Pre-merge token IDs [1, seq_len]
            post_merge_len: Length of sequence after image merging

        Returns:
            Dict mapping token type ('user_rep', 'history_emb', 'candidate_emb')
            to list of post-merge positions.
        """
        ids = input_ids[0]
        pre_merge_len = ids.size(0)

        # Find image token positions
        image_positions = (ids == self.image_token_id).nonzero(as_tuple=True)[0].tolist()
        num_images = len(image_positions)

        # Compute tokens-per-image from the expansion
        if num_images > 0:
            # Each <image> (1 token) becomes tokens_per_image tokens
            # total_expansion = post_merge_len - pre_merge_len
            # total_expansion = num_images * (tokens_per_image - 1)
            tokens_per_image = (post_merge_len - pre_merge_len + num_images) // num_images
        else:
            tokens_per_image = 0

        position_map = {
            "user_rep": [],
            "history_emb": [],
            "candidate_emb": [],
        }

        images_seen = 0
        for pos in range(pre_merge_len):
            token_id = ids[pos].item()

            if token_id == self.image_token_id:
                images_seen += 1
                continue

            # New position = original position + cumulative expansion
            new_pos = pos + images_seen * (tokens_per_image - 1)

            if token_id == self.user_rep_token_id:
                position_map["user_rep"].append(new_pos)
            elif token_id == self.history_emb_token_id:
                position_map["history_emb"].append(new_pos)
            elif token_id == self.candidate_emb_token_id:
                position_map["candidate_emb"].append(new_pos)

        return position_map

    def _merge_images_and_replace_tokens(
        self,
        input_ids: torch.Tensor,
        image_hidden_states: torch.Tensor,
        user_emb: torch.Tensor,
        history_embs: torch.Tensor,
        candidate_embs: torch.Tensor,
    ) -> tuple:
        """Merge image embeddings and replace custom tokens with projected embeddings.

        Two-step process:
        1. Use SmolVLM's inputs_merger for image-text merging (expands <image> tokens)
        2. Replace custom tokens at analytically computed positions

        Args:
            input_ids: Token IDs [1, seq_len]
            image_hidden_states: Vision encoder output
            user_emb: User representation [1, hidden_size]
            history_embs: History embeddings [1, num_history, hidden_size]
            candidate_embs: Candidate embeddings [1, num_candidates, hidden_size]

        Returns:
            Tuple of (merged_embeds, attention_mask)
        """
        # Get text embeddings
        inputs_embeds = self.model.get_input_embeddings()(input_ids)

        # Step 1: Merge images using SmolVLM's native inputs_merger
        merged_embeds = self.model.model.inputs_merger(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            image_hidden_states=image_hidden_states,
        )

        # Step 2: Compute post-merge positions analytically
        position_map = self._compute_post_merge_positions(
            input_ids, merged_embeds.size(1)
        )

        # Clone to avoid in-place modification issues under autocast
        merged_embeds = merged_embeds.clone()

        # Replace UserRep token
        for i, pos in enumerate(position_map["user_rep"]):
            if pos < merged_embeds.size(1):
                merged_embeds[0, pos] = user_emb[0]

        # Replace HistoryEmb tokens
        for i, pos in enumerate(position_map["history_emb"]):
            if i < history_embs.size(1) and pos < merged_embeds.size(1):
                merged_embeds[0, pos] = history_embs[0, i]

        # Replace CandidateEmb tokens
        for i, pos in enumerate(position_map["candidate_emb"]):
            if i < candidate_embs.size(1) and pos < merged_embeds.size(1):
                merged_embeds[0, pos] = candidate_embs[0, i]

        attention_mask = torch.ones(
            (1, merged_embeds.size(1)), device=self.device, dtype=torch.long
        )

        return merged_embeds, attention_mask

    def _replace_custom_tokens(
        self,
        input_ids: torch.Tensor,
        inputs_embeds: torch.Tensor,
        user_emb: torch.Tensor,
        history_embs: torch.Tensor,
        candidate_embs: torch.Tensor,
    ) -> torch.Tensor:
        """Replace custom token embeddings (text-only path, no image expansion).

        Used when there are no images, so no position offset computation needed.

        Args:
            input_ids: Token IDs [batch, seq_len]
            inputs_embeds: Token embeddings [batch, seq_len, hidden_size]
            user_emb: User representation [batch, hidden_size]
            history_embs: History embeddings [batch, num_history, hidden_size]
            candidate_embs: Candidate embeddings [batch, num_candidates, hidden_size]

        Returns:
            Modified inputs_embeds with custom tokens replaced
        """
        # Clone to avoid in-place modification issues
        inputs_embeds = inputs_embeds.clone()
        batch_size = input_ids.size(0)

        for b in range(batch_size):
            # UserRep
            positions = (input_ids[b] == self.user_rep_token_id).nonzero(as_tuple=True)[0]
            if len(positions) > 0:
                inputs_embeds[b, positions[0]] = user_emb[b]

            # HistoryEmb
            positions = (input_ids[b] == self.history_emb_token_id).nonzero(as_tuple=True)[0]
            for i, pos in enumerate(positions):
                if i < history_embs.size(1):
                    inputs_embeds[b, pos] = history_embs[b, i]

            # CandidateEmb
            positions = (input_ids[b] == self.candidate_emb_token_id).nonzero(as_tuple=True)[0]
            for i, pos in enumerate(positions):
                if i < candidate_embs.size(1):
                    inputs_embeds[b, pos] = candidate_embs[b, i]

        return inputs_embeds

    # -------------------------------------------------------------------------
    # Training
    # -------------------------------------------------------------------------

    def forward(
        self,
        user_emb: torch.Tensor,
        samples: Dict[str, Any],
    ) -> torch.Tensor:
        """Training forward pass with language modeling loss.

        Supports both text-only and multimodal (with images) training.
        Applies chat template and uses correct label computation.

        Args:
            user_emb: Projected user representations [batch, hidden_size]
            samples: Dictionary containing:
                - text_input: List of raw prompt texts (may contain <image> tokens)
                - text_output: List of target item titles
                - images: List of List of PIL Images (per sample), optional
                - history_embs: List of history embedding tensors
                - candidate_embs: List of candidate embedding tensors

        Returns:
            Language modeling loss (scalar tensor)
        """
        batch_size = len(samples["text_input"])
        all_losses = []

        for i in range(batch_size):
            prompt_text = samples["text_input"][i]
            target_text = samples["text_output"][i]
            images = samples.get("images", [[]])[i] if "images" in samples else []
            history_embs = samples["history_embs"][i].unsqueeze(0)
            candidate_embs = samples["candidate_embs"][i].unsqueeze(0)

            eos = self.processor.tokenizer.eos_token

            # Apply chat template for proper instruction following
            formatted_prompt = self._wrap_in_chat_template(prompt_text)

            # Full training text: prompt + space + target + EOS
            full_text = formatted_prompt + " " + target_text + eos

            # Tokenize prompt alone to find the prompt/target boundary
            prompt_encoding = self.processor.tokenizer(
                formatted_prompt, return_tensors="pt"
            ).to(self.device)
            prompt_token_len = prompt_encoding["input_ids"].size(1)

            if images and len(images) > 0:
                # === MULTIMODAL PATH ===

                # Process full text + images through processor
                inputs = self.processor(
                    text=full_text,
                    images=images,
                    return_tensors="pt",
                    padding=True,
                ).to(self.device)

                if "pixel_values" in inputs:
                    inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)

                input_ids = inputs["input_ids"]

                # Extract image features via vision encoder + connector
                image_hidden_states = self._get_image_hidden_states(
                    inputs["pixel_values"], inputs["pixel_attention_mask"]
                )

                # Merge images and replace custom tokens (with analytical positions)
                inputs_embeds, attention_mask = self._merge_images_and_replace_tokens(
                    input_ids=input_ids,
                    image_hidden_states=image_hidden_states,
                    user_emb=user_emb[i : i + 1],
                    history_embs=history_embs,
                    candidate_embs=candidate_embs,
                )

                # --- Label computation (correct for image expansion) ---
                # All <image> tokens are in the prompt, not the target.
                # After merging, the prompt expands but the target tokens
                # remain at the END of the sequence, unchanged.
                pre_merge_len = input_ids.size(1)
                post_merge_len = inputs_embeds.size(1)

                # Target tokens: everything after the prompt in the pre-merge sequence
                target_ids = input_ids[0, prompt_token_len:]
                target_len = target_ids.size(0)

                # Labels: -100 for expanded prompt, target IDs at the end
                labels = torch.full(
                    (1, post_merge_len), -100, dtype=torch.long, device=self.device
                )
                if target_len > 0:
                    labels[0, -target_len:] = target_ids

            else:
                # === TEXT-ONLY PATH ===
                full_text_clean = full_text.replace("<image>", "")

                inputs = self.processor.tokenizer(
                    full_text_clean, return_tensors="pt", padding=True
                ).to(self.device)

                input_ids = inputs["input_ids"]
                attention_mask = inputs["attention_mask"]

                # Get embeddings and replace custom tokens
                inputs_embeds = self.model.get_input_embeddings()(input_ids)
                inputs_embeds = self._replace_custom_tokens(
                    input_ids=input_ids,
                    inputs_embeds=inputs_embeds,
                    user_emb=user_emb[i : i + 1],
                    history_embs=history_embs,
                    candidate_embs=candidate_embs,
                )

                # Labels: mask prompt tokens, keep target tokens
                # Use prompt length from chat-template-formatted prompt (cleaned)
                formatted_prompt_clean = formatted_prompt.replace("<image>", "")
                prompt_enc = self.processor.tokenizer(
                    formatted_prompt_clean, return_tensors="pt"
                ).to(self.device)
                prompt_len_clean = prompt_enc["input_ids"].size(1)

                labels = input_ids.clone()
                labels[:, :prompt_len_clean] = -100

            # Forward pass through language model
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                outputs = self.model.model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    return_dict=True,
                )

                # Compute LM loss
                logits = self.model.lm_head(outputs.last_hidden_state)

                # Shift for next-token prediction
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()

                loss = nn.CrossEntropyLoss()(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                )

            all_losses.append(loss)

        # Average loss across batch
        return torch.stack(all_losses).mean()

    # -------------------------------------------------------------------------
    # Inference
    # -------------------------------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        user_emb: torch.Tensor,
        prompt_text: str,
        images: List[Image.Image],
        history_embs: torch.Tensor,
        candidate_embs: torch.Tensor,
        max_new_tokens: int = 64,
    ) -> str:
        """Generate recommendation for a single sample.

        Uses manual greedy decoding since model.generate() doesn't support
        our custom embedding injection approach.

        Args:
            user_emb: Projected user representation [1, hidden_size]
            prompt_text: Raw prompt text (may contain <image> tokens)
            images: List of PIL Images for history items
            history_embs: Projected history embeddings [1, num_history, hidden_size]
            candidate_embs: Projected candidate embeddings [1, num_candidates, hidden_size]
            max_new_tokens: Maximum tokens to generate

        Returns:
            Generated text (recommended item title)
        """
        # Apply chat template
        formatted_prompt = self._wrap_in_chat_template(prompt_text)

        if images and len(images) > 0:
            # MULTIMODAL: process images + text
            inputs = self.processor(
                text=formatted_prompt,
                images=images,
                return_tensors="pt",
                padding=True,
            ).to(self.device)

            if "pixel_values" in inputs:
                inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)

            input_ids = inputs["input_ids"]

            image_hidden_states = self._get_image_hidden_states(
                inputs["pixel_values"], inputs["pixel_attention_mask"]
            )

            inputs_embeds, attention_mask = self._merge_images_and_replace_tokens(
                input_ids=input_ids,
                image_hidden_states=image_hidden_states,
                user_emb=user_emb,
                history_embs=history_embs,
                candidate_embs=candidate_embs,
            )
        else:
            # TEXT-ONLY
            prompt_clean = formatted_prompt.replace("<image>", "")
            inputs = self.processor.tokenizer(
                prompt_clean, return_tensors="pt", padding=True
            ).to(self.device)

            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]

            inputs_embeds = self.model.get_input_embeddings()(input_ids)
            inputs_embeds = self._replace_custom_tokens(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                user_emb=user_emb,
                history_embs=history_embs,
                candidate_embs=candidate_embs,
            )

        # Manual greedy generation loop
        # NOTE: O(n^2) without KV cache. Acceptable for now; optimize later.
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            generated_tokens = []
            current_embeds = inputs_embeds
            current_attention = attention_mask

            for _ in range(max_new_tokens):
                outputs = self.model.model(
                    inputs_embeds=current_embeds,
                    attention_mask=current_attention,
                    return_dict=True,
                )

                logits = self.model.lm_head(outputs.last_hidden_state[:, -1:, :])
                next_token_id = logits.argmax(dim=-1)
                generated_tokens.append(next_token_id.item())

                # Stop on EOS
                if next_token_id.item() == self.processor.tokenizer.eos_token_id:
                    break

                # Append next token embedding
                next_token_emb = self.model.get_input_embeddings()(next_token_id)
                current_embeds = torch.cat([current_embeds, next_token_emb], dim=1)
                current_attention = torch.cat(
                    [
                        current_attention,
                        torch.ones(
                            (1, 1),
                            device=self.device,
                            dtype=current_attention.dtype,
                        ),
                    ],
                    dim=1,
                )

        return self.processor.tokenizer.decode(
            generated_tokens, skip_special_tokens=True
        ).strip()
