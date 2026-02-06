import random
import pickle
import json
import gzip
import logging

import torch
from torch.cuda.amp import autocast as autocast
import torch.nn as nn
import numpy as np

from models.recsys_model import *
from models.llm4rec import *
from sentence_transformers import SentenceTransformer

# SmolVLM imports (conditional)
try:
    from models.smolvlm4rec import SmolVLM4Rec
    from utils_image import ImageLoader
    SMOLVLM_AVAILABLE = True
except ImportError:
    SMOLVLM_AVAILABLE = False

logger = logging.getLogger(__name__)


class two_layer_mlp(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.fc1 = nn.Linear(dims, 128)
        self.fc2 = nn.Linear(128, dims)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.sigmoid(x)
        x1 = self.fc2(x)
        return x, x1

class A_llmrec_model(nn.Module):
    def __init__(self, args):
        super().__init__()
        rec_pre_trained_data = args.rec_pre_trained_data
        self.args = args
        self.device = args.device
        
        with open(f'./data/amazon/{args.rec_pre_trained_data}_text_name_dict.json.gz','rb') as ft:
            self.text_name_dict = pickle.load(ft)
        
        self.recsys = RecSys(args.recsys, rec_pre_trained_data, self.device)
        self.item_num = self.recsys.item_num
        self.rec_sys_dim = self.recsys.hidden_units
        self.sbert_dim = 768
        
        self.mlp = two_layer_mlp(self.rec_sys_dim)
        if args.pretrain_stage1:
            self.sbert = SentenceTransformer('nq-distilbert-base-v1')
            self.mlp2 = two_layer_mlp(self.sbert_dim)
        
        self.mse = nn.MSELoss()
        
        self.maxlen = args.maxlen
        self.NDCG = 0
        self.HIT = 0
        self.rec_NDCG = 0
        self.rec_HIT = 0
        self.lan_NDCG=0
        self.lan_HIT=0
        self.num_user = 0
        self.yes = 0
        
        self.bce_criterion = torch.nn.BCEWithLogitsLoss()
        
        # Check if using SmolVLM
        self.use_smolvlm = getattr(args, 'use_smolvlm', False)
        self.num_history_images = getattr(args, 'num_history_images', 3)
        
        if args.pretrain_stage2 or args.inference:
            if self.use_smolvlm:
                # Use SmolVLM for multimodal recommendations
                if not SMOLVLM_AVAILABLE:
                    raise ImportError("SmolVLM not available. Install required packages.")
                
                logger.info("Initializing SmolVLM for Stage-2...")
                self.vlm = SmolVLM4Rec(device=self.device)
                llm_hidden_size = self.vlm.hidden_size  # 960 for SmolVLM-500M
                
                # Load image URL dict
                image_dict_path = f'./data/amazon/{args.rec_pre_trained_data}_image_url_dict.json.gz'
                logger.info(f"Loading image URL dict from: {image_dict_path}")
                with gzip.open(image_dict_path, 'rt', encoding='utf-8') as f:
                    self.image_url_dict = json.load(f)
                self.image_url_dict = {int(k): v for k, v in self.image_url_dict.items()}
                logger.info(f"Loaded {len(self.image_url_dict)} image URLs")
                
                # Initialize image loader with disk cache
                self.image_loader = ImageLoader(
                    timeout=10,
                    max_retries=2,
                    cache_dir=f'./data/amazon/image_cache/{args.rec_pre_trained_data}',
                )
                
            else:
                # Use original OPT-based LLM
                self.llm = llm4rec(device=self.device, llm_model=args.llm)
                llm_hidden_size = self.llm.llm_model.config.hidden_size  # 4096 for OPT-6.7B
            
            # Projection networks (dimensions depend on LLM)
            self.log_emb_proj = nn.Sequential(
                nn.Linear(self.rec_sys_dim, llm_hidden_size),
                nn.LayerNorm(llm_hidden_size),
                nn.LeakyReLU(),
                nn.Linear(llm_hidden_size, llm_hidden_size)
            )
            nn.init.xavier_normal_(self.log_emb_proj[0].weight)
            nn.init.xavier_normal_(self.log_emb_proj[3].weight)

            self.item_emb_proj = nn.Sequential(
                nn.Linear(128, llm_hidden_size),
                nn.LayerNorm(llm_hidden_size),
                nn.GELU(),
                nn.Linear(llm_hidden_size, llm_hidden_size)
            )
            nn.init.xavier_normal_(self.item_emb_proj[0].weight)
            nn.init.xavier_normal_(self.item_emb_proj[3].weight)
            
            logger.info(f"Projection dimensions: {self.rec_sys_dim} -> {llm_hidden_size}, 128 -> {llm_hidden_size}")
            
    def save_model(self, args, epoch1=None, epoch2=None):
        out_dir = f'./models/saved_models/'
        create_dir(out_dir)
        out_dir += f'{args.rec_pre_trained_data}_{args.recsys}_{epoch1}_'
        if args.pretrain_stage1:
            torch.save(self.sbert.state_dict(), out_dir + 'sbert.pt')
            torch.save(self.mlp.state_dict(), out_dir + 'mlp.pt')
            torch.save(self.mlp2.state_dict(), out_dir + 'mlp2.pt') 
        
        # Use 'smolvlm' in filename when using SmolVLM, otherwise use args.llm
        llm_name = 'smolvlm' if self.use_smolvlm else args.llm
        out_dir += f'{llm_name}_{epoch2}_'
        if args.pretrain_stage2:
            torch.save(self.log_emb_proj.state_dict(), out_dir + 'log_proj.pt')
            torch.save(self.item_emb_proj.state_dict(), out_dir + 'item_proj.pt')
            logger.info(f"Saved Stage-2 checkpoints to: {out_dir}")
            
    def load_model(self, args, phase1_epoch=None, phase2_epoch=None):
        out_dir = f'./models/saved_models/{args.rec_pre_trained_data}_{args.recsys}_{phase1_epoch}_'
        
        mlp = torch.load(out_dir + 'mlp.pt', map_location = args.device)
        self.mlp.load_state_dict(mlp)
        del mlp
        for name, param in self.mlp.named_parameters():
            param.requires_grad = False

        if args.inference:
            # Use 'smolvlm' in filename when using SmolVLM, otherwise use args.llm
            llm_name = 'smolvlm' if self.use_smolvlm else args.llm
            out_dir += f'{llm_name}_{phase2_epoch}_'
            
            logger.info(f"Loading Stage-2 checkpoints from: {out_dir}")
            
            log_emb_proj_dict = torch.load(out_dir + 'log_proj.pt', map_location = args.device)
            self.log_emb_proj.load_state_dict(log_emb_proj_dict)
            del log_emb_proj_dict
            
            item_emb_proj_dict = torch.load(out_dir + 'item_proj.pt', map_location = args.device)
            self.item_emb_proj.load_state_dict(item_emb_proj_dict)
            del item_emb_proj_dict

    def find_item_text(self, item, title_flag=True, description_flag=True):
        t = 'title'
        d = 'description'
        t_ = 'No Title'
        d_ = 'No Description'
        if title_flag and description_flag:
            return [f'"{self.text_name_dict[t].get(i,t_)}, {self.text_name_dict[d].get(i,d_)}"' for i in item]
        elif title_flag and not description_flag:
            return [f'"{self.text_name_dict[t].get(i,t_)}"' for i in item]
        elif not title_flag and description_flag:
            return [f'"{self.text_name_dict[d].get(i,d_)}"' for i in item]
    
    def find_item_text_single(self, item, title_flag=True, description_flag=True):
        t = 'title'
        d = 'description'
        t_ = 'No Title'
        d_ = 'No Description'
        if title_flag and description_flag:
            return f'"{self.text_name_dict[t].get(item,t_)}, {self.text_name_dict[d].get(item,d_)}"'
        elif title_flag and not description_flag:
            return f'"{self.text_name_dict[t].get(item,t_)}"'
        elif not title_flag and description_flag:
            return f'"{self.text_name_dict[d].get(item,d_)}"'
        
    def get_item_emb(self, item_ids):
        with torch.no_grad():
            item_embs = self.recsys.model.item_emb(torch.LongTensor(item_ids).to(self.device))
            item_embs, _ = self.mlp(item_embs)
        
        return item_embs
    
    def get_history_images(self, item_ids, max_images=3):
        """
        Get images for the last N history items, looking back if images are missing.
        
        Args:
            item_ids: List of item IDs (chronological order, oldest first)
            max_images: Maximum number of images to load
            
        Returns:
            Tuple of (images, image_item_ids):
            - images: List of PIL Images
            - image_item_ids: List of item IDs that have images
        """
        if not self.use_smolvlm:
            return [], []
        
        images = []
        image_item_ids = []
        
        # Iterate from most recent to oldest
        for item_id in reversed(item_ids):
            if len(images) >= max_images:
                break
            
            # Check if item has an image URL
            image_url = self.image_url_dict.get(int(item_id))
            if not image_url:
                continue
            
            # Try to load the image
            img, error = self.image_loader.load_image(image_url)
            if img is not None:
                images.append(img)
                image_item_ids.append(item_id)
        
        # Reverse to get chronological order
        images = list(reversed(images))
        image_item_ids = list(reversed(image_item_ids))
        
        logger.debug(f"Loaded {len(images)}/{max_images} images from {len(item_ids)} history items")
        return images, image_item_ids
    
    def make_interact_text_with_images(self, interact_ids, interact_max_num, image_item_ids, images):
        """
        Create interaction text with image tokens for SmolVLM.
        
        Uses SmolVLM's reserved tokens for embedding injection.
        IMPORTANT: Only adds <image> tokens for items we have actual images for,
        and only once per image (handles duplicate items in history).
        
        Args:
            interact_ids: List of interaction item IDs
            interact_max_num: Maximum number of history items to include
            image_item_ids: List of item IDs that have images (same order as images)
            images: List of PIL Images (used to track count)
            
        Returns:
            Tuple of (interact_text, final_interact_ids, ordered_images)
            - ordered_images: Images in the order they appear in text
        """
        # SmolVLM reserved token for history embeddings
        HISTORY_EMB_TOKEN = "<|reserved_special_token_1|>"
        
        interact_item_titles_ = self.find_item_text(interact_ids, title_flag=True, description_flag=False)
        interact_text = []
        
        if interact_max_num == 'all':
            selected_ids = interact_ids
            selected_titles = interact_item_titles_
        else:
            selected_ids = interact_ids[-interact_max_num:]
            selected_titles = interact_item_titles_[-interact_max_num:]
        
        # Create a mapping from item_id to image (preserves order)
        # Use list of tuples to handle the case where we need to consume images in order
        image_item_to_img = {}
        for item_id, img in zip(image_item_ids, images):
            # Convert to int for consistent comparison
            item_id_int = int(item_id)
            if item_id_int not in image_item_to_img:
                image_item_to_img[item_id_int] = img
        
        # Track which images we've used (to avoid adding <image> for same item twice)
        used_image_items = set()
        ordered_images = []
        
        for item_id, title in zip(selected_ids, selected_titles):
            item_id_int = int(item_id)
            
            # Only add <image> if: 1) we have an image for this item, 2) we haven't used it yet
            if item_id_int in image_item_to_img and item_id_int not in used_image_items:
                interact_text.append(f'<image>{title}{HISTORY_EMB_TOKEN}')
                ordered_images.append(image_item_to_img[item_id_int])
                used_image_items.add(item_id_int)
            else:
                interact_text.append(f'{title}{HISTORY_EMB_TOKEN}')
        
        interact_text = ','.join(interact_text)
        return interact_text, list(selected_ids), ordered_images
    
    def forward(self, data, optimizer=None, batch_iter=None, mode='phase1'):
        if mode == 'phase1':
            self.pre_train_phase1(data, optimizer, batch_iter)
        if mode == 'phase2':
            self.pre_train_phase2(data, optimizer, batch_iter)
        if mode =='generate':
            self.generate(data)

    def pre_train_phase1(self,data,optimizer, batch_iter):
        epoch, total_epoch, step, total_step = batch_iter
        
        self.sbert.train()
        optimizer.zero_grad()

        u, seq, pos, neg = data
        indices = [self.maxlen*(i+1)-1 for i in range(u.shape[0])]
        
        with torch.no_grad():
            log_emb, pos_emb, neg_emb = self.recsys.model(u, seq, pos, neg, mode='item')
            
        log_emb_ = log_emb[indices]
        pos_emb_ = pos_emb[indices]
        neg_emb_ = neg_emb[indices]
        pos_ = pos.reshape(pos.size)[indices]
        neg_ = neg.reshape(neg.size)[indices]
        
        start_inx = 0
        end_inx = 60
        iterss = 0
        mean_loss = 0
        bpr_loss = 0
        gt_loss = 0
        rc_loss = 0
        text_rc_loss = 0
        original_loss = 0
        while start_inx < len(log_emb_):
            log_emb = log_emb_[start_inx:end_inx]
            pos_emb = pos_emb_[start_inx:end_inx]
            neg_emb = neg_emb_[start_inx:end_inx]
            
            pos__ = pos_[start_inx:end_inx]
            neg__ = neg_[start_inx:end_inx]
            
            start_inx = end_inx
            end_inx += 60
            iterss +=1
            
            pos_text = self.find_item_text(pos__)
            neg_text = self.find_item_text(neg__)
            
            pos_token = self.sbert.tokenize(pos_text)
            pos_text_embedding= self.sbert({'input_ids':pos_token['input_ids'].to(self.device),'attention_mask':pos_token['attention_mask'].to(self.device)})['sentence_embedding']
            neg_token = self.sbert.tokenize(neg_text)
            neg_text_embedding= self.sbert({'input_ids':neg_token['input_ids'].to(self.device),'attention_mask':neg_token['attention_mask'].to(self.device)})['sentence_embedding']
            
            pos_text_matching, pos_proj = self.mlp(pos_emb)
            neg_text_matching, neg_proj = self.mlp(neg_emb)
            
            pos_text_matching_text, pos_text_proj = self.mlp2(pos_text_embedding)
            neg_text_matching_text, neg_text_proj = self.mlp2(neg_text_embedding)
            
            pos_logits, neg_logits = (log_emb*pos_proj).mean(axis=1), (log_emb*neg_proj).mean(axis=1)
            pos_labels, neg_labels = torch.ones(pos_logits.shape, device=pos_logits.device), torch.zeros(neg_logits.shape, device=pos_logits.device)

            loss = self.bce_criterion(pos_logits, pos_labels)
            loss += self.bce_criterion(neg_logits, neg_labels)
            
            matching_loss = self.mse(pos_text_matching,pos_text_matching_text) + self.mse(neg_text_matching,neg_text_matching_text)
            reconstruction_loss = self.mse(pos_proj,pos_emb) + self.mse(neg_proj,neg_emb)
            text_reconstruction_loss = self.mse(pos_text_proj,pos_text_embedding.data) + self.mse(neg_text_proj,neg_text_embedding.data)
            
            total_loss = loss + matching_loss + 0.5*reconstruction_loss + 0.2*text_reconstruction_loss
            total_loss.backward()
            optimizer.step()
            
            mean_loss += total_loss.item()
            bpr_loss += loss.item()
            gt_loss += matching_loss.item()
            rc_loss += reconstruction_loss.item()
            text_rc_loss += text_reconstruction_loss.item()
            
        print("loss in epoch {}/{} iteration {}/{}: {} / BPR loss: {} / Matching loss: {} / Item reconstruction: {} / Text reconstruction: {}".format(epoch, total_epoch, step, total_step, mean_loss/iterss, bpr_loss/iterss, gt_loss/iterss, rc_loss/iterss, text_rc_loss/iterss))
    
    def make_interact_text(self, interact_ids, interact_max_num):
        interact_item_titles_ = self.find_item_text(interact_ids, title_flag=True, description_flag=False)
        interact_text = []
        if interact_max_num == 'all':
            for title in interact_item_titles_:
                interact_text.append(title + '[HistoryEmb]')
        else:
            for title in interact_item_titles_[-interact_max_num:]:
                interact_text.append(title + '[HistoryEmb]')
            interact_ids = interact_ids[-interact_max_num:]
            
        interact_text = ','.join(interact_text)
        return interact_text, interact_ids
    
    def make_candidate_text(self, interact_ids, candidate_num, target_item_id, target_item_title, use_smolvlm_tokens=False):
        """
        Create candidate text with embedding tokens.
        
        Args:
            interact_ids: User's interaction history (to exclude from candidates)
            candidate_num: Number of candidates to generate
            target_item_id: Ground truth item ID
            target_item_title: Ground truth item title
            use_smolvlm_tokens: If True, use SmolVLM reserved tokens
        """
        # Token to use for candidate embeddings
        if use_smolvlm_tokens:
            CAND_TOKEN = "<|reserved_special_token_2|>"
        else:
            CAND_TOKEN = "[CandidateEmb]"
        
        neg_item_id = []
        while len(neg_item_id)<50:
            t = np.random.randint(1, self.item_num+1)
            if not (t in interact_ids or t in neg_item_id):
                neg_item_id.append(t)
        random.shuffle(neg_item_id)
        
        candidate_ids = [target_item_id]
        candidate_text = [target_item_title + CAND_TOKEN]

        for neg_candidate in neg_item_id[:candidate_num - 1]:
            candidate_text.append(self.find_item_text_single(neg_candidate, title_flag=True, description_flag=False) + CAND_TOKEN)
            candidate_ids.append(neg_candidate)
                
        random_ = np.random.permutation(len(candidate_text))
        candidate_text = np.array(candidate_text)[random_]
        candidate_ids = np.array(candidate_ids)[random_]
            
        return ','.join(candidate_text), candidate_ids
    
    def pre_train_phase2(self, data, optimizer, batch_iter):
        """Stage-2 training - dispatches to SmolVLM or OPT based on config."""
        if self.use_smolvlm:
            self.pre_train_phase2_smolvlm(data, optimizer, batch_iter)
        else:
            self.pre_train_phase2_opt(data, optimizer, batch_iter)
    
    def pre_train_phase2_opt(self, data, optimizer, batch_iter):
        """Original OPT-based Stage-2 training."""
        epoch, total_epoch, step, total_step = batch_iter
        
        optimizer.zero_grad()
        u, seq, pos, neg = data
        mean_loss = 0
        
        text_input = []
        text_output = []
        interact_embs = []
        candidate_embs = []
        self.llm.eval()
        
        with torch.no_grad():
            log_emb = self.recsys.model(u,seq,pos,neg, mode = 'log_only')
            
        for i in range(len(u)):
            target_item_id = pos[i][-1]
            target_item_title = self.find_item_text_single(target_item_id, title_flag=True, description_flag=False)
            
            interact_text, interact_ids = self.make_interact_text(seq[i][seq[i]>0], 10)
            candidate_num = 20
            candidate_text, candidate_ids = self.make_candidate_text(seq[i][seq[i]>0], candidate_num, target_item_id, target_item_title)
            
            input_text = ''
            input_text += ' is a user representation.'
                
            if self.args.rec_pre_trained_data == 'Movies_and_TV':
                input_text += 'This user has watched '
            elif self.args.rec_pre_trained_data == 'Video_Games':
                input_text += 'This user has played '
            elif self.args.rec_pre_trained_data == 'Luxury_Beauty' or self.args.rec_pre_trained_data == 'Toys_and_Games':
                input_text += 'This user has bought '
            else:
                input_text += 'This user has bought '
                
            input_text += interact_text
            
            if self.args.rec_pre_trained_data == 'Movies_and_TV':
                input_text +=' in the previous. Recommend one next movie for this user to watch next from the following movie title set, '
            elif self.args.rec_pre_trained_data == 'Video_Games':
                input_text +=' in the previous. Recommend one next game for this user to play next from the following game title set, '            
            elif self.args.rec_pre_trained_data == 'Luxury_Beauty' or self.args.rec_pre_trained_data == 'Toys_and_Games':
                input_text +=' in the previous. Recommend one next item for this user to buy next from the following item title set, '
            else:
                input_text +=' in the previous. Recommend one next item for this user to buy next from the following item title set, '
                    
            input_text += candidate_text
            input_text += '. The recommendation is '

            text_input.append(input_text)
            text_output.append(target_item_title)

            interact_embs.append(self.item_emb_proj(self.get_item_emb(interact_ids)))
            candidate_embs.append(self.item_emb_proj(self.get_item_emb(candidate_ids)))
            
        samples = {'text_input': text_input, 'text_output': text_output, 'interact': interact_embs, 'candidate':candidate_embs}
        log_emb = self.log_emb_proj(log_emb)
        loss_rm = self.llm(log_emb, samples)
        loss_rm.backward()
        optimizer.step()
        mean_loss += loss_rm.item()
        print("A-LLMRec model loss in epoch {}/{} iteration {}/{}: {}".format(epoch, total_epoch, step, total_step, mean_loss))
    
    def pre_train_phase2_smolvlm(self, data, optimizer, batch_iter):
        """SmolVLM-based Stage-2 training with image support."""
        epoch, total_epoch, step, total_step = batch_iter
        
        optimizer.zero_grad()
        u, seq, pos, neg = data
        mean_loss = 0
        
        text_input = []
        text_output = []
        interact_embs = []
        candidate_embs = []
        all_images = []
        
        with torch.no_grad():
            log_emb = self.recsys.model(u, seq, pos, neg, mode='log_only')
        
        for i in range(len(u)):
            target_item_id = pos[i][-1]
            target_item_title = self.find_item_text_single(target_item_id, title_flag=True, description_flag=False)
            
            # Get history items
            history_ids = seq[i][seq[i] > 0]
            
            # Load images for last N history items (with fallback)
            images, image_item_ids = self.get_history_images(history_ids, max_images=self.num_history_images)
            
            # Create interaction text with image tokens where images exist
            # Returns ordered_images to ensure <image> tokens match actual images
            interact_text, interact_ids, ordered_images = self.make_interact_text_with_images(
                history_ids, 10, image_item_ids, images
            )
            
            # Create candidate text with SmolVLM tokens
            candidate_num = 20
            candidate_text, candidate_ids = self.make_candidate_text(
                history_ids, candidate_num, target_item_id, target_item_title,
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
            all_images.append(ordered_images)  # Use ordered_images to match <image> tokens
            
            # Get projected embeddings
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
        
        # Project user representation
        log_emb = self.log_emb_proj(log_emb)
        
        # Forward pass through SmolVLM
        loss_rm = self.vlm(log_emb, samples)
        loss_rm.backward()
        optimizer.step()
        mean_loss += loss_rm.item()
        
        # Log with image stats
        num_images_loaded = sum(len(imgs) for imgs in all_images)
        print(f"SmolVLM loss epoch {epoch}/{total_epoch} iter {step}/{total_step}: {mean_loss:.4f} | Images: {num_images_loaded}/{len(u)*self.num_history_images}")
        
    def generate(self, data):
        """Inference - dispatches to SmolVLM or OPT based on config."""
        if self.use_smolvlm:
            return self.generate_smolvlm(data)
        else:
            return self.generate_opt(data)
    
    def generate_opt(self, data):
        """Original OPT-based inference."""
        u, seq, pos, neg, rank = data
        
        answer = []
        text_input = []
        interact_embs = []
        candidate_embs = []
        with torch.no_grad():
            log_emb = self.recsys.model(u,seq,pos,neg, mode = 'log_only')
            for i in range(len(u)):
                target_item_id = pos[i]
                target_item_title = self.find_item_text_single(target_item_id, title_flag=True, description_flag=False)
                
                interact_text, interact_ids = self.make_interact_text(seq[i][seq[i]>0], 10)
                candidate_num = 20
                candidate_text, candidate_ids = self.make_candidate_text(seq[i][seq[i]>0], candidate_num, target_item_id, target_item_title)
                
                input_text = ''
                input_text += ' is a user representation.'
                if self.args.rec_pre_trained_data == 'Movies_and_TV':
                    input_text += 'This user has watched '
                elif self.args.rec_pre_trained_data == 'Video_Games':
                    input_text += 'This user has played '
                elif self.args.rec_pre_trained_data == 'Luxury_Beauty' or self.args.rec_pre_trained_data == 'Toys_and_Games':
                    input_text += 'This user has bought '
                else:
                    input_text += 'This user has bought '
                    
                input_text += interact_text
                
                if self.args.rec_pre_trained_data == 'Movies_and_TV':
                    input_text +=' in the previous. Recommend one next movie for this user to watch next from the following movie title set, '
                elif self.args.rec_pre_trained_data == 'Video_Games':
                    input_text +=' in the previous. Recommend one next game for this user to play next from the following game title set, '            
                elif self.args.rec_pre_trained_data == 'Luxury_Beauty' or self.args.rec_pre_trained_data == 'Toys_and_Games':
                    input_text +=' in the previous. Recommend one next item for this user to buy next from the following item title set, '
                else:
                    input_text +=' in the previous. Recommend one next item for this user to buy next from the following item title set, '
                
                input_text += candidate_text
                input_text += '. The recommendation is '
                
                answer.append(target_item_title)
                text_input.append(input_text)
                
                interact_embs.append(self.item_emb_proj(self.get_item_emb(interact_ids)))
                candidate_embs.append(self.item_emb_proj(self.get_item_emb(candidate_ids)))
        
        log_emb = self.log_emb_proj(log_emb)
        atts_llm = torch.ones(log_emb.size()[:-1], dtype=torch.long).to(self.device)
        atts_llm = atts_llm.unsqueeze(1)
        log_emb = log_emb.unsqueeze(1)
        
        with torch.no_grad():
            self.llm.llm_tokenizer.padding_side = "left"
            llm_tokens = self.llm.llm_tokenizer(
                text_input,
                padding="longest",
                return_tensors="pt"
            ).to(self.device)
            
            with torch.cuda.amp.autocast():
                inputs_embeds = self.llm.llm_model.get_input_embeddings()(llm_tokens.input_ids)
                
                llm_tokens, inputs_embeds = self.llm.replace_hist_candi_token(llm_tokens, inputs_embeds, interact_embs, candidate_embs)
                    
                attention_mask = llm_tokens.attention_mask
                inputs_embeds = torch.cat([log_emb, inputs_embeds], dim=1)
                attention_mask = torch.cat([atts_llm, llm_tokens.attention_mask], dim=1)
                    
                outputs = self.llm.llm_model.generate(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    do_sample=False,
                    top_p=0.9,
                    temperature=1,
                    num_beams=1,
                    max_length=1024,
                    min_length=1,
                    pad_token_id=self.llm.llm_tokenizer.eos_token_id,
                    repetition_penalty=1.5,
                    length_penalty=1,
                    num_return_sequences=1,
                )

            outputs[outputs == 0] = 2 # convert output id 0 to 2 (eos_token_id)
            output_text = self.llm.llm_tokenizer.batch_decode(outputs, skip_special_tokens=True)
            output_text = [text.strip() for text in output_text]

        with open('./recommendation_output.txt', 'a') as f:
            for i in range(len(text_input)):
                f.write(text_input[i])
                f.write('\n\n')
                f.write('Answer: ' + answer[i])
                f.write('\n\n')
                f.write('LLM: ' + str(output_text[i]))
                f.write('\n\n')

        return output_text
    
    def generate_smolvlm(self, data):
        """SmolVLM-based inference with image support."""
        u, seq, pos, neg, rank = data
        
        answer = []
        output_text = []
        
        with torch.no_grad():
            log_emb = self.recsys.model(u, seq, pos, neg, mode='log_only')
            log_emb = self.log_emb_proj(log_emb)
            
            for i in range(len(u)):
                target_item_id = pos[i]
                target_item_title = self.find_item_text_single(target_item_id, title_flag=True, description_flag=False)
                answer.append(target_item_title)
                
                # Get history items
                history_ids = seq[i][seq[i] > 0]
                
                # Load images for last N history items
                images, image_item_ids = self.get_history_images(history_ids, max_images=self.num_history_images)
                
                # Create interaction text with image tokens
                # Returns ordered_images to ensure <image> tokens match actual images
                interact_text, interact_ids, ordered_images = self.make_interact_text_with_images(
                    history_ids, 10, image_item_ids, images
                )
                
                # Create candidate text with SmolVLM tokens
                candidate_num = 20
                candidate_text, candidate_ids = self.make_candidate_text(
                    history_ids, candidate_num, target_item_id, target_item_title,
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
                
                # Get projected embeddings
                history_embs = self.item_emb_proj(self.get_item_emb(interact_ids)).unsqueeze(0)
                candidate_embs = self.item_emb_proj(self.get_item_emb(candidate_ids)).unsqueeze(0)
                
                # Generate with SmolVLM (use ordered_images to match <image> tokens)
                generated = self.vlm.generate(
                    user_emb=log_emb[i:i+1],
                    prompt_text=input_text,
                    images=ordered_images,
                    history_embs=history_embs,
                    candidate_embs=candidate_embs,
                    max_new_tokens=64
                )
                output_text.append(generated)
                
                # Log result
                logger.debug(f"Sample {i}: Answer={target_item_title}, Generated={generated}, Images={len(images)}")
        
        # Write results to file
        with open('./recommendation_output_smolvlm.txt', 'a') as f:
            for i in range(len(answer)):
                f.write(f'Answer: {answer[i]}\n')
                f.write(f'SmolVLM: {output_text[i]}\n')
                f.write('-' * 50 + '\n')
        
        return output_text