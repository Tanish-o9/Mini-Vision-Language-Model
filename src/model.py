from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoProcessor, CLIPModel, AutoTokenizer
from peft import LoraConfig, PeftModel, get_peft_model


class MiniVLM(nn.Module):
    def __init__(self, vision_name, language_name, use_lora=False, lora_r=8,
                 lora_alpha=16, lora_dropout=0.05, resample_tokens=None,
                 load_in_4bit=False, load_in_8bit=False):
        super().__init__()
        self.device_name = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.device_name == "cuda" else torch.float32

        self.processor = AutoProcessor.from_pretrained(vision_name)
        self.clip = CLIPModel.from_pretrained(vision_name)
        self.clip = self.clip.to(device=self.device_name, dtype=self.dtype)
        self.clip.eval()
        for p in self.clip.parameters():
            p.requires_grad = False

        self.tokenizer = AutoTokenizer.from_pretrained(language_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

        quantization_config = None
        if load_in_4bit or load_in_8bit:
            try:
                from transformers import BitsAndBytesConfig
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=load_in_4bit,
                    load_in_8bit=load_in_8bit,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
            except Exception as e:
                print(f"Warning: BitsAndBytesConfig not available ({e}), loading standard precision.")
                quantization_config = None

        qwen_kwargs = {
            "torch_dtype": self.dtype,
            "low_cpu_mem_usage": True,
        }
        if quantization_config is not None:
            qwen_kwargs["quantization_config"] = quantization_config
            qwen_kwargs["device_map"] = "auto"
            self.qwen = AutoModelForCausalLM.from_pretrained(language_name, **qwen_kwargs)
        else:
            self.qwen = AutoModelForCausalLM.from_pretrained(
                language_name, **qwen_kwargs
            ).to(self.device_name)

        for p in self.qwen.parameters():
            p.requires_grad = False

        hidden_size = self.qwen.config.hidden_size
        vision_size = self.clip.config.vision_config.hidden_size
        self.projector = nn.Sequential(
            nn.Linear(vision_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        ).to(device=self.device_name, dtype=self.dtype)

        self.resample_tokens = resample_tokens
        if resample_tokens is not None and resample_tokens != 49:
            grid_dim = int(resample_tokens ** 0.5)
            self.resampler = nn.AdaptiveAvgPool2d((grid_dim, grid_dim))
        else:
            self.resampler = None

        # Precompute visual semantic vocab text embeddings for instant zero-shot visual grounding (LLaVA-aligned)
        self._init_visual_vocab()

        self.use_lora = use_lora
        if use_lora:
            if load_in_4bit or load_in_8bit:
                try:
                    from peft import prepare_model_for_kbit_training
                    self.qwen = prepare_model_for_kbit_training(self.qwen)
                except Exception:
                    pass
            config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=["q_proj", "v_proj"],
                bias="none",
                task_type="CAUSAL_LM",
            )
            self.qwen = get_peft_model(self.qwen, config)

    def _extract_tensor(self, out):
        if hasattr(out, "pooler_output") and out.pooler_output is not None:
            return out.pooler_output
        if hasattr(out, "text_embeds") and out.text_embeds is not None:
            return out.text_embeds
        if hasattr(out, "image_embeds") and out.image_embeds is not None:
            return out.image_embeds
        if isinstance(out, (tuple, list)):
            return out[0]
        return out

    def _init_visual_vocab(self):
        """Precompute text embeddings for fine-grained everyday visual concepts with calibrated CLIP scale."""
        self.vocab_concepts = [
            # Fruits & Foods
            "an apple", "a banana", "an orange fruit", "a mango", "strawberries", "grapes",
            "a watermelon", "a tomato", "fresh vegetables", "a pizza", "a burger or sandwich",
            "pasta or noodles", "a plate of food or meal", "bread or bakery food",
            "a cake or sweet dessert", "a cup of coffee or tea", "a drink or beverage bottle",
            # Clothing & Fashion
            "a t-shirt or casual shirt", "a jacket, coat, or hoodie",
            "pants, jeans, or trousers", "shoes, sneakers, or footwear", "a dress or gown",
            "a watch on wrist", "glasses or eyewear", "a hat or cap", "a handbag or backpack",
            # Animals & Nature
            "a dog or puppy", "a cat or kitten", "a bird perched or flying", "a horse",
            "wild animals in nature", "fish underwater", "flowers or garden blossoms",
            "trees, plants, or green foliage", "outdoor landscape or scenery", "a beach and ocean water",
            "the sky and clouds", "mountains or hills",
            # People & Portraits
            "a hand holding an object", "a person or human portrait", "a man", "a woman", "a child or kid",
            # Technology & Electronics
            "a smartphone or mobile phone", "a laptop computer", "a computer monitor display",
            "headphones or audio earphones", "a television screen", "a camera",
            # Vehicles & Transport
            "a car, automobile, or road vehicle", "a motorcycle or scooter", "a bicycle or bike",
            "an airplane in flight", "a train or railway", "a boat or ship",
            # Everyday Objects & Home
            "a chair or armchair", "a sofa or couch", "a table or desk", "a bed or bedroom",
            "a book, notebook, or paper document", "a pen, pencil, or stationery",
            "an indoor room interior", "a building, house, or street scene",
            # Digital & Software
            "a computer desktop screen or software application", "a mobile app user interface screenshot"
        ]
        self.vocab_colors = [
            "blue", "navy blue", "cyan or light blue", "orange", "brown", "green", "red",
            "yellow", "black", "white", "gray", "purple", "pink"
        ]
        self.color_prompts = [f"a photo with {c} as the dominant color" for c in self.vocab_colors]
        try:
            with torch.no_grad():
                c_prompts = [f"a photo of {c}" for c in self.vocab_concepts]
                c_tokens = self.processor(text=c_prompts, padding=True, return_tensors="pt").to(self.device_name)
                raw_c = self.clip.get_text_features(**c_tokens)
                feat_c = self._extract_tensor(raw_c)
                self.text_feat_concepts = (feat_c / feat_c.norm(dim=-1, keepdim=True)).to(self.dtype)

                col_tokens = self.processor(text=self.color_prompts, padding=True, return_tensors="pt").to(self.device_name)
                raw_col = self.clip.get_text_features(**col_tokens)
                feat_col = self._extract_tensor(raw_col)
                self.text_feat_colors = (feat_col / feat_col.norm(dim=-1, keepdim=True)).to(self.dtype)
        except Exception as e:
            print(f"[WARN] Could not initialize visual vocab: {e}")
            self.text_feat_concepts = None
            self.text_feat_colors = None

    def extract_visual_semantics(self, image):
        """Extract fine-grained visual concept tags via calibrated CLIP in ~2ms."""
        if getattr(self, "text_feat_concepts", None) is None:
            return ""
        try:
            with torch.no_grad():
                img_inputs = self.processor(images=image.convert("RGB"), return_tensors="pt")["pixel_values"].to(self.device_name, dtype=self.dtype)
                raw_img = self.clip.get_image_features(pixel_values=img_inputs)
                img_feat = self._extract_tensor(raw_img)
                img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)

                scale = getattr(self.clip, "logit_scale", torch.tensor(4.6)).exp().to(self.dtype)

                # Concept probability distribution
                sim_c = (img_feat @ self.text_feat_concepts.to(dtype=self.dtype).T)[0]
                probs_c = (sim_c * scale).softmax(dim=-1)

                top_k = torch.topk(probs_c, k=3)
                top_indices = top_k.indices.tolist()
                top_probs = top_k.values.tolist()

                p0 = top_probs[0]
                primary = self.vocab_concepts[top_indices[0]]

                # Dominant color probability distribution
                sim_col = (img_feat @ self.text_feat_colors.to(dtype=self.dtype).T)[0]
                probs_col = (sim_col * scale).softmax(dim=-1)
                top_col_idx = torch.argmax(probs_col).item()
                top_col = self.vocab_colors[top_col_idx] if probs_col[top_col_idx].item() >= 0.15 else None

                parts = [f"Primary subject: {primary}"]

                # Only include secondary details if statistically prominent (>= 15% and not overshadowed 3x)
                if len(top_probs) > 1:
                    p1 = top_probs[1]
                    if p1 >= 0.15 and p0 < 3.0 * p1:
                        secondary = self.vocab_concepts[top_indices[1]]
                        if secondary not in primary and primary not in secondary:
                            parts.append(f"Visual details: {secondary}")

                if top_col:
                    parts.append(f"Color: {top_col}")

                return ". ".join(parts)
        except Exception:
            return ""

    def _normalize_query(self, text: str) -> str:
        """Helper to ensure small LLMs properly parse Hindi/Hinglish intent without English pronoun confusion."""
        q = str(text).strip()
        q_lower = q.lower()
        hinglish_markers = [
            "kya h", "kya hai", "kya dikh", "kaun sa", "kon sa", "kya cheez", "ye kya",
            "kya dekh", "kya bana", "batao", "btao", "kya chiz"
        ]
        if any(m in q_lower for m in hinglish_markers) and "what is" not in q_lower:
            return f"{q} (What is shown in this image?)"
        return q

    def format_prompt(self, text, history=None, is_caption=False, visual_context=""):
        raw_text = "Describe this image in one clear sentence." if is_caption else str(text)
        instruction = self._normalize_query(raw_text)
        user_content = f"[Image Content: {visual_context}]\nQuestion: {instruction}" if visual_context else instruction

        messages = [
            {
                "role": "system",
                "content": (
                    "You are Mini-VLM, an expert multimodal vision assistant. "
                    "Identify the visual content and answer the user question directly, accurately, and concisely."
                ),
            }
        ]
        if history:
            for turn in history:
                if isinstance(turn, (list, tuple)) and len(turn) == 2:
                    u, a = turn
                    if u:
                        messages.append({"role": "user", "content": str(u)})
                    if a:
                        messages.append({"role": "assistant", "content": str(a)})
                elif isinstance(turn, dict):
                    messages.append(turn)
        messages.append({"role": "user", "content": user_content})

        if hasattr(self.tokenizer, "apply_chat_template") and getattr(self.tokenizer, "chat_template", None):
            return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        plain = ""
        for m in messages:
            role = "USER" if m["role"] == "user" else "ASSISTANT"
            plain += f"{role}: {m['content']}\n"
        plain += "ASSISTANT:"
        return plain

    def encode_image(self, pixel_values):
        pixel_values = pixel_values.to(device=self.device_name, dtype=self.dtype)
        with torch.no_grad():
            vision_output = self.clip.vision_model(pixel_values=pixel_values, output_hidden_states=True)
            # ViT-B/32: 1 CLS token + 49 patch tokens.
            # Using penultimate layer (hidden_states[-2]) preserves richer spatial/visual representations (LLaVA-style).
            patch_features = vision_output.hidden_states[-2][:, 1:, :]

            if self.resampler is not None:
                B, N, C = patch_features.shape
                grid = patch_features.reshape(B, 7, 7, C).permute(0, 3, 1, 2)
                downsampled = self.resampler(grid)
                patch_features = downsampled.permute(0, 2, 3, 1).reshape(B, -1, C)

        return patch_features

    def forward(self, prompt, answer, pixel_values=None, image_features=None,
                max_answer_tokens=96):
        prompts = [prompt] if isinstance(prompt, str) else list(prompt)
        answers = [answer] if isinstance(answer, (str, int, float)) else [str(a) for a in answer]
        batch_size = len(prompts)

        if image_features is None:
            image_features = self.encode_image(pixel_values)
        else:
            image_features = image_features.to(device=self.device_name, dtype=self.dtype)
            if image_features.ndim == 2:
                image_features = image_features.unsqueeze(0)
            if self.resampler is not None and image_features.shape[1] == 49:
                B, N, C = image_features.shape
                grid = image_features.reshape(B, 7, 7, C).permute(0, 3, 1, 2)
                image_features = self.resampler(grid).permute(0, 2, 3, 1).reshape(B, -1, C)

        visual_tokens = self.projector(image_features)

        prompt_enc = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            add_special_tokens=True,
        ).to(self.device_name)

        answer_enc = self.tokenizer(
            answers,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_answer_tokens,
            add_special_tokens=False,
        ).to(self.device_name)

        prompt_ids = prompt_enc.input_ids
        answer_ids = answer_enc.input_ids
        prompt_mask = prompt_enc.attention_mask
        answer_mask = answer_enc.attention_mask

        eos = self.tokenizer.eos_token_id
        if eos is not None:
            eos_tensor = torch.full((batch_size, 1), eos, dtype=torch.long, device=self.device_name)
            eos_mask = torch.ones((batch_size, 1), dtype=torch.long, device=self.device_name)
            answer_ids = torch.cat([answer_ids, eos_tensor], dim=1)
            answer_mask = torch.cat([answer_mask, eos_mask], dim=1)

        embedding_layer = self.qwen.get_input_embeddings()
        prompt_embeds = embedding_layer(prompt_ids)
        answer_embeds = embedding_layer(answer_ids)

        inputs_embeds = torch.cat([visual_tokens, prompt_embeds, answer_embeds], dim=1)
        visual_mask = torch.ones(
            (batch_size, visual_tokens.shape[1]), dtype=torch.long, device=self.device_name
        )
        attention_mask = torch.cat([visual_mask, prompt_mask, answer_mask], dim=1)

        labels = torch.full(
            inputs_embeds.shape[:2], -100, dtype=torch.long, device=self.device_name
        )
        prefix_len = visual_tokens.shape[1] + prompt_ids.shape[1]

        target_answer = answer_ids.clone()
        if self.tokenizer.pad_token_id is not None:
            target_answer[target_answer == self.tokenizer.pad_token_id] = -100
        labels[:, prefix_len : prefix_len + target_answer.shape[1]] = target_answer

        output = self.qwen(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            use_cache=False,
        )
        return output.loss

    @torch.no_grad()
    def answer(self, image, question, history=None, max_new_tokens=80, secondary_image=None, return_meta=False):
        self.eval()
        pixels = self.processor(images=image.convert("RGB"), return_tensors="pt")["pixel_values"]
        visual_features = self.encode_image(pixels)
        visual_tokens = self.projector(visual_features)

        vis_cue1 = self.extract_visual_semantics(image)
        if secondary_image is not None:
            pixels2 = self.processor(images=secondary_image.convert("RGB"), return_tensors="pt")["pixel_values"]
            visual_features2 = self.encode_image(pixels2)
            visual_tokens2 = self.projector(visual_features2)
            visual_tokens = torch.cat([visual_tokens, visual_tokens2], dim=1)
            vis_cue2 = self.extract_visual_semantics(secondary_image)
            visual_context = f"Image 1: {vis_cue1} | Image 2: {vis_cue2}"
            formatted_q = f"Image 1 and Image 2 are provided. {question}"
        else:
            visual_context = vis_cue1
            formatted_q = question

        prompt = self.format_prompt(formatted_q, history=history, is_caption=False, visual_context=visual_context)
        prompt_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device_name)
        prompt_embeds = self.qwen.get_input_embeddings()(prompt_ids)

        # Smoothly blend visual projector tokens to preserve language generation fidelity
        scaled_visual_tokens = visual_tokens * 0.1

        inputs_embeds = torch.cat([scaled_visual_tokens, prompt_embeds], dim=1)
        attention_mask = torch.ones(
            inputs_embeds.shape[:2], dtype=torch.long, device=self.device_name
        )
        generated = self.qwen.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.15,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        ans = self.tokenizer.decode(generated[0], skip_special_tokens=True).strip()
        # Clean any accidental assistant prompt label bleed
        if "assistant\n" in ans:
            ans = ans.split("assistant\n")[-1].strip()
        if return_meta:
            return ans, visual_context
        return ans

    def save_projector(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.projector.state_dict(), path)

    def load_projector(self, path):
        state = torch.load(path, map_location=self.device_name, weights_only=True)
        self.projector.load_state_dict(state)

    def save_stage2(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.save_projector(directory / "projector.pt")
        self.qwen.save_pretrained(directory / "lora")
        self.tokenizer.save_pretrained(directory / "tokenizer")

    def load_lora(self, adapter_dir):
        self.qwen = PeftModel.from_pretrained(self.qwen, adapter_dir)
        self.qwen.eval()
