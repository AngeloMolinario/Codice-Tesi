'''
Custom implementation of Rotary Position Embedding for 2D inputs.

Rotary Positional Embedding (RoPE) is a mechanism for incorporating positional information into transformer models.
Unlike absolute positional embeddings, RoPE encodes position using rotation matrices, which allows the model to
learn relative positional relationships between elements in a sequence.  Specifically, RoPE leverages trigonometric
functions (sine and cosine) to encode the position. The dot product between two RoPE-encoded vectors becomes dependent
on the *relative* positional difference, rather than absolute positions. This is crucial because relative position
is often more important than absolute position, and it allows the model to generalize to sequences of different lengths.
In vision transformers, RoPE is adapted to 2D inputs to encode the spatial arrangement of image patches.

'''
import torch
from core.vision_encoder.rope import RotaryEmbedding, apply_rotary_emb

class CustomRope2D:

    def __init__(self, dim, num_prompt, use_cls_token=False):
        self.dim = dim
        self.num_prompt = num_prompt
        self.use_cls_token = use_cls_token
        self.grid_size = None
        self.freq = None

    def init_tensors(self):
        self.rope = RotaryEmbedding(self.dim // 2)

    def update_grid(self, device, grid_h, grid_w):
        if self.grid_size != (grid_h, grid_w):
            self.grid_size = (grid_h, grid_w)

            self.rope = self.rope.to(device)

            if self.use_cls_token:
                # +1 to leave space for the cls token to be (0, 0)
                grid_y_range = torch.arange(grid_h, device=device) + 1
                grid_x_range = torch.arange(grid_w, device=device) + 1
            else:
                grid_y_range = torch.arange(grid_h, device=device)
                grid_x_range = torch.arange(grid_w, device=device)

            freqs_y = self.rope(grid_y_range)[:, None].expand(grid_h, grid_w, -1)
            freqs_x = self.rope(grid_x_range)[None, :].expand(grid_h, grid_w, -1)
            freq = torch.cat([freqs_x, freqs_y], dim=-1).reshape(grid_h * grid_w, -1)

            if self.use_cls_token:
                freq = torch.cat(
                    [torch.zeros(1, freq.shape[-1], device=device), freq], dim=0
                )

            self.freq = freq[None, ...]

        self.freq = self.freq.to(device)

    def __call__(self, q, k):
        '''
            Since RoPE is a positional encoding strategy, it is applied to the image patches and not to the context vectors used for prompt tuning.
            For this reason, we only apply the rotary embedding to the parts of the query and the key that correspond to the image patches,
            skipping the context vectors.
            The context vectors are not affected by the RoPE, as they are not spatially dependent and do not require positional encoding.
            After the RoPE is applied, the query and key tensors are reconstructed to include the context vectors.
        '''
        # q, k: [batch, heads, seq_len, dim]
        num_prompts = self.num_prompt

        if num_prompts > 0:
            if self.use_cls_token:
                # Split: cls + prompts + patches
                cls_q, prompt_q, patch_q = q[:, :, 0:1], q[:, :, 1:1+num_prompts], q[:, :, 1+num_prompts:]
                cls_k, prompt_k, patch_k = k[:, :, 0:1], k[:, :, 1:1+num_prompts], k[:, :, 1+num_prompts:]

                # Apply RoPE only to cls + patches (skip prompts)
                q_to_rotate = torch.cat([cls_q, patch_q], dim=2)
                k_to_rotate = torch.cat([cls_k, patch_k], dim=2)
                
                q_rotated = apply_rotary_emb(self.freq[:, None, :, :], q_to_rotate)
                k_rotated = apply_rotary_emb(self.freq[:, None, :, :], k_to_rotate)

                # Split back the rotated tensors
                cls_q_rot, patch_q_rot = q_rotated[:, :, :1], q_rotated[:, :, 1:]
                cls_k_rot, patch_k_rot = k_rotated[:, :, :1], k_rotated[:, :, 1:]

                # Reconstruct: cls + prompts + patches (prompts unchanged)
                q = torch.cat([cls_q_rot, prompt_q, patch_q_rot], dim=2)
                k = torch.cat([cls_k_rot, prompt_k, patch_k_rot], dim=2)
            else:
                # Split: prompts + patches
                prompt_q, patch_q = q[:, :, :num_prompts], q[:, :, num_prompts:]
                prompt_k, patch_k = k[:, :, :num_prompts], k[:, :, num_prompts:]

                # Apply RoPE only to patches (skip prompts)
                patch_q_rot = apply_rotary_emb(self.freq[:, None, :, :], patch_q)
                patch_k_rot = apply_rotary_emb(self.freq[:, None, :, :], patch_k)

                # Reconstruct: prompts + patches (prompts unchanged)
                q = torch.cat([prompt_q, patch_q_rot], dim=2)
                k = torch.cat([prompt_k, patch_k_rot], dim=2)
        else:
            # No prompts: apply RoPE to the entire sequence (cls + patches or just patches) like in the original RoPE2d
            q = apply_rotary_emb(self.freq[:, None, :, :], q)
            k = apply_rotary_emb(self.freq[:, None, :, :], k)

        return q, k

