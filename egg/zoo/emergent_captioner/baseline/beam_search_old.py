import numpy as np
import torch
import torch.nn.functional as F


def generate_beam(
    self,
    embed,
    entry_length=67,
    temperature=1.0,
    stop_token: str = ".",
):
    breakpoint()
    self.eval()

    bsz, h_dim = embed.shape[0], embed.shape[-1]

    stop_token_index = self.tokenizer.encode(stop_token)[0]
    device = embed.device

    seq_lengths = torch.ones(bsz * self.beam_size, device=device)
    is_stopped = torch.zeros(bsz * self.beam_size, device=device, dtype=torch.bool)

    with torch.no_grad():
        outputs = self.gpt(inputs_embeds=embed)
        logits = outputs.logits
        logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
        logits = logits.log_softmax(-1)  # bsz X vocab_Size

        scores, next_tokens = logits.topk(self.beam_size, dim=-1)  # bsz X beam_size

    # generated of shape bsz*beam_size X n_prefix_tokens X vocab_size
    generated = embed.repeat(self.beam_size, 1, 1)
    next_tokens, scores = next_tokens.view(-1), scores.view(-1)

    next_token_embed = self.gpt.transformer.wte(next_tokens).unsqueeze(1)
    generated = torch.cat((generated, next_token_embed), dim=1)

    tokens = next_tokens.unsqueeze(1)

    with torch.no_grad():
        for i in range(entry_length - 1):  # already done a decoding step
            breakpoint()
            outputs = self.gpt(inputs_embeds=generated)
            logits = outputs.logits
            logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
            logits = logits.log_softmax(-1)
            logits[is_stopped] = -float(np.inf)
            logits[is_stopped, 0] = 0

            scores_sum = scores[:, None] + logits
            seq_lengths[~is_stopped] += 1
            scores_sum_average = scores_sum / seq_lengths[:, None]
            tmp_scores_sum_average = scores_sum_average.view(bsz, -1)
            scores_sum_average, next_tokens = tmp_scores_sum_average.topk(
                self.beam_size, -1
            )

            next_tokens_source = next_tokens // scores_sum.shape[1]
            seq_lengths = torch.gather(
                seq_lengths.view(bsz, self.beam_size),
                dim=1,
                index=next_tokens_source,
            ).view(-1)
            next_tokens = next_tokens % scores_sum.shape[1]
            tokens = torch.gather(
                tokens.view(bsz, self.beam_size), dim=1, index=next_tokens_source
            ).view(bsz * self.beam_size, -1)
            tokens = torch.cat((tokens, next_tokens.view(-1).unsqueeze(1)), dim=1)

            batch_index = torch.arange(bsz).repeat_interleave(self.beam_size)
            generated = generated.view(bsz, self.beam_size, -1, h_dim)[
                batch_index.view(bsz, self.beam_size), next_tokens_source
            ]
            scores = scores_sum_average.view(-1) * seq_lengths
            is_stopped = torch.gather(
                is_stopped.view(bsz, self.beam_size),
                dim=1,
                index=next_tokens_source,
            ).view(-1)

            next_token_embed = self.gpt.transformer.wte(next_tokens)
            generated = torch.cat(
                (
                    generated.view(bsz * self.beam_size, -1, h_dim),
                    next_token_embed.view(bsz * self.beam_size, -1).unsqueeze(1),
                ),
                dim=1,
            )
            is_stopped = is_stopped + next_tokens.eq(stop_token_index).view(-1)
            if is_stopped.all():
                break
    scores = scores / seq_lengths
    output_list = tokens.cpu().numpy()
    output_texts = [
        self.tokenizer.decode(output[: int(length)])
        for output, length in zip(output_list, seq_lengths)
    ]
    order = scores.argsort(descending=True)
    output_texts = [output_texts[i] for i in order]
    return output_texts


def generate2(
    self,
    embed,
    tokens=None,
    prompt=None,
    entry_count=1,
    entry_length=67,  # maximum number of words
    top_p=0.8,
    temperature=1.0,
    stop_token: str = ".",
):
    """This implements top_p sampling."""
    self.eval()

    generated_list = []
    stop_token_index = self.tokenizer.encode(stop_token)[0]
    filter_value = -float("Inf")
    device = embed.device

    with torch.no_grad():
        for entry_idx in range(entry_count):
            if embed is not None:
                generated = embed
            else:
                if tokens is None:
                    tokens = torch.tensor(self.tokenizer.encode(prompt))
                    tokens = tokens.unsqueeze(0).to(device)

                generated = self.gpt.transformer.wte(tokens)

            for i in range(entry_length):

                outputs = self.gpt(inputs_embeds=generated)
                logits = outputs.logits
                logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(
                    F.softmax(sorted_logits, dim=-1), dim=-1
                )
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                    ..., :-1
                ].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits[:, indices_to_remove] = filter_value
                next_token = torch.argmax(logits, -1).unsqueeze(0)
                next_token_embed = self.gpt.transformer.wte(next_token)
                if tokens is None:
                    tokens = next_token
                else:
                    tokens = torch.cat((tokens, next_token), dim=1)
                generated = torch.cat((generated, next_token_embed), dim=1)
                if stop_token_index == next_token.item():
                    break

            output_list = list(tokens.squeeze().cpu().numpy())
            output_text = self.decode(output_list)
            generated_list.append(output_text)

    return generated_list[0]


def generate_beam_backup(
    self,
    embed,
    entry_length=67,
    temperature=1.0,
    stop_token: str = ".",
):
    self.eval()

    breakpoint()
    stop_token_index = self.tokenizer.encode(stop_token)[0]
    tokens = None
    scores = None
    device = embed.device
    seq_lengths = torch.ones(self.beam_size, device=device)
    is_stopped = torch.zeros(self.beam_size, device=device, dtype=torch.bool)
    with torch.no_grad():
        generated = embed
        for i in range(entry_length):
            outputs = self.gpt(inputs_embeds=generated)
            logits = outputs.logits
            logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
            logits = logits.softmax(-1).log()
            if scores is None:
                scores, next_tokens = logits.topk(self.beam_size, dim=-1)
                generated = generated.expand(self.beam_size, *generated.shape[1:])
                next_tokens, scores = next_tokens.permute(1, 0), scores.squeeze(0)
                if tokens is None:
                    tokens = next_tokens
                else:
                    tokens = tokens.expand(self.beam_size, *tokens.shape[1:])
                    tokens = torch.cat((tokens, next_tokens), dim=1)
            else:
                logits[is_stopped] = -float(np.inf)
                logits[is_stopped, 0] = 0
                scores_sum = scores[:, None] + logits
                seq_lengths[~is_stopped] += 1
                scores_sum_average = scores_sum / seq_lengths[:, None]
                scores_sum_average, next_tokens = scores_sum_average.view(-1).topk(
                    self.beam_size, -1
                )
                next_tokens_source = next_tokens // scores_sum.shape[1]
                seq_lengths = seq_lengths[next_tokens_source]
                next_tokens = next_tokens % scores_sum.shape[1]
                next_tokens = next_tokens.unsqueeze(1)
                tokens = tokens[next_tokens_source]
                tokens = torch.cat((tokens, next_tokens), dim=1)
                generated = generated[next_tokens_source]
                scores = scores_sum_average * seq_lengths
                is_stopped = is_stopped[next_tokens_source]
            next_token_embed = self.gpt.transformer.wte(next_tokens.squeeze()).view(
                generated.shape[0], 1, -1
            )
            generated = torch.cat((generated, next_token_embed), dim=1)
            is_stopped = is_stopped + next_tokens.eq(stop_token_index).squeeze()
            if is_stopped.all():
                break
    scores = scores / seq_lengths
    output_list = tokens.cpu().numpy()
    output_texts = [
        self.tokenizer.decode(output[: int(length)])
        for output, length in zip(output_list, seq_lengths)
    ]
    order = scores.argsort(descending=True)
    output_texts = [output_texts[i] for i in order]
    return output_texts
