# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union, final

import torch
from torch import Tensor
from torch.nn.functional import log_softmax

from fairseq2.data import VocabularyInfo
from fairseq2.generation.generator import (
    Hypothesis,
    Seq2SeqGenerator,
    Seq2SeqGeneratorOutput,
    SequenceGenerator,
    SequenceGeneratorOutput,
    StepHook,
)
from fairseq2.generation.step_processor import StepProcessor
from fairseq2.models.decoder import DecoderModel
from fairseq2.models.encoder_decoder import EncoderDecoderModel
from fairseq2.models.sequence import SequenceModelOutput
from fairseq2.nn.incremental_state import IncrementalStateBag
from fairseq2.nn.padding import PaddingMask
from fairseq2.typing import finaloverride, override

import numpy as np
import time


@final
class BeamSearchSequenceGenerator(SequenceGenerator):
    """Represents a sequence generator based on beam search."""

    algorithm: BeamSearchAlgorithm
    beam_size: int
    min_gen_len: int
    max_gen_len: int
    max_seq_len: int
    echo_prompt: bool
    normalize_scores: bool
    temperature: float
    unk_penalty: float
    len_penalty: float
    step_processors: List[StepProcessor]

    def __init__(
        self,
        model: DecoderModel,
        *,
        algorithm: Optional[BeamSearchAlgorithm] = None,
        beam_size: int = 5,
        min_gen_len: int = 1,
        max_gen_len: int = 128,
        max_seq_len: int = 1024,
        echo_prompt: bool = False,
        normalize_scores: bool = True,
        temperature: float = 1.0,
        unk_penalty: float = 0.0,
        len_penalty: float = 1.0,
        step_processors: Optional[Sequence[StepProcessor]] = None,
    ) -> None:
        """
        :param model:
            The decoder model to use for generation.
        :param algorithm:
            The beam search algorithm.
        :param beam_size:
            The beam size.
        :param min_gen_len:
            The minimum allowed generation length.
        :param max_gen_len:
            The maximum allowed generation length.
        :param max_seq_len:
            The maximum allowed sequence length including prompt.
        :param echo_prompt:
            If ``True``, returns generated sequences with prompts appended.
        :param normalize_scores:
            If ``True``, normalizes scores by lengths of generated sequences.
        :param temperature:
            The logit temperature, where values greater than 1.0 produce more
            uniform logits; values less than 1.0 produce sharper logits.
        :param unk_penalty:
            The UNK symbol penalty, where values less than 0 produce more UNKs;
            values greater than 0 produce fewer UNKs.
        :param len_penalty:
            The length penalty, where values less than 1.0 favor shorter
            sequences; values greater than 1.0 favor longer sequences.
        :param step_processors:
            The processors to call at each generation step.
        """
        super().__init__(model)

        if min_gen_len < 1:
            raise ValueError(
                f"`min_gen_len` must be greater than or equal to 1, but is {min_gen_len} instead."
            )

        if max_gen_len < 1:
            raise ValueError(
                f"`max_gen_len` must be greater than or equal to 1, but is {max_gen_len} instead."
            )

        if min_gen_len > max_gen_len:
            raise ValueError(
                f"`min_gen_len` must be less than or equal to `max_gen_len` ({max_gen_len}), but is {min_gen_len} instead."
            )

        self.algorithm = algorithm or StandardBeamSearchAlgorithm()
        self.beam_size = beam_size
        self.min_gen_len = min_gen_len
        self.max_gen_len = max_gen_len
        self.max_seq_len = max_seq_len
        self.echo_prompt = echo_prompt
        self.normalize_scores = normalize_scores
        self.temperature = temperature
        self.unk_penalty = unk_penalty
        self.len_penalty = len_penalty

        if step_processors:
            self.step_processors = list(step_processors)
        else:
            self.step_processors = []

    @finaloverride
    @torch.inference_mode()
    def __call__(
        self, prompt_seqs: Tensor, prompt_padding_mask: Optional[PaddingMask]
    ) -> SequenceGeneratorOutput:
        op = _BeamSearchSequenceGeneratorOp(
            # self.model,
            prompt_seqs,
            prompt_padding_mask,
            self.algorithm,
            self.beam_size,
            self.min_gen_len,
            self.max_gen_len,
            self.max_seq_len,
            self.echo_prompt,
            self.normalize_scores,
            self.temperature,
            self.unk_penalty,
            self.len_penalty,
            self.step_processors,
            self._step_hooks,
        )

        hypotheses = op()

        return SequenceGeneratorOutput(hypotheses)


@final
class BeamSearchSeq2SeqGenerator(Seq2SeqGenerator):
    """Represents a sequence-to-sequence generator based on beam search."""

    algorithm: BeamSearchAlgorithm
    beam_size: int
    min_gen_len: int
    max_gen_len: Tuple[int, int]
    max_seq_len: int
    echo_prompt: bool
    normalize_scores: bool
    temperature: float
    unk_penalty: float
    len_penalty: float
    step_processors: List[StepProcessor]

    def __init__(
        self,
        model: EncoderDecoderModel,
        *,
        algorithm: Optional[BeamSearchAlgorithm] = None,
        beam_size: int = 5,
        min_gen_len: int = 1,
        max_gen_len: Tuple[int, int] = (1, 128),
        max_seq_len: int = 1024,
        echo_prompt: bool = False,
        normalize_scores: bool = True,
        temperature: float = 1.0,
        unk_penalty: float = 0.0,
        len_penalty: float = 1.0,
        step_processors: Optional[Sequence[StepProcessor]] = None,
    ) -> None:
        """
        :param model:
            The encoder-decoder model to use for generation.
        :param algorithm:
            The beam search algorithm.
        :param beam_size:
            The beam size.
        :param min_gen_len:
            The minimum allowed generation length.
        :param max_gen_len:
            The maximum allowed generation length as ``ax + b``, where ``x`` is
            the source sequence length.
        :param max_seq_len:
            The maximum allowed sequence length including prompt.
        :param echo_prompt:
            If ``True``, returns generated sequences with prompts appended.
        :param normalize_scores:
            If ``True``, normalizes scores by lengths of generated sequences.
        :param temperature:
            The logit temperature, where values greater than 1.0 produce more
            uniform logits; values less than 1.0 produce sharper logits.
        :param unk_penalty:
            The UNK symbol penalty, where values less than 0 produce more UNKs;
            values greater than 0 produce fewer UNKs.
        :param len_penalty:
            The length penalty, where values less than 1.0 favor shorter
            sequences; values greater than 1.0 favor longer sequences.
        :param step_processors:
            The processors to call at each generation step.
        """
        super().__init__(model)

        if min_gen_len < 1:
            raise ValueError(
                f"`min_gen_len` must be greater than or equal to 1, but is {min_gen_len} instead."
            )

        self.algorithm = algorithm or StandardBeamSearchAlgorithm()
        self.beam_size = beam_size
        self.min_gen_len = min_gen_len
        self.max_gen_len = max_gen_len
        self.max_seq_len = max_seq_len
        self.echo_prompt = echo_prompt
        self.normalize_scores = normalize_scores
        self.temperature = temperature
        self.unk_penalty = unk_penalty
        self.len_penalty = len_penalty

        if step_processors:
            self.step_processors = list(step_processors)
        else:
            self.step_processors = []

    @finaloverride
    @torch.inference_mode()
    def __call__(
        self,
        source_seqs: Tensor,
        source_padding_mask: Optional[PaddingMask],
        prompt_seqs: Tensor,
        prompt_padding_mask: Optional[PaddingMask],
        compiled_text_decoder: Optional[list] = None,
        model = None
    ) -> Seq2SeqGeneratorOutput:
        seq_len = dict()
        timer_result = dict()
        seq_len["Encoder"] = source_seqs.shape[1]

        # (P, S)
        torch.cuda.synchronize()
        start_time = time.time()
        encoder_output, encoder_padding_mask = model.encode(
            source_seqs, source_padding_mask
        )
        torch.cuda.synchronize()
        timer_result["Encoder"] = (time.time()-start_time)*1000

        if source_padding_mask is None:
            max_source_len = source_seqs.size(1)
        else:
            max_source_len = int(source_padding_mask.seq_lens.max())

        a_term, b_term = self.max_gen_len

        # In seq2seq generation, the maximum generation length is relative to
        # the source sequence length.
        max_gen_len = int(a_term * max_source_len + b_term)

        if max_gen_len < 1:
            raise ValueError(
                f"`max_gen_len` must be greater than or equal to 1, but is {max_gen_len} instead. Adjust your `max_gen_len` argument."
            )

        if self.min_gen_len > max_gen_len:
            raise ValueError(
                f"`min_gen_len` must be less than or equal to `max_gen_len` ({max_gen_len}), but is {self.min_gen_len} instead. Adjust your `max_gen_len` argument."
            )


        torch.cuda.synchronize()
        start_time = time.time()
        op = _BeamSearchSeq2SeqGeneratorOp(
            # self.model,
            model,
            encoder_output,
            encoder_padding_mask,
            prompt_seqs,
            prompt_padding_mask,
            self.algorithm,
            self.beam_size,
            self.min_gen_len,
            max_gen_len,
            self.max_seq_len,
            self.echo_prompt,
            self.normalize_scores,
            self.temperature,
            self.unk_penalty,
            self.len_penalty,
            self.step_processors,
            self._step_hooks,
        )
        # hypotheses = op()
        for layer in model.decoder.layers.drop_iter():
            if compiled_text_decoder[0] is None:
                # 1024 is hard-coded as the maximum sequence length for self-attention layers for the optimal performance. The number could be changed accordingly.
                layer.self_attn.cache_k = torch.zeros((self.beam_size, layer.self_attn.num_heads, 1024, layer.self_attn.head_dim), dtype=torch.half).cuda()
                layer.self_attn.cache_v = torch.zeros((self.beam_size, layer.self_attn.num_heads, 1024, layer.self_attn.head_dim), dtype=torch.half).cuda()
                # 256 is hard-coded as the maximum sequence length for cross-attention layers for the optimal performance. The number could be changed accordingly.
                layer.encoder_decoder_attn.cache_k = torch.zeros((self.beam_size, layer.encoder_decoder_attn.num_heads, 256, layer.encoder_decoder_attn.head_dim), dtype=torch.half).cuda()
                layer.encoder_decoder_attn.cache_v = torch.zeros((self.beam_size, layer.encoder_decoder_attn.num_heads, 256, layer.encoder_decoder_attn.head_dim), dtype=torch.half).cuda()
            layer.self_attn.kv_cache = False
            layer.encoder_decoder_attn.kv_cache = False

        hypotheses = op(compiled_text_decoder, model)

        torch.cuda.synchronize()
        timer_result["Decoder"] = (time.time()-start_time)*1000

        seq_len["Decoder"] = op.min_prompt_len-1
        return Seq2SeqGeneratorOutput(hypotheses, encoder_output, encoder_padding_mask), timer_result


class BeamSearchAlgorithm(ABC):
    """Represents a beam search algorithm."""

    @abstractmethod
    def __call__(self, beam_size: int, lprobs: Tensor, step_scores: Tensor) -> BeamStep:
        """Take a single step.

        A subclass implementation is expected to return the best 2 x `beam_size`
        candidates. The sequence generator will choose the first `beam_size` of
        these which don't predict EOS to continue with.

        :param beam_size:
            The beam size.
        :param lprobs:
            The next-step log probability of each vocabulary entry. *Shape:*
            :math:`(N,V)`, where :math:`N` is the batch size and :math:`V` is
            the size of the vocabulary.
        :param step_scores:
            The cumulative score of each step in the beam. *Shape:* :math:`(N,S)`,
            where :math:`N` is the batch size and :math:`S` is the length of the
            beam.
        """


@dataclass
class BeamStep:
    """Represents the output of a beam search algorithm."""

    seq_indices: Tensor
    """The beam sequence indices. *Shape:* :math:`(B)`, where :math:`B` is the
    beam size."""

    vocab_indices: Tensor
    """The vocabulary indices. *Shape:* Same as ``seq_indices``."""

    scores: Tensor
    """The scores. *Shape:* Same as ``seq_indices``."""

    def masked_select(self, mask: Tensor) -> BeamStep:
        """Reduce the beam to the sequences included in ``mask``."""
        seq_indices = self.seq_indices.masked_select(mask)

        vocab_indices = self.vocab_indices.masked_select(mask)

        scores = self.scores.masked_select(mask)

        return BeamStep(seq_indices, vocab_indices, scores)

    def first(self, count: int) -> BeamStep:
        """Slice the beam to the first ``count`` sequences."""
        seq_indices = self.seq_indices[:count]

        vocab_indices = self.vocab_indices[:count]

        scores = self.scores[:count]

        return BeamStep(seq_indices, vocab_indices, scores)

    @staticmethod
    def merge(steps: Sequence[BeamStep]) -> BeamStep:
        """Merge ``steps`` into a single beam."""
        seq_indices = torch.cat([s.seq_indices for s in steps])

        vocab_indices = torch.cat([s.vocab_indices for s in steps])

        scores = torch.cat([s.scores for s in steps])

        return BeamStep(seq_indices, vocab_indices, scores)


@final
class StandardBeamSearchAlgorithm(BeamSearchAlgorithm):
    """Represents a standard beam search algoritm."""

    @finaloverride
    def __call__(self, beam_size: int, lprobs: Tensor, step_scores: Tensor) -> BeamStep:
        vocab_size = lprobs.size(1)

        # Make the probabilities contain cumulative scores for each hypothesis.
        # (N, V) + (N, 1) = (N, V)
        lprobs = lprobs + step_scores[:, -1].unsqueeze(-1)

        # (N, V) -> (N x V)
        lprobs = lprobs.view(-1)

        # (2 x B)
        top_scores, top_indices = torch.topk(lprobs, k=min(2 * beam_size, vocab_size))

        return BeamStep(top_indices // vocab_size, top_indices % vocab_size, top_scores)


class _BeamSearchSequenceGeneratorOpBase(ABC):
    algorithm: BeamSearchAlgorithm
    eos_idx: int
    pad_idx: Optional[int]
    unk_idx: Optional[int]
    beam_size: int
    min_prompt_len: int
    max_prompt_len: int
    min_seq_len: int
    max_seq_len: int
    echo_prompt: bool
    normalize_scores: bool
    temperature: float
    unk_penalty: float
    len_penalty: float
    step_processors: Sequence[StepProcessor]
    step_nr: int
    state_bag: IncrementalStateBag
    prompt_lens: Optional[Tensor]
    prompt_mask: Optional[Tensor]
    beam_sizes: List[int]
    prompt_indices: Tensor
    seqs: Tensor
    step_scores: Tensor
    output: List[List[Hypothesis]]
    step_hooks: Dict[int, StepHook]

    def __init__(
        self,
        prompt_seqs: Tensor,
        prompt_padding_mask: Optional[PaddingMask],
        algorithm: BeamSearchAlgorithm,
        vocab_info: VocabularyInfo,
        beam_size: int,
        min_gen_len: int,
        max_gen_len: int,
        max_seq_len: int,
        echo_prompt: bool,
        normalize_scores: bool,
        temperature: float,
        unk_penalty: float,
        len_penalty: float,
        step_processors: Sequence[StepProcessor],
        step_hooks: Dict[int, StepHook],
    ) -> None:
        self.algorithm = algorithm

        assert vocab_info.eos_idx is not None

        self.eos_idx = vocab_info.eos_idx
        self.pad_idx = vocab_info.pad_idx
        self.unk_idx = vocab_info.unk_idx

        self.beam_size = beam_size

        min_prompt_idx: Union[int, Tensor]
        max_prompt_idx: Union[int, Tensor]

        if prompt_padding_mask is None:
            self.min_prompt_len, min_prompt_idx = prompt_seqs.size(1), 0
            self.max_prompt_len, max_prompt_idx = prompt_seqs.size(1), 0
        else:
            prompt_seq_lens = prompt_padding_mask.seq_lens

            min_prompt_len, min_prompt_idx = torch.min(prompt_seq_lens, dim=0)
            max_prompt_len, max_prompt_idx = torch.max(prompt_seq_lens, dim=0)

            self.min_prompt_len = int(min_prompt_len)
            self.max_prompt_len = int(max_prompt_len)

            if self.min_prompt_len == self.max_prompt_len:
                prompt_padding_mask = None

        if self.min_prompt_len < 1:
            raise ValueError(f"`prompt_seqs[{int(min_prompt_idx)}]` must not be empty.")

        if self.max_prompt_len >= max_seq_len:
            raise ValueError(
                f"The length of `prompt_seqs[{int(max_prompt_idx)}]` must be less than `max_seq_len` ({max_seq_len}), but is {self.max_prompt_len} instead."
            )

        self.min_seq_len = min(max_seq_len, self.max_prompt_len + min_gen_len)
        self.max_seq_len = min(max_seq_len, self.max_prompt_len + max_gen_len)

        self.echo_prompt = echo_prompt
        self.normalize_scores = normalize_scores
        self.temperature = temperature
        self.unk_penalty = unk_penalty
        self.len_penalty = len_penalty
        self.step_processors = step_processors
        self.step_hooks = step_hooks

        self.step_nr = 0

        self.state_bag = IncrementalStateBag(self.max_seq_len)

        if prompt_padding_mask is None:
            self.prompt_lens = None
            self.prompt_mask = None
        else:
            # (P)
            self.prompt_lens = prompt_padding_mask.seq_lens

            # (P, S_prm)
            self.prompt_mask = prompt_padding_mask.materialize()

        device = prompt_seqs.device

        num_prompts = prompt_seqs.size(0)

        # Holds the sizes of the beams.
        self.beam_sizes = [1 for _ in range(num_prompts)]

        # Holds the prompt indices of the generated sequences.
        # (P)
        self.prompt_indices = torch.arange(num_prompts, device=device)

        # Holds the generated sequences.
        # (P, S)
        self.seqs = torch.empty(
            (num_prompts, self.max_seq_len), device=device, dtype=torch.int64
        )

        # Holds the step scores of the generated sequences.
        # (P, S)
        self.step_scores = torch.zeros(
            (num_prompts, self.max_seq_len), device=device, dtype=torch.float32
        )

        # Bootstrap the sequences.
        self.seqs[:, : self.max_prompt_len] = prompt_seqs[:, : self.max_prompt_len]

        # Holds the sequences that have reached EOS.
        self.output = [[] for _ in range(num_prompts)]

    def params_for_incremental_gen(self, prev_pos : int, cur_pos : int, device : torch.device):
        valid_seq_pos = torch.arange(prev_pos, cur_pos, device=device)

        # 1024 is hard-coded as the maximum sequence length for the optimal performance. The number could be changed accordingly.
        mask = torch.full(
            (1, 1, 1, 1024), False, device=device
        )
        mask[:, :, :, :valid_seq_pos.item() + 1] = True
        return mask, valid_seq_pos

    def __call__(self, compiled_text_decoder = None, model = None) -> List[List[Hypothesis]]:
        if compiled_text_decoder[0] is None:
            # compiled_text_decoder[0] = torch.compile(model.decoder.forward, mode='max-autotune', fullgraph=True)
            compiled_text_decoder[0] = model.decoder.forward

        self._prepare_state(model, compiled_text_decoder[0])
        prev_pos = self.min_prompt_len-1

        for self.step_nr in range(self.min_prompt_len, self.max_seq_len):
            # output = self._step()
            # if not output:
            #     break

            cuda_graph_mask, valid_seq_pos = self.params_for_incremental_gen(
                prev_pos, self.step_nr, self.seqs.device)

            if compiled_text_decoder[1] is None and self.step_nr > self.min_prompt_len:
                compiled_text_decoder[1] = torch.compile(model.decoder.forward2, mode='max-autotune', fullgraph=True)
                # compiled_text_decoder[1] = model.decoder.forward2

            for layer in model.decoder.layers.drop_iter():
                layer.self_attn.self_attn_mask.copy_(cuda_graph_mask)


            output = self._step(cuda_graph_mask, valid_seq_pos, compiled_text_decoder[0] if self.step_nr==self.min_prompt_len else compiled_text_decoder[1], model)
            prev_pos = self.step_nr
            if not output:
                break

        # Sort the hypotheses by their scores before returning.
        for hypotheses in self.output:
            hypotheses.sort(key=lambda h: h.score, reverse=True)  # type: ignore[arg-type, return-value]

        return self.output

    def _prepare_state(self, model, cuda_graph = None) -> None:
        # Fast-forward to the first step that needs to be generated.
        if self.min_prompt_len > 1:
            self._prefill(model, cuda_graph=cuda_graph)
    
    def _prefill(self, model, cuda_graph=None) -> None:
        prefill_len = self.min_prompt_len


        # 1024 is hard-coded as the maximum sequence length for the optimal performance. The number could be changed accordingly.
        mask = torch.full(
            (1, 1, 1, 1024), False, device=self.seqs.device
        )
        mask[:, :, :, :prefill_len - 1] = True

        for layer in model.decoder.layers.drop_iter():
            layer.self_attn.self_attn_mask.copy_(mask)

        valid_seq_pos = torch.arange(0, prefill_len - 1, device=self.seqs.device)

        model_output = self._decode(self.seqs[:, 0:prefill_len - 1], mask, valid_seq_pos, cuda_graph, model)
        
        # model_output = self._decode(self.seqs[:, : prefill_len - 1])

        self.state_bag.increment_step_nr(prefill_len - 1)

        logits = model_output.logits

        if self.temperature != 1.0:
            logits /= self.temperature

        # (P, S_prm - 1, V)
        lprobs = log_softmax(logits, dim=-1, dtype=torch.float32)

        # Fetch the scores of the next prompt step.
        # (P, S_prm - 1, 1)
        prompt_scores = torch.gather(
            lprobs, dim=-1, index=self.seqs[:, 1:prefill_len].unsqueeze(-1)
        )

        # (P, S_prm - 1, 1) -> (P, S_prm - 1)
        prompt_scores.squeeze_(-1).cumsum_(dim=-1)

        # Bootstrap the step scores.
        # (P x B, S_prm - 1)
        self.step_scores[:, 1:prefill_len] = prompt_scores

        if self.step_hooks:
            seqs = self.seqs[:, :prefill_len]

            step_scores = self.step_scores[:, :prefill_len]

            for hook in self.step_hooks.values():
                hook(self.prompt_indices, seqs, step_scores, prefill=True)
    
    def _step(self, cuda_graph_mask, valid_seq_pos, cuda_graph, model) -> bool:
        # Generate the next step output.
        model_output = self._decode(self.seqs[:, self.step_nr - 1 : self.step_nr], cuda_graph_mask, valid_seq_pos, cuda_graph, model)

        self.state_bag.increment_step_nr()

        logits = model_output.logits

        if self.temperature != 1.0:
            logits /= self.temperature

        # (N, 1, V)
        lprobs = log_softmax(logits, dim=-1, dtype=torch.float32)

        # (N, 1, V) -> (N, V)
        lprobs.squeeze_(1)

        # If we are generating the last possible step, force it to be EOS
        # regardless of its score.
        if self.step_nr == self.max_seq_len - 1:
            # fmt: off
            lprobs[:, : self.eos_idx]       = -torch.inf
            lprobs[:,   self.eos_idx + 1 :] = -torch.inf
            # fmt: on
        else:
            # Process `lprobs` in-place if requested.
            for processor in self.step_processors:
                processor(self.seqs[:, : self.step_nr], lprobs, lprob=True)

            # Apply UNK penalty.
            if self.unk_idx is not None:
                lprobs[:, self.unk_idx] -= self.unk_penalty

            # Never allow PAD.
            if self.pad_idx is not None:
                lprobs[:, self.pad_idx] = -torch.inf

            # Do not allow EOS till we reach the minimum sequence length.
            if self.step_nr < self.min_seq_len - 1:
                lprobs[:, self.eos_idx] = -torch.inf

        batch_offset = 0

        new_beam_sizes: List[int] = []

        beam_next_step_list: List[BeamStep] = []

        # We split the batch by `beam_sizes` and treat each beam separately.
        for beam_idx, (beam_lprobs, beam_step_scores) in enumerate(
            zip(lprobs.split(self.beam_sizes), self.step_scores.split(self.beam_sizes))
        ):
            beam_next_step = self._search_beam(
                beam_idx, batch_offset, beam_lprobs, beam_step_scores
            )

            # Bump the beam batch offset to the next beam.
            batch_offset += self.beam_sizes[beam_idx]

            # Check if the beam is terminated.
            if beam_next_step is None:
                continue

            beam_size = len(beam_next_step.seq_indices)

            # We should have terminated the beam if there are no sequences.
            assert beam_size > 0

            new_beam_sizes.append(beam_size)

            beam_next_step_list.append(beam_next_step)

        # No beam left, we can return.
        if len(new_beam_sizes) == 0:
            return False

        self.beam_sizes = new_beam_sizes

        # (N_new)
        next_step = BeamStep.merge(beam_next_step_list)

        self._reorder_state(next_step.seq_indices, model)

        # Record the current step.
        self.seqs[:, self.step_nr] = next_step.vocab_indices

        # Record the scores of the current step.
        self.step_scores[:, self.step_nr] = next_step.scores

        if self.step_hooks:
            seqs = self.seqs[:, : self.step_nr + 1]

            step_scores = self.step_scores[:, : self.step_nr + 1]

            for hook in self.step_hooks.values():
                hook(self.prompt_indices, seqs, step_scores, prefill=False)

        return True

    def _search_beam(
        self, beam_idx: int, batch_offset: int, lprobs: Tensor, step_scores: Tensor
    ) -> Optional[BeamStep]:
        # Ignore the generated indices for the prompt sequences.
        if self.step_nr < self.max_prompt_len:
            assert self.prompt_mask is not None

            # Check if the current beam is in a prompt sequence.
            if self.prompt_mask[batch_offset, self.step_nr]:
                # The size of a beam in a prompt sequence must be always 1.
                assert len(lprobs) == 1

                seq_index = torch.tensor([batch_offset], device=lprobs.device)

                # We just extract the prompt step along with its score and treat
                # it as the next beam step. So we keep a beam of size 1 until we
                # reach the end of the prompt.
                vocab_index = self.seqs[batch_offset, self.step_nr : self.step_nr + 1]

                score = step_scores[0, self.step_nr - 1] + lprobs[0, vocab_index]

                return BeamStep(seq_index, vocab_index, score)
        else:
            self.prompt_mask = None  # Not needed anymore, release.

        # We use the same beam search method as in fairseq, where we take the
        # best 2 x `beam_size` candidates and choose the first `beam_size` of
        # these which don't predict EOS to continue with.
        # (2 x B)
        next_step = self.algorithm(
            self.beam_size, lprobs, step_scores[:, : self.step_nr]
        )

        # Translate the sequence indices from beam to batch.
        next_step.seq_indices += batch_offset

        # (2 x B)
        eos_mask = next_step.vocab_indices == self.eos_idx

        # Consider EOS only when it's among the top `beam_size` indices.
        # (F)
        eos_seq_indices = next_step.seq_indices[: self.beam_size].masked_select(
            eos_mask[: self.beam_size]
        )

        # If one or more sequences have reached EOS, move them to the output.
        if len(eos_seq_indices) > 0:
            # (F)
            eos_scores = next_step.scores[: self.beam_size].masked_select(
                eos_mask[: self.beam_size]
            )

            for seq_idx, score in zip(eos_seq_indices, eos_scores):
                # If `True`, it means we have found `beam_size` hypotheses for
                # this beam.
                if self._finish_sequence(int(seq_idx), score):
                    return None

            # Filter out the sequences that have reached EOS.
            seq_mask = ~eos_mask

            next_step = next_step.masked_select(seq_mask)

        # We can have at most `beam_size` sequences in the beam.
        return next_step.first(self.beam_size)

    @abstractmethod
    def _decode(self, seqs: Tensor, cuda_graph_mask: Tensor, valid_seq_pos: Tensor, cuda_graph, model) -> SequenceModelOutput:
        ...

    def _finish_sequence(self, seq_idx: int, score: Tensor) -> bool:
        self.seqs[seq_idx, self.step_nr] = self.eos_idx

        self.step_scores[seq_idx, self.step_nr] = score

        if self.echo_prompt:
            start_step = 0
        else:
            if self.prompt_lens is None:
                start_step = self.max_prompt_len
            else:
                start_step = int(self.prompt_lens[seq_idx])

        seq_len = self.step_nr + 1

        # (S_out)
        seq = self.seqs[seq_idx, start_step:seq_len]

        # Do not keep `seqs` in memory.
        seq = seq.clone()

        # (S_out)
        step_scores = self.step_scores[seq_idx, start_step:seq_len]

        # Similar to `seqs`, do not keep `step_scores` in memory.
        step_scores = step_scores.clone()

        # Convert from cumulative to per-step scores.
        step_scores[1:] = step_scores[1:] - step_scores[:-1]

        if self.normalize_scores:
            # Since the first step's score is always 0, do not include it in
            # the normalization.
            score /= (seq_len - 1) ** self.len_penalty

        prompt_idx = int(self.prompt_indices[seq_idx])

        hypotheses = self.output[prompt_idx]

        hypotheses.append(Hypothesis(seq, score, step_scores))

        # If we have `beam_size` hypotheses for the prompt, we can remove the
        # beam.
        return len(hypotheses) == self.beam_size

    def _reorder_state(self, new_order: Tensor, model=None) -> None:
        # self.state_bag.reorder(new_order)

        cache_ks = []
        cache_vs = []
        for layer in model.decoder.layers.drop_iter():
            cache_ks.append(layer.self_attn.cache_k)
            cache_vs.append(layer.self_attn.cache_v)
            cache_ks.append(layer.encoder_decoder_attn.cache_k)
            cache_vs.append(layer.encoder_decoder_attn.cache_v)

        @torch.compile(mode='max-autotune-no-cudagraphs')
        def reorder(k, new_order):
            for i in range(len(k)):
                k[i].copy_(k[i].index_select(0, new_order))

        reorder(cache_ks, new_order)
        reorder(cache_vs, new_order)

        # (N) -> (N - F)
        if self.prompt_lens is not None:
            self.prompt_lens = self.prompt_lens.index_select(dim=0, index=new_order)

        # (N, S_prm) -> (N - F, S_prm)
        if self.prompt_mask is not None:
            self.prompt_mask = self.prompt_mask.index_select(dim=0, index=new_order)

        # (N) -> (N - F)
        self.prompt_indices = self.prompt_indices.index_select(dim=0, index=new_order)

        # (N, S) -> (N - F, S)
        self.seqs = self.seqs.index_select(dim=0, index=new_order)

        # (N, S) -> (N - F, S)
        self.step_scores = self.step_scores.index_select(dim=0, index=new_order)


class _BeamSearchSequenceGeneratorOp(_BeamSearchSequenceGeneratorOpBase):
    model: DecoderModel

    def __init__(
        self,
        model: DecoderModel,
        prompt_seqs: Tensor,
        prompt_padding_mask: Optional[PaddingMask],
        algorithm: BeamSearchAlgorithm,
        beam_size: int,
        min_gen_len: int,
        max_gen_len: int,
        max_seq_len: int,
        echo_prompt: bool,
        normalize_scores: bool,
        temperature: float,
        unk_penalty: float,
        len_penalty: float,
        step_processors: Sequence[StepProcessor],
        step_hooks: Dict[int, StepHook],
    ) -> None:
        super().__init__(
            prompt_seqs,
            prompt_padding_mask,
            algorithm,
            model.vocab_info,
            beam_size,
            min_gen_len,
            max_gen_len,
            max_seq_len,
            echo_prompt,
            normalize_scores,
            temperature,
            unk_penalty,
            len_penalty,
            step_processors,
            step_hooks,
        )

        # self.model = model

    @override
    def _decode(self, seqs: Tensor, cuda_graph_mask: Tensor, valid_seq_pos: Tensor, cuda_graph, model) -> SequenceModelOutput:
        decoder_output, decoder_padding_mask = model.decode(
            seqs,
            None,  # We never use PAD in incremental decoding.
            state_bag=self.state_bag,
        )

        return model.project(decoder_output, decoder_padding_mask)


class _BeamSearchSeq2SeqGeneratorOp(_BeamSearchSequenceGeneratorOpBase):
    model: EncoderDecoderModel
    encoder_output: Tensor
    encoder_padding_mask: Optional[PaddingMask]

    def __init__(
        self,
        model: EncoderDecoderModel,
        encoder_output: Tensor,
        encoder_padding_mask: Optional[PaddingMask],
        prompt_seqs: Tensor,
        prompt_padding_mask: Optional[PaddingMask],
        algorithm: BeamSearchAlgorithm,
        beam_size: int,
        min_gen_len: int,
        max_gen_len: int,
        max_seq_len: int,
        echo_prompt: bool,
        normalize_scores: bool,
        temperature: float,
        unk_penalty: float,
        len_penalty: float,
        step_processors: Sequence[StepProcessor],
        step_hooks: Dict[int, StepHook],
    ) -> None:
        super().__init__(
            prompt_seqs,
            prompt_padding_mask,
            algorithm,
            model.target_vocab_info,
            beam_size,
            min_gen_len,
            max_gen_len,
            max_seq_len,
            echo_prompt,
            normalize_scores,
            temperature,
            unk_penalty,
            len_penalty,
            step_processors,
            step_hooks,
        )

        # self.model = model
        # self.encoder_output = encoder_output
        # self.encoder_padding_mask = encoder_padding_mask

        # 256 is hard-coded as the maximum sequence length for cross-attention layers for the optimal performance. The number could be changed accordingly.
        self.encoder_output = torch.cat((encoder_output, torch.zeros((encoder_output.shape[0], 256-encoder_output.shape[1], encoder_output.shape[2]), device=encoder_output.device, dtype=encoder_output.dtype)), 1)
        self.encoder_padding_mask = PaddingMask(torch.tensor([encoder_output.shape[1]], device=encoder_output.device), batch_seq_len=256)

    @override
    def _decode(self, seqs: Tensor, cuda_graph_mask: Tensor, valid_seq_pos: Tensor, cuda_graph = None, model = None) -> SequenceModelOutput:
        decoder_output, decoder_padding_mask = model.decode(
            seqs,
            None,  # We never use PAD in incremental decoding.
            self.encoder_output,
            self.encoder_padding_mask.materialize(),
            state_bag=self.state_bag,
            cuda_graph_mask=cuda_graph_mask,
            valid_seq_pos=valid_seq_pos,
            compiled_decoder=cuda_graph,
        )

        return model.project(decoder_output, decoder_padding_mask)

    @override
    def _reorder_state(self, new_order: Tensor, model=None) -> None:
        super()._reorder_state(new_order, model=model)

        self.encoder_output = self.encoder_output.index_select(dim=0, index=new_order)

        if self.encoder_padding_mask is not None:
            encoder_seq_lens = self.encoder_padding_mask.seq_lens

            # (N) -> (N - F)
            encoder_seq_lens = encoder_seq_lens.index_select(dim=0, index=new_order)

            self.encoder_padding_mask = PaddingMask(
                encoder_seq_lens, batch_seq_len=self.encoder_output.size(1)
            )
