from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Optional

import torch

from sglang.srt.batch_overlap.two_batch_overlap import TboDPAttentionPreparer
from sglang.srt.environ import envs
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.metrics.collector import DPCooperationInfo
from sglang.srt.utils.common import require_mlp_tp_gather

if TYPE_CHECKING:
    from sglang.srt.distributed.parallel_state import GroupCoordinator
    from sglang.srt.managers.scheduler import Scheduler


_ENABLE_METRICS_DP_ATTENTION = envs.SGLANG_ENABLE_METRICS_DP_ATTENTION.get()


@dataclass
class MLPSyncBatchInfo:
    dp_size: int
    tp_size: int
    cp_size: int

    num_tokens: int
    num_tokens_for_logprob: int
    can_cuda_graph: bool
    is_extend_in_batch: bool
    local_can_run_tbo: bool
    local_forward_mode: int

    # some gathered elements
    tp0_info: torch.Tensor = None
    global_num_tokens: list[int] = None
    global_num_tokens_for_logprob: list[int] = None
    tbo_split_seq_index: torch.Tensor = None
    global_forward_mode: int = None
    dp_cooperation_info: Optional[DPCooperationInfo] = None

    def _get_local_tensor(self, device, dtype=torch.int64) -> torch.Tensor:
        return torch.tensor(
            [
                self.num_tokens,
                self.num_tokens_for_logprob,
                int(self.can_cuda_graph),
                int(self.is_extend_in_batch),
                int(self.local_can_run_tbo),
                self.local_forward_mode,
            ],
            device=device,
            dtype=dtype,
        )

    def all_gather(self, device, group: torch.distributed.ProcessGroup):
        local_info_tensor = self._get_local_tensor(device=device)
        global_info_tensor = torch.empty(
            (self.dp_size, self.tp_size * self.cp_size, 6),
            dtype=torch.int64,
            device=device,
        )

        torch.distributed.all_gather_into_tensor(
            global_info_tensor.flatten(),
            local_info_tensor,
            group=group,
        )

        # Debug: print per-(dp,cp,tp) num_tokens when they disagree within the same dp shard.
        # This helps diagnose cases where one (cp,tp) rank is IDLE (num_tokens=0) while others have work.
        # try:
        #     group_rank = torch.distributed.get_rank(group=group)
        # except Exception:
        #     group_rank = -1
        # if group_rank == 0:
        #     for dp_rank in range(self.dp_size):
        #         row = global_info_tensor[dp_rank, :, 0]  # num_tokens
        #         if row.numel() == 0:
        #             continue
        #         if int(row.min().item()) != int(row.max().item()):
        #             world_per_dp = self.tp_size * self.cp_size
        #             vals = row.tolist()
        #             print(
        #                 f"[MLPSyncBatchInfo][debug] num_tokens mismatch within dp={dp_rank}: "
        #                 f"min={min(vals)} max={max(vals)} (tp_size={self.tp_size}, cp_size={self.cp_size})",
        #                 flush=True,
        #             )
        #             for idx, v in enumerate(vals):
        #                 cp_rank = idx // self.tp_size
        #                 tp_rank = idx % self.tp_size
        #                 # Rank-in-this-group is consistent with all_gather ordering.
        #                 rank_in_group = dp_rank * world_per_dp + idx
        #                 print(
        #                     f"[MLPSyncBatchInfo][debug] dp={dp_rank} cp={cp_rank} tp={tp_rank} "
        #                     f"rank_in_group={rank_in_group} num_tokens={v}",
        #                     flush=True,
        #                 )

        # # # Aggregate per-DP info across all (cp,tp) ranks.
        # tp0_info = torch.empty((self.dp_size, 6), dtype=torch.int64, device=device)
        # tp0_info[:, 0] = global_info_tensor[:, :, 0].max(dim=1).values  # num_tokens
        # tp0_info[:, 1] = global_info_tensor[:, :, 1].max(dim=1).values  # num_tokens_for_logprob
        # tp0_info[:, 2] = global_info_tensor[:, :, 2].min(dim=1).values  # can_cuda_graph (AND)
        # tp0_info[:, 3] = global_info_tensor[:, :, 3].max(dim=1).values  # is_extend_in_batch (OR)
        # tp0_info[:, 4] = global_info_tensor[:, :, 4].min(dim=1).values  # local_can_run_tbo (AND)

        # # For forward mode, ignore IDLE ranks; if multiple non-IDLE modes exist within a DP shard,
        # # set to -1 so the global agreement check fails (disables TBO safely).
        # from sglang.srt.model_executor.forward_batch_info import ForwardMode

        # for dp_rank in range(self.dp_size):
        #     modes = global_info_tensor[dp_rank, :, 5]
        #     non_idle = modes[modes != ForwardMode.IDLE.value]
        #     if non_idle.numel() == 0:
        #         tp0_info[dp_rank, 5] = ForwardMode.IDLE.value
        #     elif torch.all(non_idle == non_idle[0]):
        #         tp0_info[dp_rank, 5] = non_idle[0]
        #     else:
        #         tp0_info[dp_rank, 5] = -1

        tp0_info = global_info_tensor[:, 0, :]
        self.tp0_info = tp0_info
        self.global_num_tokens = tp0_info[:, 0].tolist()
        # prit(f"self.global_num_tokens: {self.global_num_tokens}", flush=True)

        self.global_num_tokens_for_logprob = tp0_info[:, 1].tolist()
        self.can_cuda_graph = bool(tp0_info[:, 2].min().item())
        self.is_extend_in_batch = bool(tp0_info[:, 3].max().item())
        if _ENABLE_METRICS_DP_ATTENTION:
            self.dp_cooperation_info = DPCooperationInfo.create(tp0_info[:, 5].tolist())


def _update_gather_batch(
    batch: ScheduleBatch,
    mlp_sync_info: MLPSyncBatchInfo,
    require_mlp_tp_gather: bool,
    skip_all_gather=False,
):
    # TODO: handle the case when moe_dense_tp_size != 1
    if not require_mlp_tp_gather:
        print(f"not require_mlp_tp_gather: {mlp_sync_info.num_tokens}")
        batch.global_num_tokens = [mlp_sync_info.num_tokens]
        batch.global_num_tokens_for_logprob = [mlp_sync_info.num_tokens_for_logprob]
    else:
        print(f"require_mlp_tp_gather: {mlp_sync_info.global_num_tokens}")
        batch.global_num_tokens = mlp_sync_info.global_num_tokens
        batch.global_num_tokens_for_logprob = (
            mlp_sync_info.global_num_tokens_for_logprob
        )
    if not skip_all_gather:
        batch.is_extend_in_batch = mlp_sync_info.is_extend_in_batch
        batch.tbo_split_seq_index = mlp_sync_info.tbo_split_seq_index
        batch.global_forward_mode = mlp_sync_info.global_forward_mode

    # Check forward mode for cuda graph
    batch.can_run_dp_cuda_graph = mlp_sync_info.can_cuda_graph


def prepare_mlp_sync_batch_raw(
    local_batch: ScheduleBatch,
    dp_size: int,
    attn_tp_size: int,
    attn_cp_size: int,
    tp_group: GroupCoordinator,
    get_idle_batch: Callable[[], ScheduleBatch],
    disable_cuda_graph: bool,
    require_mlp_tp_gather: bool,
    disable_overlap_schedule: bool,
    offload_tags: set[str],
):
    # Check if other DP workers have running batches
    if local_batch is None or local_batch.forward_mode.is_prebuilt():
        num_tokens = 0
        num_tokens_for_logprob = 0
    elif local_batch.forward_mode.is_decode():
        num_tokens = local_batch.batch_size()
        num_tokens_for_logprob = num_tokens
    else:
        num_tokens = local_batch.extend_num_tokens
        num_tokens_for_logprob = sum(
            # We should have at least 1 token for sample in every case.
            max(extend_len - logprob_start_len, 1)
            for logprob_start_len, extend_len in zip(
                local_batch.extend_logprob_start_lens,
                local_batch.extend_lens,
            )
        )
        assert (
            local_batch.return_logprob
            or num_tokens_for_logprob == local_batch.batch_size()
        )
    if num_tokens > 0:
        print(f"num_tokens: {num_tokens}, rank: {torch.distributed.get_rank()}")
    # print(f"num_tokens: {num_tokens}, rank: {torch.distributed.get_rank()}")

    skip_all_gather = envs.SGLANG_SCHEDULER_SKIP_ALL_GATHER.get()
    can_cuda_graph = (
        local_batch is None
        or local_batch.forward_mode.is_decode_or_idle()
        or local_batch.forward_mode.is_prebuilt()
    ) and not disable_cuda_graph

    is_extend_in_batch = local_batch.forward_mode.is_extend() if local_batch else False
    if local_batch is not None:
        local_batch.is_extend_in_batch = is_extend_in_batch

    tbo_preparer = TboDPAttentionPreparer()
    if len(offload_tags) == 0 and disable_overlap_schedule:
        group = tp_group.device_group
        device = tp_group.device
    else:
        group = tp_group.cpu_group
        device = "cpu"

    local_can_run_tbo, local_forward_mode = tbo_preparer.prepare_all_gather(local_batch)

    mlp_sync_info = MLPSyncBatchInfo(
        dp_size=dp_size,
        tp_size=attn_tp_size,
        cp_size=attn_cp_size,
        num_tokens=num_tokens,
        num_tokens_for_logprob=num_tokens_for_logprob,
        can_cuda_graph=can_cuda_graph,
        is_extend_in_batch=is_extend_in_batch,
        local_can_run_tbo=local_can_run_tbo,
        local_forward_mode=local_forward_mode,
    )

    if not skip_all_gather:
        mlp_sync_info.all_gather(device=device, group=group)

        mlp_sync_info.tbo_split_seq_index, mlp_sync_info.global_forward_mode = (
            tbo_preparer.compute_output(
                mlp_sync_info.tp0_info[:, 4:6],
            )
        )

    need_idle_batch = skip_all_gather or max(mlp_sync_info.global_num_tokens) > 0
    # print(f"need_idle_batch: {need_idle_batch}, {skip_all_gather}, {max(mlp_sync_info.global_num_tokens)}")
    if need_idle_batch:
        batch_to_gather = local_batch
        if local_batch is None:
            batch_to_gather = local_batch = get_idle_batch()
        elif local_batch.forward_mode.is_prebuilt():
            # NOTE: for prebuilt batch, we add an inner idle batch to run MLP sync
            batch_to_gather = local_batch.inner_idle_batch = get_idle_batch()
        _update_gather_batch(
            batch_to_gather, mlp_sync_info, require_mlp_tp_gather, skip_all_gather
        )

    if _ENABLE_METRICS_DP_ATTENTION and local_batch is not None:
        local_batch.dp_cooperation_info = mlp_sync_info.dp_cooperation_info

    return local_batch


class SchedulerDPAttnMixin:
    def prepare_mlp_sync_batch(self: Scheduler, local_batch: ScheduleBatch):
        return prepare_mlp_sync_batch_raw(
            local_batch,
            dp_size=self.server_args.dp_size,
            attn_tp_size=self.attn_tp_size,
            attn_cp_size=self.attn_cp_size,
            tp_group=self.tp_group,
            get_idle_batch=self.get_idle_batch,
            disable_cuda_graph=self.server_args.disable_cuda_graph,
            require_mlp_tp_gather=require_mlp_tp_gather(self.server_args),
            disable_overlap_schedule=self.server_args.disable_overlap_schedule,
            offload_tags=self.offload_tags,
        )

    def maybe_prepare_mlp_sync_batch_and_log_stats(
        self: Scheduler,
        batch: Optional[ScheduleBatch],
        need_sync: Optional[bool] = None,
        log_stats: bool = True,
    ) -> Optional[ScheduleBatch]:
        """
        Helper to pair log_prefill_stats with log_prefill_stats_late.
        Should be called after get_new_batch_prefill() to ensure proper pairing.

        Args:
            batch: The batch to process
            need_sync: If specified, overrides self.require_mlp_sync for prepare_mlp_sync_batch decision
            log_stats: Whether to call log_prefill_stats_late. Set to False for intermediate calls.
        """
        if need_sync if need_sync is not None else self.require_mlp_sync:
            batch = self.prepare_mlp_sync_batch(batch)
        if log_stats:
            self.log_prefill_stats_late(batch)
        return batch

    def get_idle_batch(self: Scheduler) -> ScheduleBatch:
        idle_batch = ScheduleBatch.init_new(
            [],
            self.req_to_token_pool,
            self.token_to_kv_pool_allocator,
            self.tree_cache,
            self.model_config,
            self.enable_overlap,
            self.spec_algorithm,
        )
        idle_batch.prepare_for_idle()
        return idle_batch
