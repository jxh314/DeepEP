import os
import time
import torch
import torch.distributed as dist

# noinspection PyUnresolvedReferences
import deep_ep
from utils import init_dist, bench, calc_diff, create_grouped_scores, inplace_unique, per_token_cast_to_fp8, per_token_cast_back

# Test compatibility with low latency functions
import test_low_latency


def test_main(num_sms: int, local_rank: int, num_local_ranks: int, num_ranks: int, 
              num_nodes: int, rank: int, buffer: deep_ep.Buffer, group: dist.ProcessGroup):
    # Settings
    num_tokens, hidden, num_topk_groups, num_topk, num_experts = 4096, 7168, min(num_nodes, 4), 8, (256 // num_ranks) * num_ranks
    assert num_experts % num_ranks == 0 and num_local_ranks == 8
    if local_rank == 0:
        print(f'[config] num_tokens={num_tokens}, hidden={hidden}, num_topk_groups={num_topk_groups}, num_topk={num_topk}', flush=True)

    # Random data
    x = torch.ones((num_tokens, hidden), dtype=torch.bfloat16, device='cuda') * rank
    x_pure_rand = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device='cuda')
    x_e4m3 = per_token_cast_to_fp8(x)
    x_e4m3 = (x_e4m3[0], x_e4m3[1].T.contiguous().T)
    scores = torch.randn((num_tokens, num_experts), dtype=torch.float32, device='cuda').abs() + 1

    # 分组和筛选
    group_scores = scores.view(num_tokens, num_nodes, -1).amax(dim=-1)
    group_idx = torch.topk(group_scores, k=num_topk_groups, dim=-1, sorted=False).indices
    masked_scores = create_grouped_scores(scores, group_idx, num_nodes)
    # masked_scores是经过分组和筛选后的分数矩阵。
    # torch.topk 返回每个 token 的 top-k 候选项索引，num_topk 决定了候选项的数量。
    # topk_idx是一个形状为 (num_tokens, num_topk) 的张量，表示每个 token 对应的 top-k 专家索引
    #   用于计算每个 token 的分发目标（如 RDMA 索引和 rank 索引）。
    topk_idx = torch.topk(masked_scores, num_topk, dim=-1, largest=True, sorted=False)[1]
    # topk_weights 是与 topk_idx 对应的权重，用于后续的通信和计算。
    topk_weights = torch.ones((num_tokens, num_topk), dtype=torch.float32, device='cuda') * rank
    topk_weights_pure_rand = torch.randn((num_tokens, num_topk), dtype=torch.float32, device='cuda')
    # top-k 个expert所在的rank
    rank_idx = topk_idx // (num_experts // num_ranks)
    rank_idx.masked_fill_(topk_idx == -1, -1)
    inplace_unique(rank_idx, num_ranks)# 去重和范围限制
    # 即node idx，token 的目标节点
    rdma_rank_idx = rank_idx // num_local_ranks
    rdma_rank_idx.masked_fill_(rank_idx == -1, -1)
    inplace_unique(rdma_rank_idx, num_nodes)

    # RDMA dispatch counts
    rdma_idx = topk_idx // (num_experts // num_nodes)
    rdma_idx.masked_fill_(topk_idx == -1, -1)# 将无效的topk_idx（值为-1）对应的rdma_idx 设置为-1，表示这些 token 不参与 RDMA 分发。
    inplace_unique(rdma_idx, num_nodes)
    # rdma_idx 中有效索引的数量。(num_tokens, num_topk)中有non负数的元素个数
    num_rdma_token_sent = rdma_idx.ne(-1).sum().item()

    # Expert meta  ——专家元数据统计
    num_tokens_per_expert = torch.zeros((num_experts, ), dtype=torch.int, device='cuda')  # 初始化每个专家的 token 数量
    for i in range(num_experts):
         # 统计每个专家接收到的 token 数量，这里只是当前rank 发送给的专家，后面做all_reduce
        num_tokens_per_expert[i] = (topk_idx == i).sum() 
    gbl_num_tokens_per_expert = num_tokens_per_expert.clone()  # 克隆本地统计结果
    # 得到所有进程中每个专家接收到的 token 总数
    dist.all_reduce(gbl_num_tokens_per_expert, group=group) 

    # Rank layout meta——rank布局元数据统计
    num_tokens_per_rank = torch.empty((num_ranks, ), dtype=torch.int, device='cuda') 
    num_tokens_per_rdma_rank = torch.empty((num_nodes, ), dtype=torch.int, device='cuda') 
    token_idx_in_rank = torch.full((num_ranks, num_tokens), -1, dtype=torch.long, device='cuda')  # 初始化 token 索引
    for i in range(num_ranks):
        num_tokens_per_rank[i] = (rank_idx == i).sum()  # 统计每个 rank 接收到的 token 数量
        # token_sel 是一维，形状为(num_tokens,)每个元素表示对应的 token 是否被分配到当前 rank i：
        token_sel = (rank_idx == i).max(dim=-1)[0] # 确定每个 token 是否被分配到当前 rank
        count = token_sel.sum().item()  # 统计被分配到rank i 的 token 数量
        # 返回【1】表示排序后的原始索引，假设token_sel=[1,0,1,0]，那么tokens=[0,2,1,3]，tokens[:count]=[0,2]
        tokens = torch.sort(token_sel.to(torch.int), descending=True)[1] 
        tokens[:count] = torch.sort(tokens[:count])[0] # tokens[:count]=[0,2] ->[0,2]
        # 为有效 token 分配连续的索引-[0,1]
        token_idx_in_rank[i][tokens[:count]] = torch.arange(count, dtype=torch.long, device='cuda') 
    for i in range(num_nodes):
        num_tokens_per_rdma_rank[i] = (rdma_rank_idx == i).sum()  # 统计每个 RDMA rank 接收到的 token 数量
    token_idx_in_rank = token_idx_in_rank.T.contiguous().to(torch.int)  # 转置并转换为整型
    is_token_in_rank = token_idx_in_rank >= 0  # 判断 token 是否有效
    gbl_num_tokens_per_rank = num_tokens_per_rank.clone() 
    # 得到所有进程中每个 rank 接收到的 token 总数
    dist.all_reduce(gbl_num_tokens_per_rank, group=group)

    ref_num_tokens_per_rank, ref_num_tokens_per_rdma_rank, \
    ref_num_tokens_per_expert, ref_is_token_in_rank, _ = \
        buffer.get_dispatch_layout(topk_idx, num_experts)
    # 使用 torch.allclose 验证计算的布局是否与参考布局(ref)一致。
    assert torch.allclose(ref_num_tokens_per_rank, num_tokens_per_rank)
    assert torch.allclose(ref_num_tokens_per_rdma_rank, num_tokens_per_rdma_rank)
    assert torch.allclose(ref_num_tokens_per_expert, num_tokens_per_expert)
    assert torch.allclose(ref_is_token_in_rank, is_token_in_rank)
    # 使用 bench 测量 buffer.get_dispatch_layout 的执行时间
    t = bench(lambda: buffer.get_dispatch_layout(topk_idx, num_experts))[0]
    if local_rank == 0:
        print(f'[layout] Kernel performance: {t * 1000:.3f} ms', flush=True)
        print('', flush=True)
    group.barrier()# 用于同步所有进程
    time.sleep(1)

    # Config
    rdma_buffer_size, nvl_buffer_size = 128, (720 if num_ranks in (144, 160) else 512)
    config = deep_ep.Config(num_sms, 8, nvl_buffer_size, 16, rdma_buffer_size)

    # Test dispatch
    # noinspection PyShadowingNames
    def check_data(check_x, recv_gbl_rank_prefix_sum):
        assert torch.allclose(check_x.amin(dim=1), check_x.amax(dim=1))# 每行相等
        check_start = 0
        for i in range(num_ranks):
            check_end = recv_gbl_rank_prefix_sum[i].item()
            assert (check_x[check_start:check_end, :].int() - i).sum().item() == 0
            check_start = check_end

    for previous_mode in (False, True):
        for async_mode in (False, True):
            for current_x in (x_pure_rand, x, x_e4m3):
                for with_topk in (False, True):
                    if local_rank == 0:
                        print(f'[testing] Running with {"FP8" if isinstance(current_x, tuple) else "BF16"}, {"with" if with_topk else "without"} top-k (async={async_mode}, previous={previous_mode}) ...', flush=True, end='')
                    dispatch_args = {'x': current_x, 'num_tokens_per_rank': num_tokens_per_rank, 'num_tokens_per_rdma_rank': num_tokens_per_rdma_rank,  'is_token_in_rank': is_token_in_rank,
                                     'num_tokens_per_expert': num_tokens_per_expert, 'config': config, 'async_finish': async_mode}
                    if with_topk:
                        dispatch_args.update({'topk_idx': topk_idx, 'topk_weights': topk_weights_pure_rand if current_x is x_pure_rand else topk_weights})
                    if previous_mode:
                        dispatch_args.update({'previous_event': buffer.capture()})
                    recv_x, recv_topk_idx, recv_topk_weights, recv_num_tokens_per_expert_list, handle, event = buffer.dispatch(**dispatch_args)
                    event.current_stream_wait() if async_mode else ()
                    recv_x = per_token_cast_back(*recv_x) if isinstance(recv_x, tuple) else recv_x

                    # Checks
                    recv_gbl_rank_prefix_sum = handle[-4]
                    assert gbl_num_tokens_per_rank[rank].item() == recv_x.size(0), f'{gbl_num_tokens_per_rank[rank].item()} != {recv_x.size(0)}'
                    assert gbl_num_tokens_per_expert.view(num_ranks, -1)[rank].tolist() == recv_num_tokens_per_expert_list
                    if current_x is not x_pure_rand:
                        check_data(recv_x, recv_gbl_rank_prefix_sum)
                    if with_topk:
                        # Check `topk_idx`
                        assert (recv_topk_idx.eq(-1) | ((recv_topk_idx >= 0) & (recv_topk_idx < (num_experts // num_ranks)))).sum().item() == recv_topk_idx.numel()
                        for i, count in enumerate(recv_num_tokens_per_expert_list):
                            assert recv_topk_idx.eq(i).sum().item() == count

                        # Check `topk_weights`
                        if current_x is not x_pure_rand:
                            recv_topk_weights[recv_topk_idx.eq(-1)] = recv_topk_weights.amax(dim=1, keepdim=True).expand_as(recv_topk_weights)[recv_topk_idx.eq(-1)]
                            check_data(recv_topk_weights, recv_gbl_rank_prefix_sum)

                    # Test cached dispatch (must without top-k staffs)
                    if not with_topk:
                        dispatch_args = {'x': current_x, 'handle': handle, 'config': config, 'async_finish': async_mode}
                        if previous_mode:
                            dispatch_args.update({'previous_event': buffer.capture()})
                        recv_x, _, _, _, _, event = buffer.dispatch(**dispatch_args)
                        event.current_stream_wait() if async_mode else ()
                        recv_x = per_token_cast_back(*recv_x) if isinstance(recv_x, tuple) else recv_x
                        if current_x is not x_pure_rand:
                            check_data(recv_x, recv_gbl_rank_prefix_sum)

                    # Test combine
                    combine_args = {'x': recv_x, 'handle': handle, 'config': config, 'async_finish': async_mode}
                    if with_topk:
                        combine_args.update({'topk_weights': recv_topk_weights})
                    if previous_mode:
                        combine_args.update({'previous_event': buffer.capture()})
                    # do combine,将接收的张量与其他张量进行合并，通常用于将分布式计算的结果聚合到一起。
                    combined_x, combined_topk_weights, event = buffer.combine(**combine_args)
                    event.current_stream_wait() if async_mode else ()
                    check_x = combined_x.float() / is_token_in_rank.sum(dim=1).unsqueeze(1)
                    ref_x = x_pure_rand if current_x is x_pure_rand else x
                    assert calc_diff(check_x, ref_x) < 5e-6
                    if with_topk:
                        check_topk_weights = combined_topk_weights if (current_x is x_pure_rand) else (combined_topk_weights / is_token_in_rank.sum(dim=1).unsqueeze(1))
                        ref_topk_weights = topk_weights_pure_rand if current_x is x_pure_rand else topk_weights
                        assert calc_diff(check_topk_weights, ref_topk_weights) < 1e-9

                    # For later tuning
                    dispatch_bf16_rdma_send_bytes = num_rdma_token_sent * hidden * 2
                    dispatch_bf16_nvl_recv_bytes = recv_x.numel() * 2
                    combine_bf16_nvl_send_bytes = dispatch_bf16_nvl_recv_bytes
                    combine_bf16_rdma_recv_bytes = dispatch_bf16_rdma_send_bytes

                    if local_rank == 0:
                        print(' passed', flush=True)
    if local_rank == 0:
        print('', flush=True)

    # Tune dispatch performance
    best_dispatch_results = None
    fp8_factor = (1 + 4 / 128) / 2
    for current_x in (x_e4m3, x):
        best_time, best_results = 1e10, None
        rdma_send_bytes = (dispatch_bf16_rdma_send_bytes * fp8_factor) if isinstance(current_x, tuple) else dispatch_bf16_rdma_send_bytes
        nvl_recv_bytes = (dispatch_bf16_nvl_recv_bytes * fp8_factor) if isinstance(current_x, tuple) else dispatch_bf16_nvl_recv_bytes
        for nvl_chunk_size in range(4, 33, 4):
            for rdma_chunk_size in range(4, 33, 4):
                config = deep_ep.Config(num_sms, nvl_chunk_size, nvl_buffer_size, rdma_chunk_size, rdma_buffer_size)
                tune_args = {'x': current_x, 'handle': handle, 'config': config}
                t = bench(lambda: buffer.dispatch(**tune_args))[0]
                if t < best_time:
                    best_time, best_results = t, (num_sms, nvl_chunk_size, rdma_chunk_size)
                if local_rank == 0:
                    print(f'[tuning] SMs {num_sms}, NVL chunk {nvl_chunk_size}, RDMA chunk {rdma_chunk_size}: {rdma_send_bytes / 1e9 / t:.2f} GB/s (RDMA), {nvl_recv_bytes / 1e9 / t:.2f} GB/s (NVL) ', flush=True)
        if local_rank == 0:
            print(f'[tuning] Best dispatch ({"FP8" if isinstance(current_x, tuple) else "BF16"}): SMs {best_results[0]}, NVL chunk {best_results[1]}, RDMA chunk {best_results[2]}: {rdma_send_bytes / 1e9 / best_time:.2f} GB/s (RDMA), {nvl_recv_bytes / 1e9 / best_time:.2f} GB/s (NVL)', flush=True)
            print('', flush=True)

        # 同步最佳配置
        if isinstance(current_x, tuple):
            # Gather FP8 the best config from rank 0
            best_dispatch_results = torch.tensor([best_results[0], best_results[1], best_results[2]], dtype=torch.int32, device='cuda')
            all_best_fp8_results_list = [torch.zeros_like(best_dispatch_results) for _ in range(torch.distributed.get_world_size())]
            # 使用 dist.all_gather 收集所有进程的最佳配置，并选择 rank 0 的配置作为最终结果。
            dist.all_gather(all_best_fp8_results_list, best_dispatch_results, group=group)
            best_dispatch_results = all_best_fp8_results_list[0].tolist()

    # 使用最佳配置重新dispatch
    dispatch_config = deep_ep.Config(best_dispatch_results[0], best_dispatch_results[1], nvl_buffer_size, best_dispatch_results[2], rdma_buffer_size)
    dispatch_args = {'x': x, 'num_tokens_per_rank': num_tokens_per_rank, 'num_tokens_per_rdma_rank': num_tokens_per_rdma_rank,
                     'is_token_in_rank': is_token_in_rank, 'num_tokens_per_expert': num_tokens_per_expert,
                     'config': dispatch_config if dispatch_config is not None else config}
    recv_x, _, _, _, handle, _ = buffer.dispatch(**dispatch_args)

    # Tune combine performance
    best_time, best_results = 1e10, None
    for nvl_chunk_size in range(1, 5, 1):
        for rdma_chunk_size in range(8, 33, 4):
            config = deep_ep.Config(num_sms, nvl_chunk_size, nvl_buffer_size, rdma_chunk_size, rdma_buffer_size)
            tune_args = {'x': recv_x, 'handle': handle, 'config': config}
            t = bench(lambda: buffer.combine(**tune_args))[0]
            if local_rank == 0:
                print(f'[tuning] SMs {num_sms}, NVL chunk {nvl_chunk_size}, RDMA chunk {rdma_chunk_size}: {combine_bf16_rdma_recv_bytes / 1e9 / t:.2f} GB/s (RDMA), {combine_bf16_nvl_send_bytes / 1e9 / t:.2f} GB/s (NVL) ', flush=True)
                if t < best_time:
                    best_time, best_results = t, (num_sms, nvl_chunk_size, rdma_chunk_size)

    if local_rank == 0:
        print(f'[tuning] Best combine: SMs {best_results[0]}, NVL chunk {best_results[1]}, RDMA chunk {best_results[2]}: {combine_bf16_rdma_recv_bytes / 1e9 / best_time:.2f} GB/s (RDMA), {combine_bf16_nvl_send_bytes / 1e9 / best_time:.2f} GB/s (NVL)', flush=True)
        print('', flush=True)


# noinspection PyUnboundLocalVariable （忽略）未绑定局部变量的检查
def test_loop(local_rank: int, num_local_ranks: int):
    # 初始化分布式环境
    num_nodes = int(os.getenv('WORLD_SIZE', 1))
    rank, num_ranks, group = init_dist(local_rank, num_local_ranks)
    test_ll_compatibility = True
    if test_ll_compatibility:
        ll_num_tokens, ll_hidden, ll_num_experts, ll_num_topk = 16, 5120, 256, 9

    num_sms = 24
    num_qps_per_rank = max(num_sms, ll_num_experts // num_ranks if test_ll_compatibility else 0)

    # 创建通信缓冲区
    buffer = deep_ep.Buffer(group, int(1e9), int(1e9), low_latency_mode=test_ll_compatibility,
                            num_qps_per_rank=num_qps_per_rank)
    assert num_local_ranks == 8 and num_ranks > 8
    torch.manual_seed(rank)# 设置当前进程的随机数种子

    # 调用主测试函数
    for i in (num_sms, ):
        test_main(i, local_rank, num_local_ranks, num_ranks, num_nodes, rank, buffer, group)
        if local_rank == 0:
            print('', flush=True)# 确保输出立即刷新

    # 测试低延迟模式
    # Test compatibility with low latency functions
    if test_ll_compatibility:
        buffer.clean_low_latency_buffer(ll_num_tokens, ll_hidden, ll_num_experts)
        test_low_latency.test_main(ll_num_tokens, ll_hidden, ll_num_experts, ll_num_topk, rank, num_ranks, group, buffer, seed=1)

    # Destroy the communication group
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    num_processes = 8
    torch.multiprocessing.spawn(test_loop, args=(num_processes, ), nprocs=num_processes)
