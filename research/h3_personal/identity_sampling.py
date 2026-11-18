"""Opt-in 65/35 replacement sampling; stdlib-only until iteration."""
from __future__ import annotations

ELIGIBLE_CLIPS = 715
POOL_MASS = {"identity": 0.65, "motion": 0.35}


def validate_sampling_config(config, context):
    budget = config.data.identity_sampling_budget
    start_draw = getattr(config.data, 'identity_sampling_start_draw', 0)
    if type(start_draw) is not int or start_draw < 0:
        raise ValueError('sampling start draw must be a nonnegative integer')
    if budget is None:
        if start_draw:
            raise ValueError('sampling cursor requires identity sampler')
        return
    if type(budget) is not int or budget <= 0:
        raise ValueError("identity_sampling_budget must be a positive draw count")
    mode = getattr(config.data, 'identity_sampling_mode', 'replacement')
    if mode not in ('replacement', 'coverage_first') or (mode == 'coverage_first' and budget != ELIGIBLE_CLIPS):
        raise ValueError('coverage_first requires exactly715 draws per epoch')
    if (context.world_size != 1 or context.rank != 0 or context.local_rank != 0
            or config.acceleration.strategy != "ddp"):
        raise ValueError("identity sampling requires single-process ddp (world_size=1)")
    if config.optimization.batch_size != 1 or not config.data.shuffle:
        raise ValueError("identity sampling requires batch_size=1 and shuffle=true")
    if config.data.val_split_every != 0:
        raise ValueError("all 715 eligible clips require val_split_every=0; no held-out claim")


class IdentityBatchSampler:
    """Each epoch draws budget singleton batches with replacement.

    Per-clip probability is pool mass / pool size. A positive probability does
    not guarantee visiting a clip in any finite epoch. CPU torch Generator is
    private; seed+epoch is independent of model/global RNG and record order.
    """

    def __init__(self, dataset, batch_size=1, shuffle=True, seed=0,
                 drop_last=False, *, sample_budget=715, max_seq_tokens=None, mode='replacement', start_draw=0):
        if batch_size != 1 or shuffle is not True or drop_last is not False:
            raise ValueError("requires singleton replacement batches, shuffle=true, drop_last=false")
        if type(sample_budget) is not int or sample_budget <= 0:
            raise ValueError("sample_budget must be a positive integer")
        if type(seed) is not int or not 0 <= seed < 2**63:
            raise ValueError("seed must be an integer in [0, 2**63)")
        if (getattr(dataset, "component_index", None) is None
                or dataset.split != "train" or len(dataset.records) != ELIGIBLE_CLIPS):
            raise ValueError("requires v2 train dataset with exactly 715 eligible records")
        ids = [r.id for r in dataset.records]
        if any(not isinstance(sid, str) or not sid for sid in ids) or len(set(ids)) != ELIGIBLE_CLIPS:
            raise ValueError("requires 715 unique nonempty IDs")
        self.buckets = {}
        pools = {k: [] for k in POOL_MASS}
        for index, record in enumerate(dataset.records):
            pool = (record.extras or {}).get("quality_pool")
            if not isinstance(pool, str) or pool not in pools:
                raise ValueError(f"missing/invalid quality_pool for {record.id}")
            if max_seq_tokens is not None and record.sequence_length > max_seq_tokens:
                raise ValueError(f"eligible clip exceeds max_seq_tokens: {record.id}")
            pools[pool].append(index)
            self.buckets.setdefault(record.bucket_key, []).append(index)
        if any(not members for members in pools.values()):
            raise ValueError("both identity and motion pools must be nonempty")
        self.indices = sorted(range(ELIGIBLE_CLIPS), key=lambda i: ids[i])
        self.weights = [POOL_MASS[dataset.records[i].extras["quality_pool"]]
                        / len(pools[dataset.records[i].extras["quality_pool"]]) for i in self.indices]
        if mode not in ('replacement', 'coverage_first') or (mode == 'coverage_first' and sample_budget != ELIGIBLE_CLIPS):
            raise ValueError('coverage_first requires exactly715 draws per epoch')
        self.mode = mode
        self.dataset = dataset
        self.seed = seed
        self.sample_budget = sample_budget
        if type(start_draw) is not int or start_draw < 0:
            raise ValueError('start_draw must be a nonnegative integer')
        self.start_draw = start_draw
        self.start_epoch, self.start_offset = divmod(start_draw, sample_budget)
        if seed + self.start_epoch >= 2**63:
            raise ValueError('sampling cursor seed/epoch overflow')
        self.epoch = self.start_epoch
        self.pool_counts = {k: len(v) for k, v in pools.items()}

    def set_epoch(self, epoch):
        if type(epoch) is not int or epoch < 0 or self.seed + self.start_epoch + epoch >= 2**63:
            raise ValueError("epoch must be nonnegative with seed+global epoch < 2**63")
        self.epoch = self.start_epoch + epoch

    def __iter__(self):
        if self.mode == 'coverage_first':
            # Coverage and65/35 replacement mass cannot both hold in one715-draw epoch.
            # Every clip appears once; pool frequencies are the actual590/125 counts.
            import random
            indices = list(self.indices)
            random.Random(self.seed + self.epoch).shuffle(indices)
            offset = self.start_offset if self.epoch == self.start_epoch else 0
            return iter([[i] for i in indices[offset:]])
        import torch
        generator = torch.Generator(device="cpu").manual_seed(self.seed + self.epoch)
        draws = torch.multinomial(torch.tensor(self.weights, dtype=torch.float64),
                                  self.sample_budget, replacement=True, generator=generator)
        offset = self.start_offset if self.epoch == self.start_epoch else 0
        return iter([[self.indices[i]] for i in draws.tolist()[offset:]])

    def __len__(self):
        return self.sample_budget - (self.start_offset if self.epoch == self.start_epoch else 0)


def build_sampler(dataset, config, fallback):
    budget = config.data.identity_sampling_budget
    kwargs = dict(batch_size=config.optimization.batch_size,
                  shuffle=config.data.shuffle, seed=config.seed)
    if budget is None:
        return fallback(dataset, **kwargs)
    return IdentityBatchSampler(dataset, **kwargs, sample_budget=budget,
                                max_seq_tokens=config.optimization.max_seq_tokens,
                                mode=getattr(config.data, 'identity_sampling_mode', 'replacement'),
                                start_draw=getattr(config.data, 'identity_sampling_start_draw', 0))
