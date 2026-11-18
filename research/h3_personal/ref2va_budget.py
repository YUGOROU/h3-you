"""USER HF remaining authority $9.40 from bound221.672274; Modal separate."""
from decimal import Decimal
from research.h3_personal.realtime_first import budget_bound as legacy_bound

ORIGIN=Decimal('169.35')
HF_INCREMENT=Decimal('26.00')
# New pool supersedes (does not add) the old $1.526663 remainder.
HF_LIVE_POOL_BASE=Decimal('193.823337')
HF_LIVE_POOL=Decimal('22.79')
# USER go: one new finite full-pipeline test <=$2.60 from bound214.163374.
# Only the needed headroom is added, not the reported $9.18 credit balance.
HF_ATTENDED_HEADROOM=Decimal('0.150037')
# Latest USER increases REMAINING authority to9.40, replacing (not adding to)2.613326.
HF_AUTHORIZED_REMAINDER_BASE=Decimal('221.672274')
HF_AUTHORIZED_REMAINDER=Decimal('9.40')
HF_CAP=HF_AUTHORIZED_REMAINDER_BASE+HF_AUTHORIZED_REMAINDER
MODAL_SHARED=Decimal('20.44')


def budget_bound(*,ledger_bound=ORIGIN,hourly_rate=Decimal('10.00002'),timeout_seconds=1500):
    # Keep legacy rate/finite-time protections; do not expand legacy $170 authority.
    cost=legacy_bound(ledger_bound=Decimal(0),hourly_rate=hourly_rate,timeout_seconds=timeout_seconds)['job_bound']
    if not ledger_bound.is_finite() or ledger_bound<ORIGIN or ledger_bound+cost>HF_CAP:
        raise ValueError('HF authorized cap exceeded or origin uncertain')
    return dict(job_bound=cost,cumulative_bound=ledger_bound+cost,
                remaining=HF_CAP-ledger_bound-cost,authorized_cap=HF_CAP)
