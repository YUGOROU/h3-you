"""Make opt-in local CPU aspect-correction candidates; never overwrites original."""
import argparse
import json
import os
from pathlib import Path
from research.h3_personal.geometry_preview import render,filters


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--ratio',type=float,action='append',required=True)
    p.add_argument('--execute',action='store_true');a=p.parse_args(argv)
    for r in a.ratio:filters(768,1344,r)
    if not a.execute:
        print(json.dumps(dict(plan_only=True,ratios=a.ratio,modes=['pad','crop'],gpu=False,
                              automatic_ratio_estimation=False)));return 0
    os.umask(0o077);r=render(a.input,a.output,a.ratio)
    print(json.dumps(dict(output=str(a.output),candidates=len(r['candidates']),
                         ratio_verified=False,original_unchanged=r['original_unchanged'])));return 0


if __name__=='__main__':raise SystemExit(main())
