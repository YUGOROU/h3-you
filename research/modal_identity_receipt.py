"""Bounded private execution receipt; not compute-country or likeness proof."""
import json
from pathlib import Path


def publish(api, *, repo, request_sha256, report, placement, operation, references=None):
    if api.repo_info(repo, repo_type='model').private is not True:
        raise ValueError('private execution receipt repository required')
    value = {'schema': 'h3-modal-execution-receipt-v1', 'request_sha256': request_sha256,
             'requested_placement': placement, 'observed_country_verified': False,
             'quality_verified': False,
             'pipeline': {key: report[key] for key in ('stage', 'readiness', 'failure_type',
                         'canonical_reload_verified', 'private_uploaded') if key in report}}
    if references:
        value['checkpoint_references'] = {key: references[key] for key in
            ('run_id', 'escrow_revision', 'escrow_prefix', 'escrow_exported_only') if key in references}
    data = json.dumps(value, sort_keys=True, allow_nan=False).encode()
    if len(data) > 8192:
        raise ValueError('bounded execution receipt exceeded')
    commit = api.create_commit(repo, repo_type='model', operations=[operation(
        path_in_repo='execution/modal/' + request_sha256 + '/receipt.json', path_or_fileobj=data)],
        commit_message='Private bounded Modal execution receipt')
    return commit.oid
