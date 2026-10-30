"""Per-filesystem capacity checks for the independent AutoDL deployment."""
import os
from pathlib import Path
import shutil

GIB = 2 ** 30
PROFILE = 'autodl_30g50g_v1'


def _path(name):
    raw = os.environ.get(name)
    if not raw:
        raise ValueError(f'{name} is required by the cloud storage profile')
    return Path(raw).expanduser()


def _existing(path):
    value = Path(path)
    while not value.exists() and value != value.parent:
        value = value.parent
    if not value.exists():
        raise FileNotFoundError(f'No existing parent for storage path: {path}')
    return value.resolve()


def _measure(path):
    actual = _existing(path)
    stat = actual.stat()
    usage = shutil.disk_usage(actual)
    return dict(path=str(path), resolved=str(actual), device=stat.st_dev,
                total_bytes=usage.total, free_bytes=usage.free,
                free_GiB=round(usage.free / GIB, 3))


def storage_status(phase='run'):
    """Measure actual mounted filesystems; never add free space across volumes."""
    if phase not in ('install', 'data_stage', 'run', 'monitor'):
        raise ValueError('Unknown storage check phase')
    blockers, roles = [], {}
    root = _path('S4_AT_STORAGE_ROOT')
    roles['storage_root'] = _measure(root)
    root_device = roles['storage_root']['device']
    roles['system_root'] = _measure(Path('/'))
    mount = _path('S4_AT_STORAGE_MOUNT')
    roles['storage_mount'] = _measure(mount)
    mounted_path = Path(roles['storage_mount']['resolved'])
    if not os.path.ismount(str(mounted_path)):
        blockers.append('CONFIGURED_STORAGE_PATH_IS_NOT_A_MOUNTPOINT')
    if roles['storage_mount']['device'] != root_device:
        blockers.append('STORAGE_ROOT_NOT_ON_CONFIGURED_MOUNT')
    if roles['storage_mount']['device'] == roles['system_root']['device']:
        blockers.append('DATA_VOLUME_NOT_DISTINCT_FROM_SYSTEM_ROOT')

    if phase == 'install':
        roles['source'] = _measure(_path('S4_AT_SOURCE_ROOT'))
        roles['owned_output'] = _measure(_path('S4_AT_OWNED_ROOT'))
        roles['environment'] = _measure(_path('S4_AT_ENV_ROOT'))
        for role in ('source', 'owned_output', 'environment'):
            if roles[role]['device'] != root_device:
                blockers.append(f'{role.upper()}_NOT_ON_CONFIGURED_DATA_VOLUME')
        target = {root_device: ('storage_root/environment setup', 15.0)}
        target.setdefault(roles['system_root']['device'], ('system root reserve', 2.0))
    elif phase == 'data_stage':
        roles['s4_parent'] = _measure(_path('S4_AT_S4_PARENT_ROOT'))
        if roles['s4_parent']['device'] != root_device:
            blockers.append('S4_PARENT_NOT_ON_CONFIGURED_DATA_VOLUME')
        target = {roles['s4_parent']['device']: ('poison archive + expansion + reserve', 12.0)}
        target.setdefault(roles['system_root']['device'], ('system root reserve', 2.0))
    else:
        names = {
            'source': 'S4_AT_SOURCE_ROOT',
            'owned_output': 'S4_AT_OWNED_ROOT',
            'environment': 'S4_AT_ENV_ROOT',
            'clean_data': 'S4_AT_CLEAN_ROOT',
            's4_parent': 'S4_AT_S4_PARENT_ROOT',
            'temporary': 'TMPDIR',
            'pip_cache': 'PIP_CACHE_DIR',
            'torch_cache': 'TORCH_HOME',
            'xdg_cache': 'XDG_CACHE_HOME',
        }
        for role, variable in names.items():
            configured = _path(variable)
            if not configured.exists():
                blockers.append(f'REQUIRED_STORAGE_PATH_MISSING:{variable}')
            roles[role] = _measure(configured)
        for role in ('source', 'owned_output', 'environment', 'clean_data', 's4_parent',
                     'temporary', 'pip_cache', 'torch_cache', 'xdg_cache'):
            if roles[role]['device'] != root_device:
                blockers.append(f'{role.upper()}_NOT_ON_CONFIGURED_DATA_VOLUME')
        # The 3 GiB output allowance includes model snapshots, full-state
        # checkpoint temp writes, PGD audit, validation and plots. Keep 5 GiB
        # free after that allowance; inputs/caches keep an independent 2 GiB.
        target = {root_device: (('runtime safety reserve', 5.0) if phase == 'monitor' else
                                ('one-arm output 3 GiB + safety reserve 5 GiB', 8.0))}
        target.setdefault(roles['clean_data']['device'], ('dataset reserve', 2.0))
        target.setdefault(roles['s4_parent']['device'], ('dataset reserve', 2.0))
        target.setdefault(roles['temporary']['device'], ('cache/temp reserve', 2.0))
        target.setdefault(roles['system_root']['device'], ('system root reserve', 2.0))
    filesystems = {}
    for role, measure in roles.items():
        device = measure['device']
        group = filesystems.setdefault(device, dict(device=device, paths=[], free_bytes=measure['free_bytes'],
                                                      total_bytes=measure['total_bytes']))
        group['paths'].append(role)
        group['free_bytes'] = min(group['free_bytes'], measure['free_bytes'])
        group['total_bytes'] = min(group['total_bytes'], measure['total_bytes'])
    for device, (reason, required_gib) in target.items():
        group = filesystems.get(device)
        if group is None or group['free_bytes'] < required_gib * GIB:
            actual = group['free_bytes'] / GIB if group else 0.0
            blockers.append(f'DISK_FREE_BELOW_BUDGET:{reason}:{actual:.3f}<{required_gib:.3f}GiB')
    summary = [dict(device=item['device'], paths=item['paths'], total_GiB=round(item['total_bytes']/GIB, 3),
                    free_GiB=round(item['free_bytes']/GIB, 3)) for item in filesystems.values()]
    return dict(admitted=not blockers, phase=phase, profile=PROFILE, blockers=blockers,
                roles=roles, filesystems=summary, targets_GiB={str(k): v[1] for k, v in target.items()},
                free_space_was_not_summed_across_filesystems=True)
