"""CPU/GPU speed telemetry; unchanged capacity and sustained I/O safeguards."""

POLICY_ID = 'cpu_gpu_speed_only_with_io_guard_human_20261019_v1'


def capacity_blockers(snapshot):
    """Keep capacity limits; no busy-CPU, GPU-utilization or occupancy veto."""
    raw = snapshot['gpu']['device_raw'].splitlines()[0].split(',')
    free_mib = int(raw[3].strip())
    disk_free = snapshot.get('disk_free_bytes')
    if disk_free is None:
        disk_free = snapshot['disk']['free_bytes']
    blockers = []
    if snapshot['mem']['MemAvailable_kib'] * 1024 < 32 * 1024**3:
        blockers.append('MemAvailable below 32 GiB')
    if free_mib < 24000:
        blockers.append('GPU free memory below 24000 MiB')
    if disk_free < 150 * 1024**3:
        blockers.append('disk free below 150 GiB')
    return blockers


def sustained_iowait(samples):
    """Preserve original all-three-windows I/O-wait>=20% safeguard."""
    return len(samples) >= 3 and all(s['io_wait_percent_sample'] >= 20.0 for s in samples)
