import json
import sys
from pathlib import Path

context = json.load(sys.stdin)
root = Path(context['output_root'])
file = root / 'smoke_summary.json'
done = file.exists() and json.loads(file.read_text()).get('state') == 'complete'
print(json.dumps({'protocol': 'rrctl.adapter.v1', 'healthy': done, 'complete': done,
                  'progress': {}, 'observations': {},
                  'artifacts': ['smoke_summary.json'] if done else []}))
