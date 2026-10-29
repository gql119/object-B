"""Retain only the pinned native loop's optimizer-attempt phase for resume.

No third-party files or update arithmetic are modified. Fresh training starts
at the same -1 phase. The additional attribute records even skipped AMP
attempts, exactly as the native local last_opt_step does.
"""
import ast
import hashlib
import inspect
from pathlib import Path
import textwrap
from ultralytics.engine import trainer as native
from .common import protocol


def phase_preserving_loop():
    expected = protocol()['native_sources']['engine/trainer.py']
    if hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest() != expected:
        raise ValueError('Native loop source differs from the pinned implementation')
    tree = ast.parse(textwrap.dedent(inspect.getsource(native.BaseTrainer._do_train)))
    initializers = 0
    attempts = 0

    class Phase(ast.NodeTransformer):
        def visit_Assign(self, node):
            nonlocal initializers, attempts
            if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name) or node.targets[0].id != 'last_opt_step':
                return node
            if ast.dump(node.value) == ast.dump(ast.parse('-1', mode='eval').body):
                initializers += 1
                if initializers == 1:
                    node.value = ast.parse('self.at_last_opt_step', mode='eval').body
                return node
            if isinstance(node.value, ast.Name) and node.value.id == 'ni':
                attempts += 1
                return [node, ast.parse('self.at_last_opt_step = last_opt_step').body[0]]
            raise ValueError('Unknown native optimizer phase assignment')

    tree = Phase().visit(tree)
    if (initializers, attempts) != (2, 1):
        raise ValueError('Native loop optimizer cadence contract changed')
    ast.fix_missing_locations(tree)
    scope = dict(vars(native))
    exec(compile(tree, '<AT pinned native optimizer phase binding>', 'exec'), scope)
    return scope['_do_train']
