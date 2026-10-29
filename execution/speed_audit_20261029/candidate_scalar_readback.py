"""Isolated candidate: batch diagnostic scalar readback after unchanged PGD-10.

Not imported by the formal worker. All immediate finite/GT/gradient/budget guards,
native model/loss calls, gradient calls, and adversarial updates remain unchanged.
"""
import ast
import hashlib
import inspect
from pathlib import Path
import textwrap

from robustness_s4_at import pgd

def build_candidate(expected_base_sha256, *, vectorize_counts=False):
    actual = hashlib.sha256(Path(pgd.__file__).read_bytes()).hexdigest()
    if actual != expected_base_sha256:
        raise ValueError("Reference PGD source differs from the live-audited source")
    tree = ast.parse(textwrap.dedent(inspect.getsource(pgd.PGDAttack.__call__)))
    changed = 0
    inserted = 0
    counts_changed = 0

    class Readback(ast.NodeTransformer):
        def visit_Assign(self, node):
            nonlocal counts_changed
            if (vectorize_counts and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                    and node.targets[0].id == "matched"):
                counts_changed += 1
                return ast.parse("""
row_ids = torch.arange(len(fg), device=local_gt.device, dtype=local_gt.dtype).view(-1, 1)
keys = row_ids * (inputs[5].shape[1] + 1) + local_gt
matched = torch.unique(keys[fg.bool()]).numel()
""").body
            return self.generic_visit(node)

        def visit_Expr(self, node):
            nonlocal changed
            call = node.value
            if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
                    and isinstance(call.func.value, ast.Name) and call.func.value.id == "steps"
                    and call.func.attr == "append"):
                return node
            changed += 1
            return ast.parse("""
steps.append((step, loss.detach(), tuple((k, v.detach()) for k, v in parts.items()),
              grad.detach().norm(), dict(assignments[-1])))
""").body[0]

        def visit_If(self, node):
            nonlocal inserted
            if ast.unparse(node.test) != "before_gt != gt_identity(batch)":
                return self.generic_visit(node)
            inserted += 1
            materialize = ast.parse("""
records = steps
packed = torch.stack([torch.stack([loss, *[v for _, v in parts], norm])
                      for _, loss, parts, norm, _ in records]).cpu().tolist()
steps = [dict(step=record[0], loss=row[0],
              loss_parts=dict(zip([k for k, _ in record[2]], row[1:-1])),
              input_grad_norm=row[-1], **record[4])
         for record, row in zip(records, packed)]
""").body
            return materialize + [node]

    tree = Readback().visit(tree)
    if (changed, inserted) != (1, 1):
        raise ValueError("Candidate did not match exactly the two expected diagnostic sites")
    if counts_changed != int(vectorize_counts):
        raise ValueError("Candidate did not match the expected diagnostic counting site")
    ast.fix_missing_locations(tree)
    scope = dict(vars(pgd))
    exec(compile(tree, "<isolated PGD diagnostic readback candidate>", "exec"), scope)
    candidate = type("BatchedScalarReadbackAttack", (pgd.PGDAttack,), {"__call__": scope["__call__"]})
    return candidate, ast.unparse(tree)
