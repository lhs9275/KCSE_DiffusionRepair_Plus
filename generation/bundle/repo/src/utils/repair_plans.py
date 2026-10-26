"""Source-addressable repair plans; independent of suspicious-node diagnostics."""
import ast
from difflib import SequenceMatcher

EDIT_OPERATIONS = frozenset({'Replace', 'InsertBefore', 'InsertAfter', 'Delete', 'GuardAdd'})
MAX_PROMPT_EDITS = 6
PLAN_BINDING_POLICY = 'source_span_v1'


def source_node_map(code, max_nodes=None):
    try:
        tree = ast.parse(code)
    except (SyntaxError, AttributeError) as exc:
        raise ValueError('repair plan: source cannot be parsed') from exc
    return {i: node for i, node in enumerate(ast.walk(tree))
            if (max_nodes is None or i < max_nodes) and getattr(node, 'lineno', None) is not None}


def validate_repair_edits(code, edits, max_nodes=None):
    """Reject an entire malformed plan, never silently repair or partly drop it."""
    if not isinstance(edits, list):
        raise ValueError('repair plan: edits must be a list')
    nodes = source_node_map(code, max_nodes) if edits else {}
    for edit in edits:
        if not isinstance(edit, dict) or edit.get('op') not in EDIT_OPERATIONS:
            raise ValueError('repair plan: invalid operation')
        nid = edit.get('node')
        if type(nid) is not int or nid not in nodes:
            raise ValueError('repair plan: node is not a source-addressable AST index')
    return nodes


def source_ranges_touched(before, after, ranges):
    """Line-diff proxy only: changed span or insertion at/inside its boundaries."""
    changed, inserted = set(), set()
    for tag, i1, i2, _, _ in SequenceMatcher(None, before.splitlines(), after.splitlines(), autojunk=False).get_opcodes():
        if tag in ('replace', 'delete'):
            changed.update(range(i1 + 1, i2 + 1))
        elif tag == 'insert':
            inserted.add(i1)  # boundary after i1 original lines
    return [bool(changed.intersection(range(start, end + 1))) or
            any(start - 1 <= boundary <= end for boundary in inserted)
            for start, end in ranges]


def source_span_touch_fraction(before, after, edits):
    """Fraction of addressed edits touched, not semantic operation fidelity."""
    nodes = validate_repair_edits(before, edits)
    if not edits:
        return None
    spans = [(nodes[e['node']].lineno, nodes[e['node']].end_lineno) for e in edits]
    touched = source_ranges_touched(before, after, spans)
    return sum(touched) / len(touched)


def addressed_plan_line_touched(before, after, edits):
    """Historical line-touch gate with unambiguous target addresses.

    Preserve historical diff matching and adjacent insertion acceptance. A
    compound node's body is not a substitute for touching its rendered line.
    """
    nodes = validate_repair_edits(before, edits)
    if not edits:
        return True
    targets = {nodes[edit['node']].lineno - 1 for edit in edits[:MAX_PROMPT_EDITS]}
    changed, inserted = set(), set()
    for tag, i1, i2, _, _ in SequenceMatcher(None, before.splitlines(), after.splitlines()).get_opcodes():
        if tag in ('replace', 'delete'):
            changed.update(range(i1, i2))
        elif tag == 'insert':
            inserted.add(i1)
    return any(line in changed or line in inserted or line + 1 in inserted for line in targets)
