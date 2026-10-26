"""Fail-closed structural checks; passing these is not semantic validation."""
import ast
import copy
import re


EXAMPLE_REASON = 'Appending to deque without checking order requirements'
AST_FREE_POLICY = 'none_v1'


def without_ast_diagnostics(item):
    """Drop diagnostics and cached downstream products before an AST-free run."""
    result = copy.deepcopy(item)
    for key in ('suspicious_nodes_topk', 'suspicious_nodes_topk_reranked',
                'static_analysis_nodes', 'bm25', 'diffusion_plan', 'prompt'):
        result.pop(key, None)
    return result


def reason_error(result, expected_line=None):
    if not isinstance(result, dict):
        return 'result is not an object'
    line, reason = result.get('line'), result.get('reason')
    if type(line) is not int or line < 1 or (expected_line is not None and line != expected_line):
        return 'line mismatch'
    if not isinstance(reason, str) or not reason.strip():
        return 'empty or non-string reason'
    if not reason.isascii() or not re.search('[A-Za-z]', reason) or len(reason.split()) > 20:
        return 'reason violates English/length format'
    lowered = reason.strip().lower()
    if lowered.rstrip('.') in {'this is a bug', 'this line is suspicious', 'this node is suspicious'}:
        return 'tautology instead of an explanation'
    if lowered.startswith(('no valid json', 'llm ', 'no_output')) or lowered.rstrip('.') == EXAMPLE_REASON.lower():
        return 'failure sentinel or copied prompt example'
    if result.get('reason_status', 'valid') != 'valid':
        return 'reason is not marked valid'
    try:
        ast.parse(reason.strip(), mode='eval')
    except SyntaxError:
        pass
    else:
        return 'code expression instead of an English explanation'
    return None


def validate_ast_hints(item, nodes):
    if not nodes:
        raise ValueError('AST hint: enabled input has no nodes')
    code = (item.get('function') or {}).get('function_before') or ''
    try:
        walked = list(ast.walk(ast.parse(code)))
    except SyntaxError as error:
        raise ValueError('AST hint: invalid source function') from error
    for node in nodes:
        error = reason_error(node)
        if error:
            raise ValueError('AST hint: ' + error)
        node_id = node.get('node_id')
        if type(node_id) is not int or not 0 <= node_id < len(walked):
            raise ValueError('AST hint: node ID mismatch')
        actual = walked[node_id]
        if getattr(actual, 'lineno', None) != node['line'] or type(actual).__name__ != node.get('type'):
            raise ValueError('AST hint: node/line/type mismatch')
        # These records are untruncated analyzer nodes; compare normalized source.
        if hasattr(ast, 'unparse') and ast.unparse(actual).strip() != str(node.get('code') or '').strip():
            raise ValueError('AST hint: node code mismatch')
