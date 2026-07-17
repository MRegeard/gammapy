import ast
import re
import pathlib
from typing import Sequence, List, NamedTuple
import logging
from numpydoc.docscrape import NumpyDocString
import tempfile
import os

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


ARG_IGNORED = ["self", "cls"]


class NoArgumentError(Exception):
    pass


class Edit(NamedTuple):
    """A single text insertion to apply to the source file."""

    lineno: int
    col: int
    text: str


def get_source_and_tree(path: pathlib.Path):
    with open(path, "r", encoding="utf-8") as f:
        source = f.read()
    tree = ast.parse(source, filename=path)
    return source, tree


def get_func_args(func: ast.FunctionDef | ast.AsyncFunctionDef) -> list[ast.arg]:
    a = func.args
    args_list: list[ast.arg] = []
    args_list += [arg for arg in a.posonlyargs]
    args_list += [arg for arg in a.args]
    if a.vararg:
        args_list.append(a.vararg)
    args_list += [arg for arg in a.kwonlyargs]
    if a.kwarg:
        args_list.append(a.kwarg)
    return [arg for arg in args_list if arg.arg not in ARG_IGNORED]

    return [arg for arg in args_list if arg.arg not in ARG_IGNORED]


def get_list_functions_ast(
    node: ast.AST,
) -> List[ast.FunctionDef | ast.AsyncFunctionDef]:
    return [
        n
        for n in ast.walk(node)
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]


def is_annotate(arg: ast.arg) -> bool:
    return arg.annotation is not None


def get_doc_param_types(docstring: str) -> dict[str, str]:
    """Map each documented parameter name to its numpydoc type string.

    Handles grouped names like ``x, y : int`` by mapping both x and y.
    """
    parsed = NumpyDocString(docstring)
    types: dict[str, str] = {}
    for param in parsed["Parameters"]:
        for name in (n.strip() for n in param.name.split(",")):
            if name:
                types[name] = param.type
    return types


def translate_prose(t: str) -> str:
    """Best-effort mapping of numpydoc prose types to Python annotation syntax.

    Handles a few common idioms, recursively:
        "list of str"        -> "list[str]"
        "sequence of float"  -> "Sequence[float]"
        "dict of {str: int}" -> "dict[str, int]"
    Anything not recognized is returned unchanged for the caller to validate.
    """
    t = t.strip()
    # "list of str" / "set of int" / "sequence of float" -> "list[str]" ...
    m = re.fullmatch(
        r"(list|set|frozenset|sequence|iterable|tuple)\s+of\s+(.+)", t, re.I
    )
    if m:
        container = m.group(1).lower()
        container = {"sequence": "Sequence", "iterable": "Iterable"}.get(
            container, container
        )
        return f"{container}[{translate_prose(m.group(2))}]"
    # "dict of {str: int}" -> "dict[str, int]"
    m = re.fullmatch(r"dict\s+of\s+\{?\s*(.+?)\s*:\s*(.+?)\s*\}?", t, re.I)
    if m:
        return f"dict[{translate_prose(m.group(1))}, {translate_prose(m.group(2))}]"
    return t


def clean_type(type_str: str) -> str | None:
    """Turn a numpydoc type string into a usable annotation, or None.

    numpydoc types are free-form prose (``int, optional``, ``list of str``),
    so we trim common suffixes and then verify the result actually parses as
    a Python annotation expression. Anything that doesn't parse is skipped.
    """
    t = type_str.strip()
    for sep in (", optional", ", default", " optional"):
        idx = t.find(sep)
        if idx != -1:
            t = t[:idx].strip()

    if not t:
        return None
    t = translate_prose(t)
    try:
        ast.parse(f"_x: {t}")
    except SyntaxError:
        return None
    return t


def annotate_arg(arg: ast.arg, type_str: str) -> Edit | None:
    """Build the Edit that inserts ``: <type>`` right after the arg name."""
    cleaned = clean_type(type_str)
    if cleaned is None:
        return None
    return Edit(lineno=arg.end_lineno, col=arg.end_col_offset, text=f": {cleaned}")


def apply_edits(source: str, edits: list[Edit]) -> str:
    """Apply insertions edits to source text.

    Edits are applied from the end of the file backwargs so that earlier
    insertions don't shift the position of later ones.
    """
    lines = source.splitlines(keepends=True)
    line_start = [0]
    for line in lines:
        line_start.append(line_start[-1] + len(line))

    abs_edits = [(line_start[e.lineno - 1] + e.col, e.text) for e in edits]
    result = source
    for pos, text in sorted(abs_edits, key=lambda x: x[0], reverse=True):
        result = result[:pos] + text + result[pos:]
    return result


def collect_edits(
    fn_list: List[ast.FunctionDef | ast.AsyncFunctionDef],
) -> list[Edit]:
    edits: list[Edit] = []
    for fn in fn_list:
        fn_args = get_func_args(fn)
        if not fn_args:
            logger.info(f"Function {fn.name!r} has no arguments, skipping...")
            continue

        docstring = ast.get_docstring(fn)
        if docstring is None:
            logger.info(f"Function {fn.name!r} has no docstring, skipping...")
            continue

        doc_types = get_doc_param_types(docstring)
        if not doc_types:
            logger.info(
                f"Function {fn.name!r} has no documented parameters, skipping..."
            )
            continue

        for arg in fn_args:
            if is_annotate(arg):
                continue
            type_str = doc_types.get(arg.arg)
            if type_str is None:
                logger.warning(
                    f"Argument {arg.arg!r} in {fn.name!r} has no matching "
                    + "docstring parameter."
                )
                continue
            edit = annotate_arg(arg, type_str)
            if edit is None:
                logger.warning(
                    f"Could not turn documented type {type_str!r} for "
                    + f"{arg.arg!r} in {fn.name!r} into a valid annotation."
                )
                continue
            logger.info(f"Annotation {fn.name}.{arg.arg} -> {edit.text.strip()}")
            edits.append(edit)
    return edits


def main(argv: Sequence[str] | None = None):
    if argv is None:
        argv = []
    if len(argv) == 0:
        raise NoArgumentError("Missing required argument: path to apply this script.")

    path = pathlib.Path(argv[0])
    source, tree = get_source_and_tree(path)

    fn_list = get_list_functions_ast(tree)
    logger.info(f"Found {len(fn_list)} function definition in file {path}.")

    edits = collect_edits(fn_list)
    if not edits:
        logger.info("No annotations to add.")
        return

    new_source = apply_edits(source, edits)

    ast.parse(new_source, filename=str(path))

    fd, tmp_path = tempfile.mkstemp(dir=path.parent, suffix=".py")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as tmp_file:
            tmp_file.write(new_source)
        os.replace(tmp_path, path)
    except BaseException:
        os.unlink(tmp_path)
        raise

    logger.info(f"Applied {len(edits)} annotation(s) to {path}.")


if __name__ == "__main__":
    import sys

    main(sys.argv[1:])
