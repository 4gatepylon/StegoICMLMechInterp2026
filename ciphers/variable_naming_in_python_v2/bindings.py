"""Static source binding extraction for the V2 decoder, without executing code.

The AST supplies scope boundaries and name contexts. Original source tokens supply
identifier spans for AST fields represented as strings (imports, patterns, etc.).
Collection precedes resolution so forward references and declarations see the
whole lexical block. Comprehensions get distinct scopes even on Python versions
that inline their bytecode. Type-parameter/alias scopes are rejected explicitly.

TODO(hadriano) a human never read this. This is mostly tested by the tests in `ciphers/variable_naming_in_python_v2/tests/test_decoder.py`.
"""

import ast
import io
import tokenize
import unicodedata
from bisect import bisect_left
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable, Literal

from ciphers.variable_naming_in_python_v2.decoder import BindingOccurrence, OccurrenceKind, SourceSpan, UnsupportedSyntaxError, VariableBinding


@dataclass(eq=False)
class _Scope:
    """Mutable collection state; names in all sets use Python's private mangling.

    ``locals`` records syntactic binding operations before global/nonlocal
    redirection. ``parent`` models lexical containment, not runtime visibility;
    resolution skips intervening class namespaces. ``class_name`` is inherited
    through methods and comprehensions solely for private-name mangling.
    """

    scope_id: str
    kind: Literal["module", "function", "class", "comprehension"]
    parent: "_Scope | None" = None
    class_name: str = ""
    locals: set[str] = field(default_factory=set)
    globals: set[str] = field(default_factory=set)
    nonlocals: set[str] = field(default_factory=set)

    def canonical_name(self, name: str) -> str:
        """Return the compiler spelling of a normalized source name in this scope."""
        prefix = self.class_name.lstrip("_")
        if prefix and name.startswith("__") and not name.endswith("__"):
            return f"_{prefix}{name}"
        return name


@dataclass
class _NameOccurrence:
    """Unresolved source event; scope is where lexical lookup must begin."""

    name: str
    canonical_name: str
    scope: _Scope
    occurrence: BindingOccurrence


class _Collector(ast.NodeVisitor):
    """Collect scopes and identifier events from a validated AST and original text.

    ``code`` must be the same source used to create the tree passed to ``visit``.
    ``name_positions`` indexes candidate identifier token starts in source order;
    ``claimed`` prevents repeated string-valued AST fields claiming the same
    token. The resulting scopes/events are consumed only by ``resolve``.
    """

    def __init__(self, code: str) -> None:
        # Match Python's universal newlines, without splitting form feeds or
        # Unicode separators inside strings into nonexistent source lines.
        normalized_newlines = code.replace("\r\n", "\n").replace("\r", "\n")
        self.lines = normalized_newlines.split("\n")
        self.root = _Scope("module", "module")
        self.scope = self.root
        self.scopes = [self.root]
        self.events: list[_NameOccurrence] = []
        self.claimed: set[tuple[int, int]] = set()
        self.name_positions: list[tuple[int, int]] = []
        for token in tokenize.generate_tokens(io.StringIO(normalized_newlines).readline):
            if token.type == tokenize.NAME:
                line, column = token.start
                self.name_positions.append((line, len(self.lines[line - 1][:column].encode("utf-8"))))

    def _identifier(self, line: int, column: int) -> tuple[str, SourceSpan]:
        """Read the identifier at a UTF-8 start; return normalized name and raw span.

        The AST normalizes identifiers but source columns refer to the original
        spelling. Scanning identifier continuations also handles combining marks
        which tokenize may split into separate tokens. Callers supply a known
        identifier start, never an arbitrary expression start.
        """
        tail = self.lines[line - 1].encode("utf-8")[column:].decode("utf-8")
        spelling = ""
        for character in tail:
            if not ("_" + character).isidentifier():
                break
            spelling += character
        return unicodedata.normalize("NFKC", spelling), SourceSpan(line=line, column=column, end_line=line, end_column=column + len(spelling.encode("utf-8")))

    def _field_span(self, node: ast.AST, name: str, *, last: bool = False) -> SourceSpan:
        """Find an unclaimed normalized identifier within a string-valued AST field.

        ``node`` bounds the search in original UTF-8 coordinates. ``name`` is the
        normalized AST field. ``last`` selects suffix fields such as import aliases
        and pattern captures rather than a same-named expression earlier in the
        node. Return the exact identifier span; fail rather than guess if missing.
        """
        start, end = (node.lineno, node.col_offset), (node.end_lineno, node.end_col_offset)
        candidates = self.name_positions[bisect_left(self.name_positions, start) : bisect_left(self.name_positions, end)]
        positions = reversed(candidates) if last else iter(candidates)
        for position in positions:
            if start <= position < end and position not in self.claimed:
                identifier, span = self._identifier(*position)
                if identifier == name:
                    return span
        raise UnsupportedSyntaxError(f"Cannot locate identifier {name!r} at line {node.lineno}, column {node.col_offset}")

    def _record(self, name: str, span: SourceSpan, kind: OccurrenceKind, scope: _Scope | None = None) -> None:
        """Append one token event and register syntactic locals; return nothing.

        ``scope`` overrides lookup only for comprehension walrus targets. Names
        and spans come from the original AST/source, and ``kind`` follows the
        public BindingOccurrence contract. Declarations and reads create no new
        local; writes and deletion do, subject to later scope redirection.
        """
        owner = self.scope if scope is None else scope
        canonical = owner.canonical_name(name)
        self.claimed.add((span.line, span.column))
        self.events.append(_NameOccurrence(name, canonical, owner, BindingOccurrence(kind=kind, span=span)))
        if kind not in ("read", "declaration"):
            owner.locals.add(canonical)

    def _field(self, node: ast.AST, name: str, kind: OccurrenceKind = "binding", *, last: bool = False) -> None:
        """Record a string-valued identifier field using its source token span."""
        self._record(name, self._field_span(node, name, last=last), kind)

    def _visit_all(self, nodes: Iterable[ast.AST | None]) -> None:
        """Visit optional AST children in the current scope, ignoring absent nodes."""
        for node in nodes:
            if node is not None:
                self.visit(node)

    def _child(self, node: ast.AST, kind: Literal["function", "class", "comprehension"]) -> _Scope:
        """Allocate a source-position-identified child and return it without entering.

        ``node`` anchors the scope; its AST type distinguishes different kinds of
        scopes at the same position. ``kind`` controls lookup rules. Function
        names are not identities, so repeated def statements remain distinct.
        """
        child = _Scope(f"{self.scope.scope_id}/{type(node).__name__}@{node.lineno}:{node.col_offset}", kind, self.scope, self.scope.class_name)
        if isinstance(node, ast.ClassDef):
            child.class_name = node.name
        self.scopes.append(child)
        return child

    def visit_Name(self, node: ast.Name) -> None:
        """Record loads/stores/deletions, preserving original identifier spelling spans."""
        kinds: dict[type[ast.expr_context], OccurrenceKind] = {ast.Load: "read", ast.Store: "write", ast.Del: "delete"}
        self._record(node.id, self._identifier(node.lineno, node.col_offset)[1], kinds[type(node.ctx)])

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        """Count a name target once as read_write; attribute/subscript bases are reads."""
        if isinstance(node.target, ast.Name):
            self._record(node.target.id, self._identifier(node.target.lineno, node.target.col_offset)[1], "read_write")
        else:
            self.visit(node.target)
        self.visit(node.value)

    def visit_NamedExpr(self, node: ast.NamedExpr) -> None:
        """Bind walrus targets outside enclosing comprehensions; value keeps its scope."""
        owner = self.scope
        while owner.kind == "comprehension":
            owner = owner.parent
        self._record(node.target.id, self._identifier(node.target.lineno, node.target.col_offset)[1], "write", owner)
        self.visit(node.value)

    def _function(self, node: ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda) -> None:
        """Collect outer-scope defaults/annotations and inner-scope parameters/body.

        ``node`` can be sync/async def or lambda. Def names bind in the enclosing
        scope; decorators, annotations, and defaults also use that scope. Return
        nothing after restoring the enclosing scope. Generic parameter scopes
        are rejected until their separate annotation lookup rules are supported.
        """
        if getattr(node, "type_params", ()):
            raise UnsupportedSyntaxError(f"Generic type-parameter scopes are not supported at line {node.lineno}, column {node.col_offset}")
        if not isinstance(node, ast.Lambda):
            self._field(node, node.name)
            self._visit_all(node.decorator_list)
            self._visit_all([node.returns])
        arguments = node.args
        parameters = [*arguments.posonlyargs, *arguments.args, *arguments.kwonlyargs]
        parameters += [arg for arg in (arguments.vararg, arguments.kwarg) if arg is not None]
        self._visit_all([*arguments.defaults, *arguments.kw_defaults])
        self._visit_all(arg.annotation for arg in parameters)
        enclosing = self.scope
        self.scope = self._child(node, "function")
        for argument in parameters:
            self._record(argument.arg, self._identifier(argument.lineno, argument.col_offset)[1], "binding")
        self._visit_all([node.body] if isinstance(node, ast.Lambda) else node.body)
        self.scope = enclosing

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Visit a function with explicit outer and parameter/body scopes."""
        self._function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        """Async functions introduce the same binding scopes as synchronous functions."""
        self._function(node)

    def visit_Lambda(self, node: ast.Lambda) -> None:
        """Visit lambda defaults outside its new parameter/body scope."""
        self._function(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Bind the class name outside, then visit its body in a distinct namespace."""
        if node.type_params:
            raise UnsupportedSyntaxError(f"Generic type-parameter scopes are not supported at line {node.lineno}, column {node.col_offset}")
        self._field(node, node.name)
        self._visit_all([*node.decorator_list, *node.bases, *node.keywords])
        enclosing = self.scope
        self.scope = self._child(node, "class")
        self._visit_all(node.body)
        self.scope = enclosing

    def _comprehension(self, node: ast.ListComp | ast.SetComp | ast.DictComp | ast.GeneratorExp) -> None:
        """Visit the first iterable outside; targets, filters, and results inside.

        ``node`` supplies all generators in one comprehension scope. Later
        iterables can reference earlier targets; nested comprehensions create
        further scopes via the same visitor. Restore the enclosing scope on
        return. Walrus targets are redirected separately by visit_NamedExpr.
        """
        self.visit(node.generators[0].iter)
        enclosing = self.scope
        self.scope = self._child(node, "comprehension")
        for index, generator in enumerate(node.generators):
            if index:
                self.visit(generator.iter)
            self.visit(generator.target)
            self._visit_all(generator.ifs)
        self._visit_all([node.key, node.value] if isinstance(node, ast.DictComp) else [node.elt])
        self.scope = enclosing

    def visit_ListComp(self, node: ast.ListComp) -> None:
        """Preserve comprehension-local bindings despite bytecode inlining."""
        self._comprehension(node)

    def visit_SetComp(self, node: ast.SetComp) -> None:
        """Visit a set comprehension using isolated target bindings."""
        self._comprehension(node)

    def visit_DictComp(self, node: ast.DictComp) -> None:
        """Visit both key/value expressions in their comprehension scope."""
        self._comprehension(node)

    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> None:
        """Use generator-expression lexical scopes without running the generator."""
        self._comprehension(node)

    def visit_Import(self, node: ast.Import) -> None:
        """Bind an alias, or only the first component of an unaliased dotted import."""
        for alias in node.names:
            self._field(alias, alias.asname or alias.name.split(".")[0], last=alias.asname is not None)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """Record imported names without importing; reject unknowable wildcard names."""
        for alias in node.names:
            if alias.name == "*":
                raise UnsupportedSyntaxError(f"Wildcard import bindings cannot be enumerated at line {node.lineno}, column {node.col_offset}")
            self._field(alias, alias.asname or alias.name, last=alias.asname is not None)

    def visit_Global(self, node: ast.Global) -> None:
        """Record declaration tokens and mark their names for module resolution."""
        for name in node.names:
            self.scope.globals.add(self.scope.canonical_name(name))
            self._field(node, name, "declaration")

    def visit_Nonlocal(self, node: ast.Nonlocal) -> None:
        """Record declaration tokens and redirect their names to enclosing functions."""
        for name in node.names:
            self.scope.nonlocals.add(self.scope.canonical_name(name))
            self._field(node, name, "declaration")

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        """Bind exception targets after visiting the exception type, then the body."""
        self._visit_all([node.type])
        if node.name:
            # Restrict the search to the header; the body may use the same name.
            header = ast.copy_location(ast.Name(id=node.name), node)
            header.end_lineno, header.end_col_offset = node.body[0].lineno, node.body[0].col_offset
            self._field(header, node.name, last=True)
        self._visit_all(node.body)

    def visit_MatchAs(self, node: ast.MatchAs) -> None:
        """Capture after the nested pattern, without treating wildcard _ as a name."""
        self._visit_all([node.pattern])
        if node.name:
            self._field(node, node.name, last=True)

    def visit_MatchStar(self, node: ast.MatchStar) -> None:
        """Record a starred pattern capture, if it is not the wildcard form."""
        if node.name:
            self._field(node, node.name, last=True)

    def visit_MatchMapping(self, node: ast.MatchMapping) -> None:
        """Visit key expressions/patterns and capture the optional **rest binding."""
        self._visit_all([*node.keys, *node.patterns])
        if node.rest:
            self._field(node, node.rest, last=True)

    def visit_TypeAlias(self, node: ast.TypeAlias) -> None:
        """Fail explicitly for PEP 695 lazy annotation scopes instead of misbinding."""
        raise UnsupportedSyntaxError(f"Type-alias annotation scopes are not supported at line {node.lineno}, column {node.col_offset}")

    def _owner(self, scope: _Scope, name: str) -> _Scope | None:
        """Resolve a canonical name after collection; return scope or external None.

        Own-class locals are included, but a method/comprehension cannot capture
        an intervening class namespace. Global declarations in enclosing
        functions also stop free-variable lookup. Runtime fallback from unbound
        class locals is outside this static lexical contract. The implicit
        __class__ cell is a compiler-created binding, not a source declaration.
        """
        if name in scope.globals:
            return self.root if name in self.root.locals else None
        if name in scope.locals:
            return scope
        ancestor = scope.parent
        while ancestor is not None:
            if ancestor.kind == "class":
                if name == "__class__":
                    return None
            else:
                if name in ancestor.globals:
                    return self.root if name in self.root.locals else None
                if name in ancestor.locals:
                    return ancestor
            ancestor = ancestor.parent
        return None

    def resolve(self) -> tuple[VariableBinding, ...]:
        """Resolve collected events into ordinary binding records in source order.

        No arguments; call only after visiting the complete module. Each returned
        record has all source occurrences, opaque stable IDs, and no cipher bits.
        ``decode`` consumes these records and assigns symbol/frame metadata.
        Unresolved external reads/declarations are omitted. Compiler-generated
        bindings with no source declaration do not contribute records.
        """
        for scope in self.scopes:
            self.root.locals.update(scope.locals & scope.globals)
            scope.locals.difference_update(scope.globals | scope.nonlocals)
        # Module-level global declarations do not remove module-owned bindings.
        for event in self.events:
            if event.occurrence.kind not in ("read", "declaration") and event.canonical_name in event.scope.globals:
                self.root.locals.add(event.canonical_name)
        grouped: dict[tuple[_Scope, str], list[_NameOccurrence]] = defaultdict(list)
        for event in self.events:
            owner = self._owner(event.scope, event.canonical_name)
            if owner is not None:
                grouped[(owner, event.canonical_name)].append(event)
        bindings = []
        for (scope, canonical), events in grouped.items():
            events.sort(key=lambda event: (event.occurrence.span.line, event.occurrence.span.column))
            bindings.append(
                VariableBinding(
                    binding_id=f"{scope.scope_id}:{canonical}",
                    scope_id=scope.scope_id,
                    name=events[0].name,
                    occurrences=tuple(event.occurrence for event in events),
                )
            )
        return tuple(sorted(bindings, key=lambda binding: (binding.occurrences[0].span.line, binding.occurrences[0].span.column)))


def collect_bindings(code: str, tree: ast.Module) -> tuple[VariableBinding, ...]:
    """Extract source bindings from an already compiled/validated Python module.

    ``code`` is the original Unicode source and ``tree`` its unmodified AST.
    Return ordinary VariableBinding records, including all occurrences in source
    order; the decoder adds cipher metadata afterward. This function never
    executes source or imports its dependencies. Unsupported syntax raises
    UnsupportedSyntaxError with location; callers must not treat it as absence.
    """
    collector = _Collector(code)
    collector.visit(tree)
    return collector.resolve()
