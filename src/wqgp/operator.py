"""OperatorDef: a BRAIN operator signature.

Design decisions vs. the reference project (VIPspaceman/wq):
- `arity` counts DATA arguments only. Parameters (windows, lookbacks, ...)
  are stored separately in `param_domains`. The reference project conflates
  them into one arity, which corrupts every downstream type check once a
  parametric operator like ts_mean appears. See DECISIONS.md D-007.
- `associative` does NOT imply `commutative`. We do NOT enforce that
  implication. Counterexample: matrix multiplication. See DECISIONS.md D-008.
- `input_types` supports per-position unions (MATRIX | VECTOR), normalized
  to frozensets in __post_init__. See DECISIONS.md D-009.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from wqgp.types import DataType, ParamDomain


# What a caller may pass per argument position: a single DataType, or a
# tuple of acceptable DataTypes (union). Normalized to frozenset internally.
_InputSpec = DataType | tuple[DataType, ...]


@dataclass(frozen=True, slots=True)
class OperatorDef:
    """Immutable descriptor for a single BRAIN operator.

    Invariants (enforced in __post_init__):
      - name is non-empty, no surrounding whitespace
      - input_types is non-empty (arity >= 1)
      - each position resolves to a non-empty set of DataType
      - param_domains: names non-empty, unique, no surrounding whitespace
      - commutative/associative/symmetric require arity >= 2
    """
    name: str
    input_types: tuple[_InputSpec, ...]
    output_type: DataType
    param_domains: (
        tuple[tuple[str, ParamDomain], ...] | dict[str, ParamDomain]
    ) = ()
    commutative: bool = False
    associative: bool = False
    symmetric: bool = False
    deterministic: bool = True

    def __post_init__(self) -> None:
        # --- name ---
        if not self.name or not self.name.strip():
            raise ValueError("operator name must be a non-empty string")
        if self.name != self.name.strip():
            raise ValueError(
                f"operator name has surrounding whitespace: {self.name!r}"
            )

        # --- input_types: non-empty, normalize to tuple[frozenset] ---
        if not self.input_types:
            raise ValueError(
                f"operator {self.name!r} must declare at least one input type"
            )
        normalized: list[frozenset[DataType]] = []
        for pos, spec in enumerate(self.input_types):
            if isinstance(spec, DataType):
                normalized.append(frozenset({spec}))
            elif isinstance(spec, tuple):
                if not spec:
                    raise ValueError(
                        f"operator {self.name!r} position {pos}: "
                        f"empty input-type union"
                    )
                for t in spec:
                    if not isinstance(t, DataType):
                        raise ValueError(
                            f"operator {self.name!r} position {pos}: "
                            f"invalid type {t!r}"
                        )
                normalized.append(frozenset(spec))
            else:
                raise ValueError(
                    f"operator {self.name!r} position {pos}: "
                    f"expected DataType or tuple, got {type(spec).__name__}"
                )
        object.__setattr__(self, "input_types", tuple(normalized))

        # --- param_domains: normalize dict -> tuple of pairs, then validate ---
        if isinstance(self.param_domains, dict):
            object.__setattr__(
                self, "param_domains", tuple(self.param_domains.items())
            )
        seen: set[str] = set()
        for pname, pdom in self.param_domains:
            if not pname or not pname.strip():
                raise ValueError(
                    f"operator {self.name!r}: empty parameter name"
                )
            if pname != pname.strip():
                raise ValueError(
                    f"operator {self.name!r}: parameter name has "
                    f"surrounding whitespace: {pname!r}"
                )
            if pname in seen:
                raise ValueError(
                    f"operator {self.name!r}: duplicate parameter {pname!r}"
                )
            if not isinstance(pdom, ParamDomain):
                raise ValueError(
                    f"operator {self.name!r} param {pname!r}: "
                    f"expected ParamDomain, got {type(pdom).__name__}"
                )
            seen.add(pname)

        # --- flag invariants ---
        # arity here = number of data args, which equals len(input_types).
        arity = len(self.input_types)
        if arity < 2 and (self.commutative or self.associative or self.symmetric):
            flags = []
            if self.commutative:
                flags.append("commutative")
            if self.associative:
                flags.append("associative")
            if self.symmetric:
                flags.append("symmetric")
            raise ValueError(
                f"operator {self.name!r}: arity={arity} but "
                f"{', '.join(flags)} set (requires arity >= 2)"
            )
        # NOTE: we deliberately do NOT enforce associative => commutative.
        # Matmul and string concat are counterexamples. See DECISIONS.md D-008.

    # --- derived views ---

    @property
    def arity(self) -> int:
        """Number of DATA arguments (excludes parameters)."""
        return len(self.input_types)

    @property
    def param_names(self) -> tuple[str, ...]:
        return tuple(name for name, _ in self.param_domains)

    @property
    def total_args(self) -> int:
        """Data arguments + parameters, matching the flat call syntax."""
        return self.arity + len(self.param_domains)

    def param_domain(self, name: str) -> ParamDomain:
        """Look up a parameter domain by name. Raises KeyError if absent."""
        for pname, pdom in self.param_domains:
            if pname == name:
                return pdom
        raise KeyError(
            f"operator {self.name!r} has no parameter {name!r}"
        )

    # --- type checking ---

    def accepts(self, arg_types: tuple[DataType, ...]) -> bool:
        """True iff the given data-argument types fit this operator's signature."""
        if len(arg_types) != self.arity:
            return False
        return all(
            arg in accepted
            for arg, accepted in zip(arg_types, self.input_types)
        )