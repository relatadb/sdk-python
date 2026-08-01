"""Client-side canonical validation — Tier-1 set (#2248).

Pure, dependency-free validators + normalisers for the eight Tier-1 canonical
identifier kinds, ported from ``crates/relata-canonical/src/`` (``email``,
``phone``, ``iban``, ``imei``, ``vin``, ``msisdn``, ``ipv4``, ``ipv6``).

These run **client-side** with no server round-trip — deterministic
checksum/format checks that let callers reject obviously bad input before it
hits the wire. The server's ``relata-canonical`` crate remains the authority;
this module mirrors its logic so both sides agree.

Two functions per kind:

* ``validate_<kind>(value) -> bool``
* ``normalize_<kind>(value) -> str``  (raises :class:`CanonicalError` on reject)

Test vectors are shared across all four SDKs from
``sdks/shared/canonical-vectors.json``.

Scope: Tier-1 only (8 of ~76 canonical kinds). The remaining 65 kinds stay
server-side; typing them client-side is the remaining scope of #2248.
"""

from __future__ import annotations

_MAX_LOCAL = 64
_MAX_DOMAIN = 255
_MAX_LABEL = 63


class CanonicalError(ValueError):
    """Raised by ``normalize_*`` when input fails canonical validation.

    Attributes
    ----------
    kind:
        Machine-friendly tag (``"empty"``, ``"structure"``, ``"checksum"``, …).
    """

    def __init__(self, kind: str, message: str) -> None:
        super().__init__(f"{kind}: {message}")
        self.kind = kind
        self.message = message


# ── Email ────────────────────────────────────────────────────────────────────

def validate_email(value: str) -> bool:
    """Return ``True`` when *value* is a valid email address."""
    try:
        normalize_email(value)
    except CanonicalError:
        return False
    return True


def normalize_email(value: str) -> str:
    """Return the canonical lowercase-domain form of *value*.

    Raises :class:`CanonicalError` on any structural violation.
    """
    if not value:
        raise CanonicalError("empty", "empty input")
    if value.count("@") != 1:
        raise CanonicalError("structure", "must contain exactly one '@'")
    local, domain = value.split("@", 1)
    if not local:
        raise CanonicalError("structure", "empty local part")
    if len(local) > _MAX_LOCAL:
        raise CanonicalError("length", f"local part > {_MAX_LOCAL} chars")
    if local.startswith(".") or local.endswith("."):
        raise CanonicalError("structure", "local part cannot start or end with '.'")
    prev_dot = False
    for ch in local:
        ok = ch.isalnum() and ch.isascii() or ch in ".+-_"
        if not ok:
            raise CanonicalError("invalid_char", f"bad char {ch!r} in local part")
        if ch == ".":
            if prev_dot:
                raise CanonicalError("structure", "consecutive '.' in local part")
            prev_dot = True
        else:
            prev_dot = False
    if not domain:
        raise CanonicalError("structure", "empty domain")
    if len(domain) > _MAX_DOMAIN:
        raise CanonicalError("length", f"domain > {_MAX_DOMAIN} chars")
    labels = domain.split(".")
    if len(labels) < 2:
        raise CanonicalError("structure", "domain must have at least two labels")
    for label in labels:
        if not label:
            raise CanonicalError("structure", "empty label")
        if len(label) > _MAX_LABEL:
            raise CanonicalError("structure", "label > 63 chars")
        if label.startswith("-") or label.endswith("-"):
            raise CanonicalError("structure", "label cannot start or end with '-'")
        if not all(c.isalnum() and c.isascii() or c == "-" for c in label):
            raise CanonicalError("invalid_char", "bad char in domain label")
    tld = labels[-1]
    if len(tld) < 2:
        raise CanonicalError("structure", "TLD < 2 chars")
    if tld.isdigit():
        raise CanonicalError("structure", "all-numeric TLD")
    return f"{local}@{domain.lower()}"


# ── Phone (E.164) ────────────────────────────────────────────────────────────

def validate_phone(value: str) -> bool:
    """Return ``True`` when *value* is a valid E.164 phone number."""
    try:
        normalize_phone(value)
    except CanonicalError:
        return False
    return True


def normalize_phone(value: str) -> str:
    """Return the canonical ``+<digits>`` E.164 form of *value*."""
    if not value:
        raise CanonicalError("empty", "empty input")
    if not value.startswith("+"):
        raise CanonicalError("structure", "missing '+' prefix")
    rest = value[1:]
    if not rest:
        raise CanonicalError("structure", "no digits after '+'")
    if rest.startswith("0"):
        raise CanonicalError("structure", "leading zero (country code starts at 1)")
    if not rest.isdigit():
        raise CanonicalError("invalid_char", "non-digit in phone number")
    if len(rest) > 15:
        raise CanonicalError("structure", "more than 15 digits")
    n = int(rest)  # safe: ≤15 digits fits in 64 bits
    return f"+{n}"


# ── IBAN (mod-97) ────────────────────────────────────────────────────────────

def validate_iban(value: str) -> bool:
    """Return ``True`` when *value* is a valid IBAN (mod-97 checksum)."""
    try:
        normalize_iban(value)
    except CanonicalError:
        return False
    return True


def normalize_iban(value: str) -> str:
    """Return the canonical contiguous uppercased form of *value*."""
    if not value:
        raise CanonicalError("empty", "empty input")
    normalised = "".join(c.upper() for c in value if not c.isspace())
    if not (15 <= len(normalised) <= 34):
        raise CanonicalError("length", "expected 15–34 chars")
    if not normalised[0].isalpha() or not normalised[1].isalpha():
        raise CanonicalError("structure", "country code must be letters")
    if not normalised[2].isdigit() or not normalised[3].isdigit():
        raise CanonicalError("structure", "check digits must be numeric")
    for b in normalised[4:]:
        if not b.isalnum():
            raise CanonicalError("invalid_char", f"non-alphanumeric char {b!r}")
    if not _mod97_valid(normalised):
        raise CanonicalError("checksum", "mod-97 check failed")
    return normalised


def _mod97_valid(normalised: str) -> bool:
    """Pure mod-97 check on an uppercased, contiguous IBAN string."""
    rotated = normalised[4:] + normalised[:4]
    remainder = 0
    for ch in rotated:
        if ch.isdigit():
            val = int(ch)
        elif ch.isupper():
            val = ord(ch) - ord("A") + 10
        else:
            return False
        if val >= 10:  # noqa: SIM108
            remainder = (remainder * 100 + val) % 97
        else:
            remainder = (remainder * 10 + val) % 97
    return remainder == 1


# ── IMEI (Luhn) ──────────────────────────────────────────────────────────────

def validate_imei(value: str) -> bool:
    """Return ``True`` when *value* is a valid IMEI (Luhn checksum)."""
    try:
        normalize_imei(value)
    except CanonicalError:
        return False
    return True


def normalize_imei(value: str) -> str:
    """Return the canonical 15-digit form of *value* (whitespace trimmed)."""
    s = value.strip()
    if len(s) != 15:
        raise CanonicalError("length", f"expected 15 digits, got {len(s)}")
    if not s.isdigit():
        raise CanonicalError("structure", "non-digit in IMEI")
    digits = [int(c) for c in s]
    total = 0
    for i, d in enumerate(digits):
        pos_from_right = 14 - i
        if pos_from_right % 2 == 1:
            doubled = d * 2
            total += doubled - 9 if doubled > 9 else doubled
        else:
            total += d
    if total % 10 != 0:
        raise CanonicalError("checksum", "Luhn check failed")
    return s


# ── VIN (ISO 3779) ───────────────────────────────────────────────────────────

_VIN_TRANSLIT = {
    "A": 1, "J": 1, "B": 2, "K": 2, "S": 2, "C": 3, "L": 3, "T": 3,
    "D": 4, "M": 4, "U": 4, "E": 5, "N": 5, "V": 5, "F": 6, "W": 6,
    "G": 7, "P": 7, "X": 7, "H": 8, "Y": 8, "R": 9, "Z": 9,
}
_VIN_WEIGHTS = [8, 7, 6, 5, 4, 3, 2, 10, 0, 9, 8, 7, 6, 5, 4, 3, 2]


def validate_vin(value: str) -> bool:
    """Return ``True`` when *value* is a valid VIN (ISO 3779 check digit)."""
    try:
        normalize_vin(value)
    except CanonicalError:
        return False
    return True


def normalize_vin(value: str) -> str:
    """Return the canonical 17-char uppercase form of *value*."""
    s = value.strip().upper()
    if len(s) != 17:
        raise CanonicalError("length", f"expected 17 chars, got {len(s)}")
    values = []
    for ch in s:
        if ch.isdigit():
            values.append(int(ch))
        elif ch in _VIN_TRANSLIT:
            values.append(_VIN_TRANSLIT[ch])
        else:
            raise CanonicalError("structure", f"invalid VIN character {ch!r}")
    total = sum(v * w for v, w in zip(values, _VIN_WEIGHTS, strict=False))
    check = total % 11
    expected = "X" if check == 10 else str(check)
    if s[8] != expected:
        raise CanonicalError("checksum", "check-digit mismatch")
    return s


# ── MSISDN ───────────────────────────────────────────────────────────────────

def validate_msisdn(value: str) -> bool:
    """Return ``True`` when *value* is a valid MSISDN (7–15 digits)."""
    try:
        normalize_msisdn(value)
    except CanonicalError:
        return False
    return True


def normalize_msisdn(value: str) -> str:
    """Return the canonical bare-digit form of *value* (leading ``+`` stripped)."""
    stripped = value.strip().lstrip("+")
    if not stripped:
        raise CanonicalError("empty", "empty input")
    if not stripped.isdigit():
        raise CanonicalError("structure", "non-digit character in MSISDN")
    if not (7 <= len(stripped) <= 15):
        raise CanonicalError("length", f"expected 7–15 digits, got {len(stripped)}")
    if stripped[0] == "0":
        raise CanonicalError("structure", "E.164 MSISDN must not start with 0")
    return stripped


# ── IPv4 ─────────────────────────────────────────────────────────────────────

def validate_ipv4(value: str) -> bool:
    """Return ``True`` when *value* is a valid dotted-quad IPv4 address."""
    try:
        normalize_ipv4(value)
    except CanonicalError:
        return False
    return True


def normalize_ipv4(value: str) -> str:
    """Return the canonical ``a.b.c.d`` form of *value* (no leading zeros)."""
    if not value:
        raise CanonicalError("empty", "empty input")
    octets = []
    digit_count = 0
    current = 0
    had_leading_zero = False
    for ch in value:
        if ch == ".":
            if digit_count == 0:
                raise CanonicalError("structure", "empty octet")
            if len(octets) >= 3:
                raise CanonicalError("structure", "too many octets")
            octets.append(current)
            current = 0
            digit_count = 0
            had_leading_zero = False
            continue
        if not ch.isdigit():
            raise CanonicalError("invalid_char", f"non-digit {ch!r}")
        d = int(ch)
        if digit_count == 0 and d == 0:
            had_leading_zero = True
        elif had_leading_zero:
            raise CanonicalError("structure", "leading zero in octet")
        current = current * 10 + d
        digit_count += 1
        if current > 255:
            raise CanonicalError("structure", "octet > 255")
        if digit_count > 3:
            raise CanonicalError("structure", "octet has too many digits")
    if digit_count == 0:
        raise CanonicalError("structure", "trailing dot")
    if len(octets) != 3:
        raise CanonicalError("structure", "wrong number of octets")
    octets.append(current)
    return ".".join(str(o) for o in octets)


# ── IPv6 ─────────────────────────────────────────────────────────────────────

def validate_ipv6(value: str) -> bool:
    """Return ``True`` when *value* is a valid IPv6 address (RFC 4291)."""
    try:
        normalize_ipv6(value)
    except CanonicalError:
        return False
    return True


def parse_ipv6(value: str) -> bytes:
    """Parse an IPv6 address into 16 canonical big-endian bytes.

    Raises :class:`CanonicalError` on any structural violation.
    """
    if not value:
        raise CanonicalError("empty", "empty input")
    if value.count("::") > 1:
        raise CanonicalError("structure", "multiple '::' compressions")

    def _groups(s: str) -> list[int]:
        out: list[int] = []
        for part in s.split(":"):
            if not part:
                raise CanonicalError("structure", "empty group")
            if len(part) > 4:
                raise CanonicalError("structure", "group has > 4 hex digits")
            try:
                out.append(int(part, 16))
            except ValueError as exc:
                raise CanonicalError("invalid_char", f"non-hex in group {part!r}") from exc
        return out

    if "::" in value:
        left, right = value.split("::", 1)
        lg = _groups(left) if left else []
        rg = _groups(right) if right else []
        zeros = 8 - len(lg) - len(rg)
        if zeros < 0:
            raise CanonicalError("structure", "too many groups around '::'")
        if zeros == 0:
            raise CanonicalError("structure", "'::' must compress at least one group")
        groups = lg + [0] * zeros + rg
    else:
        groups = _groups(value)
        if len(groups) != 8:
            raise CanonicalError("structure", "expected 8 groups")

    out = bytearray(16)
    for i, g in enumerate(groups):
        out[i * 2] = (g >> 8) & 0xFF
        out[i * 2 + 1] = g & 0xFF
    return bytes(out)


def display_ipv6(addr: bytes) -> str:
    """Format 16 canonical bytes as RFC 5952 lowercase-compressed text."""
    groups = [int.from_bytes(addr[i * 2:i * 2 + 2], "big") for i in range(8)]
    best_start, best_len = -1, 0
    cur_start, cur_len = -1, 0
    for i, g in enumerate(groups):
        if g == 0:
            if cur_len == 0:
                cur_start = i
            cur_len += 1
            if cur_len > best_len:
                best_len = cur_len
                best_start = cur_start
        else:
            cur_len = 0
    compress = best_len >= 2

    out = ""
    i = 0
    while i < 8:
        if compress and i == best_start:
            out += "::"
            i += best_len
            continue
        if out and not out.endswith(":"):
            out += ":"
        out += f"{groups[i]:x}"
        i += 1
    return out if out else "::"


def normalize_ipv6(value: str) -> str:
    """Return the canonical RFC 5952 lowercase-compressed form of *value*."""
    return display_ipv6(parse_ipv6(value))


__all__ = [
    "CanonicalError",
    "validate_email", "normalize_email",
    "validate_phone", "normalize_phone",
    "validate_iban", "normalize_iban",
    "validate_imei", "normalize_imei",
    "validate_vin", "normalize_vin",
    "validate_msisdn", "normalize_msisdn",
    "validate_ipv4", "normalize_ipv4",
    "validate_ipv6", "normalize_ipv6", "parse_ipv6", "display_ipv6",
]
