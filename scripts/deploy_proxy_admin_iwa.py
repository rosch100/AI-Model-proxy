#!/usr/bin/env python3
"""Install the intranet-only SPNEGO ingress for proxy.altanis.de/admin."""

from __future__ import annotations

import argparse
import ipaddress
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

PROXY_HOSTNAME = "proxy.altanis.de"
KERBEROS_REALM = "ALTANIS.DE"
ROUTER_ADDRESS = "192.168.20.31"
PROXY_PRIVATE_ADDRESS = "192.168.20.11"
DOMAIN_CONTROLLER_ADDRESS = "192.168.253.5"
KERBEROS_PORT = 88
PRIVATE_INTERFACE = "ens224"
ROUTER_WIREGUARD_INTERFACE = "wg2"
ROUTER_LAN_INTERFACE = "ens256"
NGINX_GSS_PACKAGE = "libnginx-mod-http-auth-spnego"
KERBEROS_TOOLS_PACKAGE = "krb5-user"
NGINX_SERVICE_GROUP = "www-data"
NGINX_SITE = Path("/etc/nginx/sites-available/proxy.altanis.de")
NGINX_GEO_CONFIG = Path("/etc/nginx/conf.d/proxy-admin-iwa-geo.conf")
NGINX_LOCATION_CONFIG = Path(
    "/etc/nginx/snippets/proxy-altanis-admin-iwa-locations.conf"
)
NETWORK_CONFIG = Path("/etc/network/interfaces.d/ens224.cfg")
KRB5_CONFIG = Path("/etc/krb5.conf")
KRB5_DROP_IN = Path("/etc/krb5.conf.d/proxy-admin-iwa.conf")
KEYTAB_PATH = Path("/etc/nginx/proxy-admin.keytab")
ROUTE_PREFIX = f"{DOMAIN_CONTROLLER_ADDRESS}/32"
ROUTE_COMMAND = (
    f"up ip route replace {ROUTE_PREFIX} via {ROUTER_ADDRESS} dev {PRIVATE_INTERFACE}"
)
KEYTAB_MODE = 0o640
TRUSTED_IPV4_NETWORKS = tuple(
    ipaddress.ip_network(value)
    for value in ("10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16")
)
TRUSTED_IPV6_NETWORK = ipaddress.ip_network("fc00::/7")


class DeploymentError(RuntimeError):
    """A required precondition or deployment operation failed."""


@dataclass(frozen=True)
class FileSnapshot:
    """File content and metadata needed to roll back an incomplete deployment."""

    content: str | None
    mode: int
    uid: int
    gid: int


def normalize_trusted_networks(networks: Sequence[str]) -> tuple[str, ...]:
    """Accept unique RFC1918 or unique-local CIDRs for trusted admin clients."""
    if not networks:
        raise ValueError("At least one trusted intranet/VPN CIDR is required.")
    normalized: list[str] = []
    for value in networks:
        try:
            network = ipaddress.ip_network(value, strict=True)
        except ValueError as exc:
            raise ValueError(f"Invalid trusted CIDR: {value}") from exc
        if network.version == 4:
            trusted = any(
                network.subnet_of(allowed) for allowed in TRUSTED_IPV4_NETWORKS
            )
        else:
            trusted = network.subnet_of(TRUSTED_IPV6_NETWORK)
        if not trusted:
            raise ValueError(f"Trusted CIDR must be RFC1918 or unique-local: {value}")
        canonical = str(network)
        if canonical in normalized:
            raise ValueError(f"Duplicate trusted CIDR: {canonical}")
        normalized.append(canonical)
    return tuple(normalized)


def validate_service_principal(principal: str) -> str:
    """Require the exact HTTP SPN and realm for the public service hostname."""
    expected = f"HTTP/{PROXY_HOSTNAME}@{KERBEROS_REALM}"
    if principal != expected:
        raise ValueError(f"Service principal must be exactly {expected}.")
    return principal


def render_nginx_geo(networks: Sequence[str]) -> str:
    """Render an HTTP-context client network gate, defaulting to deny."""
    normalized = normalize_trusted_networks(networks)
    entries = "\n".join(f"    {network} 1;" for network in normalized)
    return (
        "# Managed by deploy_proxy_admin_iwa.py.\n"
        "geo $proxy_admin_iwa_allowed {\n"
        "    default 0;\n"
        f"{entries}\n"
        "}\n"
    )


def _validate_keytab_path(keytab_path: str) -> Path:
    """Reject paths that could alter the generated Nginx directive syntax."""
    keytab = Path(keytab_path)
    if (
        not keytab.is_absolute()
        or str(keytab) != keytab_path
        or ".." in keytab.parts
        or re.fullmatch(r"/[A-Za-z0-9_./-]+", keytab_path) is None
    ):
        raise ValueError(
            "Keytab path must be a canonical absolute path with safe characters."
        )
    return keytab


def render_nginx_locations(keytab_path: str, principal: str) -> str:
    """Render the protected Flask admin location without changing public proxying."""
    keytab = _validate_keytab_path(keytab_path)
    validate_service_principal(principal)
    service = principal.split("@", maxsplit=1)[0]
    locations = []
    protected_directives = (
        "    satisfy all;",
        "    if ($proxy_admin_iwa_allowed = 0) { return 404; }",
        "    auth_gss on;",
        f"    auth_gss_keytab {keytab};",
        f"    auth_gss_service_name {service};",
        f"    auth_gss_realm {KERBEROS_REALM};",
        "    auth_gss_allow_basic_fallback off;",
        "    proxy_pass http://127.0.0.1:5000;",
        "    proxy_http_version 1.1;",
        "    proxy_set_header Host $host;",
        "    proxy_set_header X-Real-IP $remote_addr;",
        "    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;",
        "    proxy_set_header X-Forwarded-Proto $scheme;",
        '    proxy_set_header Authorization "";',
        '    proxy_set_header Connection "";',
        "    proxy_buffering off;",
        "    proxy_cache off;",
        "    proxy_read_timeout 3600s;",
        "    proxy_send_timeout 3600s;",
    )
    for declaration in ("= /admin", "^~ /admin/"):
        locations.extend((f"location {declaration} {{", *protected_directives, "}", ""))
    return "# Managed by deploy_proxy_admin_iwa.py.\n" + "\n".join(locations)


def _nginx_brace_delta(line: str) -> int:
    depth = 0
    quote = None
    escaped = False
    for character in line:
        if escaped:
            escaped = False
            continue
        if character == "\\":
            escaped = True
            continue
        if quote:
            if character == quote:
                quote = None
            continue
        if character in ('"', "'"):
            quote = character
        elif character == "#":
            break
        elif character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
    return depth


def _nginx_block_end(config: str, opening_brace: int) -> int:
    depth = 0
    quote = None
    escaped = False
    comment = False
    for index in range(opening_brace, len(config)):
        character = config[index]
        if comment:
            if character == "\n":
                comment = False
            continue
        if escaped:
            escaped = False
            continue
        if character == "\\":
            escaped = True
            continue
        if quote:
            if character == quote:
                quote = None
            continue
        if character in ('"', "'"):
            quote = character
        elif character == "#":
            comment = True
        elif character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0:
                return index + 1
    raise ValueError("Nginx server block has unmatched braces.")


def _nginx_statements(
    config: str,
) -> list[tuple[int, int, tuple[str, ...], bool]]:
    statements = []
    arguments = []
    argument = []
    statement_offset = None
    depth = 0
    quote = None
    escaped = False
    comment = False

    def finish_argument() -> None:
        if argument:
            arguments.append("".join(argument))
            argument.clear()

    def finish_statement(is_block: bool) -> None:
        nonlocal statement_offset
        finish_argument()
        if arguments:
            statements.append((depth, statement_offset, tuple(arguments), is_block))
            arguments.clear()
        statement_offset = None

    index = 0
    while index < len(config):
        character = config[index]
        if comment:
            if character == "\n":
                comment = False
            index += 1
            continue
        if escaped:
            argument.append(character)
            escaped = False
            index += 1
            continue
        if character == "\\":
            if statement_offset is None:
                statement_offset = index
            argument.append(character)
            escaped = True
            index += 1
            continue
        if quote:
            if character == quote:
                quote = None
            else:
                argument.append(character)
            index += 1
            continue
        if character in ('"', "'"):
            if statement_offset is None:
                statement_offset = index
            quote = character
        elif character == "#":
            finish_argument()
            comment = True
        elif character.isspace():
            finish_argument()
        elif character == ";":
            finish_statement(False)
        elif character == "{":
            finish_statement(True)
            depth += 1
        elif character == "}":
            finish_argument()
            if arguments or statement_offset is not None or depth == 0:
                raise ValueError("Nginx server block has unmatched braces.")
            depth -= 1
        else:
            if statement_offset is None:
                statement_offset = index
            argument.append(character)
        index += 1

    if quote or escaped or depth != 0 or arguments or argument:
        raise ValueError("Nginx server block has an incomplete statement.")
    return statements


def _server_statements(
    config: str,
) -> list[tuple[int, int, tuple[str, ...], bool]]:
    return [statement for statement in _nginx_statements(config) if statement[0] == 0]


def _location_declarations(
    statements: Sequence[tuple[int, int, tuple[str, ...], bool]],
) -> list[tuple[int, str | None, str, tuple[str, ...]]]:
    declarations = []
    for _, offset, arguments, is_block in statements:
        if arguments[0] != "location":
            continue
        if not is_block or len(arguments) not in (2, 3):
            raise ValueError("Could not safely parse an Nginx location declaration.")
        modifier = arguments[1] if len(arguments) == 3 else None
        path = arguments[2] if len(arguments) == 3 else arguments[1]
        if path.startswith(('"', "'")) and path[-1:] == path[:1]:
            path = path[1:-1]
        if modifier not in (None, "=", "^~", "~", "~*"):
            raise ValueError("Could not safely parse an Nginx location declaration.")
        declarations.append((offset, modifier, path, arguments))
    return declarations


def _unquote_nginx_argument(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
        return value[1:-1]
    return value


def _validate_no_unprotected_admin_routes(server_body: str) -> None:
    if "\\" in server_body:
        raise ValueError("Nginx escape sequences prevent proving admin isolation.")
    for depth, _, arguments, is_block in _nginx_statements(server_body):
        directive = arguments[0]
        values = arguments[1:]
        if directive == "location" and depth > 0:
            raise ValueError("Nested Nginx locations prevent proving admin isolation.")
        if directive == "location" and not is_block:
            raise ValueError("Could not safely parse an Nginx location declaration.")
        if directive in ("if", "return", "rewrite", "try_files", "error_page"):
            raise ValueError(
                f"Nginx {directive} directives prevent proving admin isolation."
            )
        if directive == "include":
            if values not in (
                (str(NGINX_LOCATION_CONFIG),),
                ("/etc/letsencrypt/options-ssl-nginx.conf",),
            ):
                raise ValueError(
                    "External Nginx includes prevent proving admin route isolation."
                )
        elif directive == "proxy_pass":
            target = " ".join(_unquote_nginx_argument(value) for value in values)
            if "$" in target or re.search(
                r"^[a-z][a-z0-9+.-]*://[^/?#]+(?:/|\?|#)", target
            ):
                raise ValueError(
                    "Nginx proxy_pass can rewrite the upstream URI outside protected locations."
                )


def _proxies_to_flask(
    statements: Sequence[tuple[int, int, tuple[str, ...], bool]],
) -> bool:
    """Identify the Flask upstream, including URI suffixes for later validation."""
    return any(
        not is_block
        and len(arguments) == 2
        and arguments[0] == "proxy_pass"
        and re.match(r"^http://127\.0\.0\.1:5000(?:$|[/?#])", arguments[1])
        for _, _, arguments, is_block in statements
    )


def _validate_unique_active_proxy_server(config: str, site_path: Path) -> None:
    """Require one active Flask proxy block from the exact managed file content."""
    sections = re.split(r"(?m)^# configuration file (.+):\s*$", config)
    expected_content = site_path.read_text(encoding="utf-8").strip("\n")
    if sections[0].strip():
        raise DeploymentError("Could not safely parse nginx -T configuration output.")
    matching_sources = []
    managed_sections = []
    for index in range(1, len(sections), 2):
        source_path = Path(sections[index])
        section = sections[index + 1]
        try:
            source_content = source_path.read_text(encoding="utf-8")
        except OSError as exc:
            raise DeploymentError(
                f"Could not verify active Nginx config source {source_path}: {exc}"
            ) from exc
        if section.strip("\n") != source_content.strip("\n"):
            raise DeploymentError(
                f"nginx -T output does not match source file {source_path}."
            )
        if source_path.resolve() == site_path.resolve():
            managed_sections.append(section.strip("\n"))
        for _, offset, arguments, is_block in _nginx_statements(section):
            if arguments[0] != "server" or not is_block:
                continue
            if arguments != ("server",):
                raise ValueError("Could not safely parse an Nginx server declaration.")
            opening = section.find("{", offset)
            end = _nginx_block_end(section, opening)
            body = section[opening + 1 : end - 1]
            statements = _nginx_statements(body)
            server_names = (
                name
                for depth, _, statement, is_block_statement in statements
                if depth == 0
                and not is_block_statement
                and statement[0] == "server_name"
                for name in statement[1:]
            )
            if any(
                name.casefold() == PROXY_HOSTNAME.casefold() for name in server_names
            ) and _proxies_to_flask(statements):
                matching_sources.append(source_path.resolve())

    if (
        matching_sources != [site_path.resolve()]
        or len(managed_sections) != 1
        or managed_sections[0].strip() != expected_content
    ):
        raise DeploymentError(
            "Expected exactly one active proxy.altanis.de Flask proxy server block "
            f"from the unmodified {site_path} configuration."
        )


def insert_admin_locations(site_config: str) -> str:
    """Insert protected admin locations into the matching Nginx server block."""
    server_blocks = []
    for _, offset, arguments, is_block in _nginx_statements(site_config):
        if arguments[0] != "server" or not is_block:
            continue
        if arguments != ("server",):
            raise ValueError("Could not safely parse an Nginx server declaration.")
        opening = site_config.find("{", offset)
        end = _nginx_block_end(site_config, opening)
        body = site_config[opening + 1 : end - 1]
        if "\\" in body:
            raise ValueError(
                "Nginx escape sequences prevent safely identifying server blocks."
            )
        server_blocks.append((offset, opening, end))

    matching_blocks = []
    for _, opening, end in server_blocks:
        body = site_config[opening + 1 : end - 1]
        statements = _nginx_statements(body)
        server_names = (
            name
            for depth, _, arguments, is_block in statements
            if depth == 0 and not is_block and arguments[0] == "server_name"
            for name in arguments[1:]
        )
        if any(
            name.casefold() == PROXY_HOSTNAME.casefold() for name in server_names
        ) and _proxies_to_flask(statements):
            _validate_no_unprotected_admin_routes(body)
            matching_blocks.append((opening, body, statements))
    if len(matching_blocks) != 1:
        raise ValueError("Expected exactly one Flask proxy server block for proxy.altanis.de.")

    opening, body, statements = matching_blocks[0]
    catch_all = []
    conflicts = []
    for relative_offset, modifier, path, declaration in _location_declarations(
        _server_statements(body)
    ):
        if path == "/" and modifier in (None, "^~"):
            line_end = body.find("\n", relative_offset)
            original_line = body[relative_offset : line_end if line_end >= 0 else None]
            catch_all.append((relative_offset, original_line))
        if path == "/admin" or path.startswith("/admin/"):
            if modifier in (None, "=", "^~"):
                conflicts.append(" ".join(declaration))
        elif modifier in ("~", "~*"):
            conflicts.append(" ".join(declaration))
    if len(catch_all) != 1:
        raise ValueError("Expected exactly one public location / block.")
    if conflicts:
        raise ValueError(
            "Existing Nginx locations overlap protected admin paths: "
            + ", ".join(conflicts)
        )

    existing_includes = [
        (offset, arguments)
        for depth, offset, arguments, is_block in statements
        if depth == 0
        and not is_block
        and arguments == ("include", str(NGINX_LOCATION_CONFIG))
    ]
    marker = "proxy-altanis-admin-iwa-locations.conf"
    if len(existing_includes) > 1 or (
        marker in site_config
        and not any(
            arguments[1:] == (str(NGINX_LOCATION_CONFIG),)
            for _, arguments in existing_includes
        )
    ):
        raise ValueError("Unexpected admin IWA include in Nginx site.")
    catch_all_offset, _ = catch_all[0]
    if existing_includes:
        if (
            len(existing_includes[0][1]) != 2
            or existing_includes[0][1][1] != str(NGINX_LOCATION_CONFIG)
            or existing_includes[0][0] >= catch_all_offset
        ):
            raise ValueError(
                "Admin IWA include must precede the public catch-all location."
            )
        return site_config

    location_offset = catch_all_offset
    line_start = body.rfind("\n", 0, location_offset) + 1
    line_prefix = body[line_start:location_offset]
    indent = re.match(r"[ \t]*", line_prefix).group(0)
    if line_prefix.strip():
        include = f"\n{indent}include {NGINX_LOCATION_CONFIG};\n{indent}"
        insert_at = opening + 1 + location_offset
    else:
        include = f"{indent}include {NGINX_LOCATION_CONFIG};\n"
        insert_at = opening + 1 + line_start
    return site_config[:insert_at] + include + site_config[insert_at:]


def validate_router_kerberos_address(address: str) -> str:
    """Require the router's verified private IPv4 address on its KDC-facing path."""
    try:
        parsed = ipaddress.ip_address(address)
    except ValueError as exc:
        raise ValueError(f"Invalid router Kerberos IPv4 address: {address}") from exc
    if parsed.version != 4 or not any(
        parsed in network for network in TRUSTED_IPV4_NETWORKS
    ):
        raise ValueError("Router Kerberos address must be an RFC1918 IPv4 address.")
    return str(parsed)


def render_router_rules(
    router_kerberos_address: str,
    *,
    forward_chain_handle: int,
    nat_chain_handle: int,
) -> str:
    """Render KDC rules positioned before reviewed router-chain rules."""
    if forward_chain_handle <= 0 or nat_chain_handle <= 0:
        raise ValueError("Router chain insertion handles must be positive integers.")
    snat_address = validate_router_kerberos_address(router_kerberos_address)
    tcp_selector = (
        f'iifname "{ROUTER_LAN_INTERFACE}" oifname "{ROUTER_WIREGUARD_INTERFACE}" '
        f"ip saddr {PROXY_PRIVATE_ADDRESS} ip daddr {DOMAIN_CONTROLLER_ADDRESS} "
        f"tcp dport {KERBEROS_PORT}"
    )
    udp_selector = tcp_selector.replace("tcp dport", "udp dport")
    return "\n".join(
        (
            "# Kerberos KDC access for the proxy administrator SPNEGO acceptor.",
            f"insert rule inet wireguard_filter forward handle {forward_chain_handle} "
            f"{tcp_selector} counter accept",
            f"insert rule inet wireguard_filter forward handle {forward_chain_handle} "
            f"{udp_selector} counter accept",
            f"insert rule inet wireguard_filter forward handle {forward_chain_handle} "
            f'iifname "{ROUTER_WIREGUARD_INTERFACE}" '
            f'oifname "{ROUTER_LAN_INTERFACE}" ip saddr {DOMAIN_CONTROLLER_ADDRESS} '
            f"ip daddr {PROXY_PRIVATE_ADDRESS} tcp sport {KERBEROS_PORT} "
            "ct state established,related counter accept",
            f"insert rule inet wireguard_filter forward handle {forward_chain_handle} "
            f'iifname "{ROUTER_WIREGUARD_INTERFACE}" '
            f'oifname "{ROUTER_LAN_INTERFACE}" ip saddr {DOMAIN_CONTROLLER_ADDRESS} '
            f"ip daddr {PROXY_PRIVATE_ADDRESS} udp sport {KERBEROS_PORT} "
            "ct state established,related counter accept",
            f"insert rule ip wireguard_nat4 postrouting handle {nat_chain_handle} "
            f"ip saddr {PROXY_PRIVATE_ADDRESS} ip daddr {DOMAIN_CONTROLLER_ADDRESS} "
            f'tcp dport {KERBEROS_PORT} oifname "{ROUTER_WIREGUARD_INTERFACE}" '
            f"snat to {snat_address}",
            f"insert rule ip wireguard_nat4 postrouting handle {nat_chain_handle} "
            f"ip saddr {PROXY_PRIVATE_ADDRESS} ip daddr {DOMAIN_CONTROLLER_ADDRESS} "
            f'udp dport {KERBEROS_PORT} oifname "{ROUTER_WIREGUARD_INTERFACE}" '
            f"snat to {snat_address}",
            "",
        )
    )


def render_krb5_config() -> str:
    """Render the AD KDC mapping without changing host-wide Kerberos defaults."""
    return (
        "# Managed by deploy_proxy_admin_iwa.py.\n"
        "[realms]\n"
        f"    {KERBEROS_REALM} = {{\n"
        f"        kdc = {DOMAIN_CONTROLLER_ADDRESS}\n"
        "    }\n"
    )


def _network_config_with_route(config: str) -> str:
    """Add the AD host route to the existing static ens224 stanza."""
    lines = config.splitlines(keepends=True)
    stanza = re.compile(r"[ \t]*iface\s+ens224\s+inet\s+static[ \t]*")
    starts = [
        index
        for index, line in enumerate(lines)
        if stanza.fullmatch(line.rstrip("\r\n"))
    ]
    if len(starts) != 1:
        raise DeploymentError("Expected one static ens224 interface in its config.")
    start = starts[0]
    end = next(
        (
            index
            for index in range(start + 1, len(lines))
            if re.match(r"^[ \t]*(?:auto|allow-|iface|mapping|source)\b", lines[index])
        ),
        len(lines),
    )
    interface_lines = lines[start + 1 : end]
    route = re.compile(r"^\s*up\s+ip\s+route\s+replace\s+192\.168\.253\.5/32\b")
    existing = [line for line in interface_lines if route.match(line)]
    if existing:
        if len(existing) != 1 or existing[0].strip() != ROUTE_COMMAND:
            raise DeploymentError("ens224 already has a conflicting route directive.")
        return config

    directive = re.compile(
        r"^(?P<indent>[ \t]+)(?:address|netmask|gateway|broadcast|network|"
        r"dns-[\w-]+|up|down)\b"
    )
    indentation = next(
        (
            match.group("indent")
            for line in interface_lines
            if (match := directive.match(line)) is not None
        ),
        " " * (len(lines[start]) - len(lines[start].lstrip()) + 4),
    )
    insertion = end
    while insertion > start + 1 and not lines[insertion - 1].strip():
        insertion -= 1
    if insertion > start + 1 and not lines[insertion - 1].endswith(("\n", "\r")):
        lines[insertion - 1] += "\n"
    lines.insert(insertion, f"{indentation}{ROUTE_COMMAND}\n")
    return "".join(lines)


def _run(command: Sequence[str], *, env: dict[str, str] | None = None) -> str:
    result = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise DeploymentError(
            f"Command failed ({result.returncode}): {command[0]}: {detail}"
        )
    return result.stdout


def _snapshot(path: Path) -> FileSnapshot:
    if path.is_symlink():
        raise DeploymentError(f"Expected a regular configuration file: {path}")
    if not path.exists():
        return FileSnapshot(None, 0o644, 0, 0)
    if not path.is_file():
        raise DeploymentError(f"Expected a regular configuration file: {path}")
    metadata = path.stat()
    return FileSnapshot(
        path.read_text(encoding="utf-8"),
        stat.S_IMODE(metadata.st_mode),
        metadata.st_uid,
        metadata.st_gid,
    )


def _atomic_write(path: Path, content: str, snapshot: FileSnapshot) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.chown(temporary, snapshot.uid, snapshot.gid)
        os.chmod(temporary, snapshot.mode)
        os.replace(temporary, path)
    except BaseException:  # noqa: B902 — remove the temp file even on interruption.
        Path(temporary).unlink(missing_ok=True)
        raise


def _restore(path: Path, snapshot: FileSnapshot) -> None:
    if snapshot.content is None:
        path.unlink(missing_ok=True)
        return
    _atomic_write(path, snapshot.content, snapshot)


def _ensure_packages(install_packages: bool) -> None:
    packages = (NGINX_GSS_PACKAGE, KERBEROS_TOOLS_PACKAGE)
    missing = []
    for package in packages:
        installed = subprocess.run(
            ["dpkg-query", "-W", "-f=${db:Status-Status}", package],
            check=False,
            capture_output=True,
            text=True,
        )
        if installed.returncode != 0 or installed.stdout.strip() != "installed":
            missing.append(package)
    if not missing:
        return
    if not install_packages:
        raise DeploymentError(
            f"Required packages are missing ({', '.join(missing)}); "
            "rerun with --install-spnego-package."
        )
    environment = os.environ | {"DEBIAN_FRONTEND": "noninteractive"}
    _run(["apt-get", "update"], env=environment)
    _run(["apt-get", "install", "--yes", *missing], env=environment)


def _keytab_group_id() -> int:
    entry = _run(["getent", "group", NGINX_SERVICE_GROUP]).strip()
    parts = entry.split(":")
    if len(parts) < 3 or not parts[2].isdecimal():
        raise DeploymentError(f"Could not resolve the {NGINX_SERVICE_GROUP} group.")
    return int(parts[2])


def _keytab_lists_principal(listing: str, principal: str) -> bool:
    return any(principal in line.split() for line in listing.splitlines())


def _validate_keytab(path: Path, principal: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise DeploymentError(f"Keytab must be a regular file: {path}")
    metadata = path.stat()
    if metadata.st_uid != 0:
        raise DeploymentError("Keytab must be owned by root.")
    if stat.S_IMODE(metadata.st_mode) != KEYTAB_MODE:
        raise DeploymentError("Keytab permissions must be 0640 (root:www-data).")
    if metadata.st_gid != _keytab_group_id():
        raise DeploymentError("Keytab group must be www-data for the Nginx worker.")
    listing = _run(["klist", "-k", "-t", str(path)])
    if not _keytab_lists_principal(listing, principal):
        raise DeploymentError(
            "Keytab does not contain the expected HTTP service principal."
        )


def _route_matches_expected(current: str) -> bool:
    parts = current.split()
    if len(parts) < 5 or parts[1] != "via" or parts[3] != "dev":
        return False
    try:
        destination = ipaddress.ip_network(parts[0], strict=False)
    except ValueError:
        return False
    expected_destination = ipaddress.ip_network(ROUTE_PREFIX)
    return (
        destination == expected_destination
        and parts[2] == ROUTER_ADDRESS
        and parts[4] == PRIVATE_INTERFACE
    )


def _ensure_route() -> bool:
    current = _run(["ip", "-4", "route", "show", "exact", ROUTE_PREFIX]).strip()
    if current:
        if not _route_matches_expected(current):
            raise DeploymentError(f"Conflicting route already exists: {current}")
        return False
    _run(
        [
            "ip",
            "-4",
            "route",
            "add",
            ROUTE_PREFIX,
            "via",
            ROUTER_ADDRESS,
            "dev",
            PRIVATE_INTERFACE,
        ]
    )
    return True


def _ensure_krb5_include(config: str) -> str:
    if re.search(r"(?m)^\s*includedir\s+/etc/krb5\.conf\.d/?\s*$", config):
        return config
    return "includedir /etc/krb5.conf.d\n" + config


def _require_host_identity() -> None:
    hostname = _run(["hostname", "-f"]).strip()
    if hostname != PROXY_HOSTNAME:
        raise DeploymentError(
            f"Installer must run on {PROXY_HOSTNAME}; found {hostname}."
        )


def _verify_keytab(path: Path, principal: str) -> None:
    """Check credentials using a principal matching the AD account userPrincipalName."""
    with tempfile.TemporaryDirectory(
        prefix="proxy-admin-iwa-", dir="/run"
    ) as cache_dir:
        cache_name = f"FILE:{cache_dir}/ccache"
        environment = os.environ | {"KRB5CCNAME": cache_name}
        _run(["kinit", "-k", "-t", str(path), principal], env=environment)


def _rollback_changes(
    changed_paths: Sequence[Path],
    snapshots: dict[Path, FileSnapshot],
    route_added: bool,
) -> tuple[str, ...]:
    """Restore files/runtime before removing a deployment-only route."""
    errors = []
    failed_restore_paths = set()
    for path in reversed(changed_paths):
        try:
            _restore(path, snapshots[path])
        except (OSError, DeploymentError) as exc:
            failed_restore_paths.add(path)
            errors.append(f"{path}: {exc}")

    nginx_paths = {NGINX_GEO_CONFIG, NGINX_LOCATION_CONFIG, NGINX_SITE}
    nginx_changes = nginx_paths.intersection(changed_paths)
    if nginx_changes and not nginx_paths.intersection(failed_restore_paths):
        try:
            _run(["nginx", "-t"])
            _run(["systemctl", "reload", "nginx"])
        except (DeploymentError, OSError) as exc:
            errors.append(f"restored Nginx runtime: {exc}")

    if route_added and not errors:
        try:
            _run(["ip", "-4", "route", "del", ROUTE_PREFIX])
        except (DeploymentError, OSError) as exc:
            errors.append(f"route {ROUTE_PREFIX}: {exc}")
    elif route_added:
        errors.append(f"route {ROUTE_PREFIX}: retained because rollback is incomplete")
    return tuple(errors)


def install(args: argparse.Namespace) -> None:
    """Install packages, private route, Kerberos realm, and protected Nginx route."""
    if os.geteuid() != 0:
        raise DeploymentError("Run this installer as root on proxy.altanis.de.")
    _require_host_identity()
    networks = normalize_trusted_networks(args.trusted_network)
    principal = validate_service_principal(args.service_principal)
    try:
        keytab = _validate_keytab_path(args.keytab)
    except ValueError as exc:
        raise DeploymentError(str(exc)) from exc
    for required in ("dpkg-query", "nginx", "ip", "getent"):
        if shutil.which(required) is None:
            raise DeploymentError(f"Required command is unavailable: {required}")
    if not keytab.is_file() or keytab.is_symlink():
        raise DeploymentError(
            f"Provision the AD service keytab securely before deployment: {keytab}"
        )
    try:
        _validate_unique_active_proxy_server(_run(["nginx", "-T"]), NGINX_SITE)
    except ValueError as exc:
        raise DeploymentError(str(exc)) from exc

    paths = (
        NETWORK_CONFIG,
        KRB5_CONFIG,
        KRB5_DROP_IN,
        NGINX_GEO_CONFIG,
        NGINX_LOCATION_CONFIG,
        NGINX_SITE,
    )
    snapshots = {path: _snapshot(path) for path in paths}
    network_content = snapshots[NETWORK_CONFIG].content
    krb5_content = snapshots[KRB5_CONFIG].content
    site_content = snapshots[NGINX_SITE].content
    if network_content is None or krb5_content is None or site_content is None:
        raise DeploymentError("Required host configuration file is missing.")
    updated_network = _network_config_with_route(network_content)
    _ensure_krb5_include(krb5_content)
    krb5_dropin = render_krb5_config()
    if snapshots[KRB5_DROP_IN].content not in (None, krb5_dropin):
        raise DeploymentError(f"Refusing to overwrite unmanaged file: {KRB5_DROP_IN}")
    updated_site = insert_admin_locations(site_content)

    _ensure_packages(args.install_spnego_package)
    for required in ("klist", "kinit"):
        if shutil.which(required) is None:
            raise DeploymentError(f"Required command is unavailable: {required}")
    _validate_keytab(keytab, principal)

    route_added = False
    changed_paths: list[Path] = []
    try:
        route_added = _ensure_route()
        updated_files = (
            (NETWORK_CONFIG, updated_network),
            (KRB5_DROP_IN, krb5_dropin),
            (KRB5_CONFIG, _ensure_krb5_include(krb5_content)),
            (NGINX_GEO_CONFIG, render_nginx_geo(networks)),
            (NGINX_LOCATION_CONFIG, render_nginx_locations(str(keytab), principal)),
            (NGINX_SITE, updated_site),
        )
        for path, content in updated_files:
            if snapshots[path].content == content:
                continue
            changed_paths.append(path)
            _atomic_write(path, content, snapshots[path])

        _verify_keytab(keytab, principal)
        _run(["nginx", "-t"])
        _run(["systemctl", "reload", "nginx"])
    except BaseException as deployment_error:  # noqa: B902 — rollback on interruption.
        rollback_errors = _rollback_changes(changed_paths, snapshots, route_added)
        if rollback_errors:
            details = "; ".join(rollback_errors)
            raise DeploymentError(
                f"Deployment failed ({deployment_error}); rollback was incomplete: {details}"
            ) from deployment_error
        raise


def main(argv: Sequence[str] | None = None) -> int:
    """Run the installer CLI or print validated router rules."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trusted-network",
        action="append",
        default=[],
        help="Private CIDR whose clients may use /admin (repeatable).",
    )
    parser.add_argument(
        "--service-principal",
        default=f"HTTP/{PROXY_HOSTNAME}@{KERBEROS_REALM}",
    )
    parser.add_argument("--keytab", default=str(KEYTAB_PATH))
    parser.add_argument(
        "--install-spnego-package",
        action="store_true",
        help="Install Debian's Nginx SPNEGO module and Kerberos tools if missing.",
    )
    parser.add_argument(
        "--render-router-rules",
        action="store_true",
        help="Print the least-privilege nftables rules for the WireGuard router and exit.",
    )
    parser.add_argument(
        "--router-kerberos-address",
        help="Verified private IPv4 address used by the router to reach the AD KDC.",
    )
    parser.add_argument(
        "--forward-chain-handle",
        type=int,
        help="Reviewed nftables rule handle before which forward rules are inserted.",
    )
    parser.add_argument(
        "--nat-chain-handle",
        type=int,
        help="Reviewed nftables rule handle before which SNAT rules are inserted.",
    )
    args = parser.parse_args(argv)
    if args.render_router_rules:
        if args.router_kerberos_address is None:
            parser.error(
                "--router-kerberos-address is required with --render-router-rules."
            )
        if args.forward_chain_handle is None or args.nat_chain_handle is None:
            parser.error(
                "--forward-chain-handle and --nat-chain-handle are required "
                "with --render-router-rules."
            )
        print(
            render_router_rules(
                args.router_kerberos_address,
                forward_chain_handle=args.forward_chain_handle,
                nat_chain_handle=args.nat_chain_handle,
            ),
            end="",
        )
        return 0
    try:
        install(args)
    except (DeploymentError, OSError, ValueError) as exc:
        print(f"Proxy admin IWA installation failed: {exc}", file=sys.stderr)
        return 1
    print("Proxy /admin now requires an intranet/VPN source and Kerberos SPNEGO.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
