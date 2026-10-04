"""Deployment configuration tests for the proxy administrator IWA ingress."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import call, patch

from scripts.deploy_proxy_admin_iwa import NGINX_SITE as NGINX_SITE_PATH
from scripts.deploy_proxy_admin_iwa import (
    DeploymentError,
    FileSnapshot,
    _keytab_lists_principal,
    _network_config_with_route,
    _rollback_changes,
    _route_matches_expected,
    _snapshot,
    _validate_keytab_path,
    _validate_unique_active_proxy_server,
    insert_admin_locations,
    normalize_trusted_networks,
    render_krb5_config,
    render_nginx_geo,
    render_nginx_locations,
    render_router_rules,
    validate_router_kerberos_address,
    validate_service_principal,
)

NGINX_SITE = """server {
    server_name proxy.altanis.de;

    location ^~ /.well-known/acme-challenge/ {
        root /var/www/letsencrypt;
    }

    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Authorization $http_authorization;
    }
}
"""

NGINX_TLS_SITE = NGINX_SITE.replace(
    "    server_name proxy.altanis.de;",
    "    listen 443 ssl;\n"
    "    server_name proxy.altanis.de;\n"
    "    include /etc/letsencrypt/options-ssl-nginx.conf;",
)
NGINX_REDIRECT_SITE = """server {
    listen 80;
    server_name proxy.altanis.de;
    if ($host = proxy.altanis.de) {
        return 301 https://$host$request_uri;
    }
    return 404;
}
"""


class ProxyAdminIwaDeploymentTests(unittest.TestCase):
    """Validate ingress parsing, deployment safety, and rollback behavior."""

    def test_trusted_networks_are_private_and_canonical(self):
        """Normalize allowed private CIDRs without changing their meaning."""
        self.assertEqual(
            normalize_trusted_networks(
                ["192.168.20.0/24", "10.66.1.0/24", "fd42:66:1::/64"]
            ),
            ("192.168.20.0/24", "10.66.1.0/24", "fd42:66:1::/64"),
        )

    def test_public_or_duplicate_trusted_networks_are_rejected(self):
        """Reject public, duplicate, and empty trusted network lists."""
        for networks in (
            ["0.0.0.0/0"],
            ["8.8.8.0/24"],
            ["100.64.0.0/10"],
            ["192.0.2.0/24"],
            ["192.168.20.0/24", "192.168.20.0/24"],
            [],
        ):
            with self.subTest(networks=networks), self.assertRaises(ValueError):
                normalize_trusted_networks(networks)

    def test_service_principal_must_match_proxy_http_service(self):
        """Bind Kerberos credentials to the expected HTTP service principal."""
        self.assertEqual(
            validate_service_principal("HTTP/proxy.altanis.de@ALTANIS.DE"),
            "HTTP/proxy.altanis.de@ALTANIS.DE",
        )
        for principal in (
            "HTTP/other.altanis.de@ALTANIS.DE",
            "host/proxy.altanis.de@ALTANIS.DE",
            "HTTP/proxy.altanis.de@altanis.de",
            "HTTP/proxy.altanis.de",
        ):
            with self.subTest(principal=principal), self.assertRaises(ValueError):
                validate_service_principal(principal)

    def test_keytab_listing_detects_principal_before_encryption_type(self):
        """Match the principal token independently of keytab encryption type."""
        self.assertTrue(
            _keytab_lists_principal(
                "KVNO Timestamp Principal\n"
                "   2 10/03/26 HTTP/proxy.altanis.de@ALTANIS.DE "
                "(aes256-cts-hmac-sha1-96)\n",
                "HTTP/proxy.altanis.de@ALTANIS.DE",
            )
        )
        self.assertFalse(
            _keytab_lists_principal(
                "   2 10/03/26 HTTP/other.altanis.de@ALTANIS.DE "
                "(aes256-cts-hmac-sha1-96)\n",
                "HTTP/proxy.altanis.de@ALTANIS.DE",
            )
        )

    def test_geo_map_defaults_to_deny_and_allows_only_selected_networks(self):
        """Render an allowlist whose default policy denies admin access."""
        rendered = render_nginx_geo(("192.168.20.0/24", "10.66.1.0/24"))
        self.assertIn("default 0;", rendered)
        self.assertIn("192.168.20.0/24 1;", rendered)
        self.assertIn("10.66.1.0/24 1;", rendered)

    def test_keytab_path_rejects_nginx_syntax_injection(self):
        """Reject path values that could inject Nginx directives."""
        self.assertEqual(
            _validate_keytab_path("/etc/nginx/proxy-admin.keytab"),
            Path("/etc/nginx/proxy-admin.keytab"),
        )
        for path in (
            "relative.keytab",
            "/etc/nginx/proxy-admin.keytab; return 200",
            "/etc/nginx/proxy-admin.keytab\\nreturn 200;",
            "/etc/nginx/proxy-admin.keytab\nreturn 200;",
            "/etc/nginx/../tmp/keytab",
        ):
            with self.subTest(path=path), self.assertRaises(ValueError):
                _validate_keytab_path(path)

    def test_admin_locations_guard_admin_and_keep_bearer_header_private(self):
        """Protect admin routes with SPNEGO and strip bearer credentials."""
        rendered = render_nginx_locations(
            "/etc/nginx/proxy-admin.keytab",
            "HTTP/proxy.altanis.de@ALTANIS.DE",
        )
        self.assertIn("location = /admin {", rendered)
        self.assertIn("location ^~ /admin/ {", rendered)
        self.assertEqual(rendered.count("auth_gss on;"), 2)
        self.assertEqual(rendered.count("satisfy all;"), 2)
        self.assertEqual(
            rendered.count("if ($proxy_admin_iwa_allowed = 0) { return 404; }"), 2
        )
        self.assertEqual(rendered.count('proxy_set_header Authorization "";'), 2)
        self.assertEqual(rendered.count("auth_gss_allow_basic_fallback off;"), 2)

    def test_admin_locations_override_inherited_satisfy_any(self):
        """Require both network authorization and SPNEGO despite server ACLs."""
        configured = insert_admin_locations(
            NGINX_SITE.replace(
                "server_name proxy.altanis.de;",
                "server_name proxy.altanis.de; satisfy any; "
                "allow 192.168.20.0/24; deny all;",
            )
        )
        self.assertEqual(
            render_nginx_locations(
                "/etc/nginx/proxy-admin.keytab",
                "HTTP/proxy.altanis.de@ALTANIS.DE",
            ).count("satisfy all;"),
            2,
        )
        self.assertIn("satisfy any;", configured)

    def test_inserted_admin_locations_do_not_change_public_proxy_location(self):
        """Add the protected include without altering public proxy behavior."""
        configured = insert_admin_locations(NGINX_SITE)
        self.assertEqual(configured.count("location / {"), 1)
        include = "include /etc/nginx/snippets/proxy-altanis-admin-iwa-locations.conf;"
        self.assertIn(include, configured)
        self.assertIn(
            "    include /etc/nginx/snippets/proxy-altanis-admin-iwa-locations.conf;",
            configured,
        )
        self.assertLess(configured.index(include), configured.index("location / {"))
        self.assertIn("proxy_set_header Authorization $http_authorization;", configured)
        self.assertEqual(configured.count("proxy-altanis-admin-iwa-locations.conf"), 1)
        self.assertEqual(insert_admin_locations(configured), configured)

    def test_alternate_admin_aliases_and_rewrites_are_rejected(self):
        """Reject route aliases that could bypass the protected admin locations."""
        aliased_admin = NGINX_SITE.replace(
            "    location / {",
            "    location /manage/ {\n"
            "        proxy_pass http://127.0.0.1:5000/admin/;\n"
            "    }\n\n    location / {",
        )
        rewritten_admin = NGINX_SITE.replace(
            "    location / {",
            "    rewrite ^/manage/(.*)$ /admin/$1 last;\n\n    location / {",
        )
        for config in (aliased_admin, rewritten_admin):
            with self.subTest(config=config), self.assertRaises(ValueError):
                insert_admin_locations(config)

    def test_tls_proxy_with_optional_redirect_server(self):
        """Protect TLS ingress and preserve the separate HTTP redirect block."""
        for original in (
            NGINX_TLS_SITE,
            NGINX_REDIRECT_SITE + NGINX_TLS_SITE,
            NGINX_TLS_SITE + NGINX_REDIRECT_SITE,
        ):
            with self.subTest(original=original), tempfile.TemporaryDirectory() as directory:
                configured = insert_admin_locations(original)
                include = "    include /etc/nginx/snippets/proxy-altanis-admin-iwa-locations.conf;\n"
                self.assertEqual(configured.count(include), 1)
                self.assertEqual(configured.replace(include, ""), original)
                self.assertEqual(insert_admin_locations(configured), configured)
                site_path = Path(directory) / "proxy.altanis.de"
                for content in (original, configured):
                    site_path.write_text(content, encoding="utf-8")
                    _validate_unique_active_proxy_server(
                        f"# configuration file {site_path}:\n{content}", site_path
                    )

    def test_only_exact_letsencrypt_options_include_is_allowed(self):
        """Keep external route includes and wildcard TLS includes rejected."""
        for path in (
            "/etc/letsencrypt/*.conf",
            "/etc/letsencrypt/options-ssl-nginx.conf.extra",
            "/etc/nginx/snippets/admin-routes.conf",
            "/etc/letsencrypt/options-ssl-nginx.conf /tmp/routes.conf",
        ):
            with self.subTest(path=path), self.assertRaises(ValueError):
                insert_admin_locations(
                    NGINX_TLS_SITE.replace("/etc/letsencrypt/options-ssl-nginx.conf", path)
                )

    def test_multiline_admin_location_is_rejected(self):
        """Fail closed when an admin location declaration is multiline."""
        multiline_admin = NGINX_SITE.replace(
            "    location / {",
            "    location\n        ^~\n        /admin/api/\n    {\n"
            "        proxy_pass http://127.0.0.1:5000;\n"
            "    }\n\n    location / {",
        )
        with self.assertRaises(ValueError):
            insert_admin_locations(multiline_admin)

    def test_multiline_rewrites_and_external_includes_are_rejected(self):
        """Reject multiline directives that could change route interpretation."""
        multiline_directives = (
            "    include\n        /etc/nginx/snippets/admin-routes.conf;",
            "    rewrite\n        ^/manage/(.*)$ /admin/$1 last;",
            "        proxy_pass\n            http://127.0.0.1:5000/admin/;",
        )
        for directive in multiline_directives:
            with self.subTest(directive=directive), self.assertRaises(ValueError):
                insert_admin_locations(
                    NGINX_SITE.replace(
                        "    location / {", f"{directive}\n\n    location / {{"
                    )
                )

    def test_inline_public_location_receives_idempotent_admin_include(self):
        """Insert the admin include in compact Nginx syntax idempotently."""
        inline_site = (
            "server { server_name proxy.altanis.de; location / { "
            "proxy_pass http://127.0.0.1:5000; } }"
        )
        configured = insert_admin_locations(inline_site)
        self.assertIn(
            "include /etc/nginx/snippets/proxy-altanis-admin-iwa-locations.conf;",
            configured,
        )
        self.assertEqual(insert_admin_locations(configured), configured)

    def test_effective_nginx_config_requires_unique_proxy_virtual_host(self):
        """Reject duplicate, foreign, and spoofed active proxy source blocks."""
        with tempfile.TemporaryDirectory() as directory:
            site_path = Path(directory) / "sites-available" / "proxy.altanis.de"
            other_path = Path(directory) / "conf.d" / "other.conf"
            site_path.parent.mkdir()
            other_path.parent.mkdir()
            site_path.write_text(NGINX_SITE, encoding="utf-8")
            other_content = "server { server_name other.altanis.de; }\n"
            other_path.write_text(other_content, encoding="utf-8")

            active_config = (
                f"# configuration file {other_path}:\n{other_content}"
                f"# configuration file {site_path}:\n{NGINX_SITE}"
            )
            _validate_unique_active_proxy_server(active_config, site_path)
            duplicate_content = NGINX_SITE.replace("proxy.altanis.de", "PROXY.ALTANIS.DE")
            duplicate_path = Path(directory) / "conf.d" / "duplicate.conf"
            duplicate_path.write_text(duplicate_content, encoding="utf-8")
            duplicate_active_config = (
                active_config
                + f"# configuration file {duplicate_path}:\n{duplicate_content}"
            )
            with self.assertRaises(DeploymentError):
                _validate_unique_active_proxy_server(duplicate_active_config, site_path)

            foreign_path = Path(directory) / "conf.d" / "active-proxy.conf"
            foreign_path.write_text(NGINX_SITE, encoding="utf-8")
            foreign_config = f"# configuration file {foreign_path}:\n{NGINX_SITE}"
            with self.assertRaises(DeploymentError):
                _validate_unique_active_proxy_server(foreign_config, site_path)

            spoof_path = Path(directory) / "conf.d" / "spoof.conf"
            spoof_content = (
                f"# configuration file {site_path}:\n"
                "server { server_name proxy.altanis.de; }\n"
            )
            spoof_path.write_text(spoof_content, encoding="utf-8")
            spoof_config = f"# configuration file {spoof_path}:\n{spoof_content}"
            with self.assertRaises(DeploymentError):
                _validate_unique_active_proxy_server(spoof_config, site_path)

    def test_escaped_duplicate_proxy_server_name_is_rejected(self):
        """Reject escaped server directives that hide duplicate proxy hosts."""
        escaped_duplicate = NGINX_SITE.rstrip() + (
            r" server { server\_name proxy.altanis.de; location / { } }"
        )
        with self.assertRaises(ValueError):
            insert_admin_locations(escaped_duplicate)

    def test_inline_duplicate_proxy_server_block_is_rejected(self):
        """Reject a second matching host block even in compact syntax."""
        duplicate_server = NGINX_SITE.rstrip() + (
            " server { server_name proxy.altanis.de; location / { "
            "proxy_pass http://127.0.0.1:5000; } }"
        )
        with self.assertRaises(ValueError):
            insert_admin_locations(duplicate_server)

    def test_nested_admin_location_is_rejected(self):
        """Reject nested locations that make admin route coverage ambiguous."""
        nested_admin = NGINX_SITE.replace(
            "    location / {",
            "    location / {\n"
            "        location ^~ /admin/api/ {\n"
            "            proxy_pass http://127.0.0.1:5000;\n"
            "        }",
        )
        with self.assertRaises(ValueError):
            insert_admin_locations(nested_admin)

    def test_escaped_directive_syntax_is_rejected(self):
        """Reject escaped directives that evade route safety checks."""
        escaped_directive = NGINX_SITE.replace(
            "server_name proxy.altanis.de;",
            r"server_name proxy.altanis.de; rew\rite ^/manage/(.*)$ /admin/$1 last;",
        )
        with self.assertRaises(ValueError):
            insert_admin_locations(escaped_directive)

    def test_inline_try_files_and_error_page_are_rejected(self):
        """Reject internal routing directives that could expose admin paths."""
        for directive in (
            "try_files $uri /admin/index.html;",
            "error_page 404 /admin/login;",
            "return 302 https://attacker.example/;",
            "if ($request_uri ~ ^/admin) { return 200; }",
        ):
            with self.subTest(directive=directive), self.assertRaises(ValueError):
                insert_admin_locations(
                    NGINX_SITE.replace(
                        "server_name proxy.altanis.de;",
                        f"server_name proxy.altanis.de; {directive}",
                    )
                )

    def test_quoted_location_directive_cannot_bypass_admin_route_check(self):
        """Treat quoted directive names as unsafe instead of missing routes."""
        quoted_location = NGINX_SITE.replace(
            "server_name proxy.altanis.de;",
            'server_name proxy.altanis.de; "location" ^~ /admin/api/ { '
            "proxy_pass http://127.0.0.1:5000; }",
        )
        with self.assertRaises(ValueError):
            insert_admin_locations(quoted_location)

    def test_quoted_proxy_pass_uri_rewrite_is_rejected(self):
        """Reject quoted upstream URIs that rewrite requests into admin paths."""
        quoted_proxy_pass = NGINX_SITE.replace(
            "proxy_pass http://127.0.0.1:5000;",
            'proxy_pass "http://127.0.0.1:5000/admin/";',
        )
        with self.assertRaises(ValueError):
            insert_admin_locations(quoted_proxy_pass)

    def test_inline_admin_routes_and_route_mutating_directives_are_rejected(self):
        """Reject direct admin routes and unsafe Nginx routing directives."""
        inline_configs = (
            NGINX_SITE.replace(
                "server_name proxy.altanis.de;",
                "server_name proxy.altanis.de; location ^~ /admin/api/ { "
                "proxy_pass http://127.0.0.1:5000; }",
            ),
            NGINX_SITE.replace(
                "server_name proxy.altanis.de;",
                "server_name proxy.altanis.de; include "
                "/etc/nginx/snippets/admin-routes.conf;",
            ),
            NGINX_SITE.replace(
                "server_name proxy.altanis.de;",
                "server_name proxy.altanis.de; rewrite ^/manage/(.*)$ /admin/$1 last;",
            ),
            NGINX_SITE.replace(
                "server_name proxy.altanis.de;",
                "server_name proxy.altanis.de; proxy_pass "
                "http://127.0.0.1:5000/admin/;",
            ),
        )
        for config in inline_configs:
            with self.subTest(config=config), self.assertRaises(ValueError):
                insert_admin_locations(config)

    def test_site_with_unexpected_shape_is_rejected(self):
        """Reject ambiguous server blocks and overlapping route declarations."""
        with self.assertRaises(ValueError):
            insert_admin_locations(
                "server { server_name other.altanis.de; location / { } }"
            )
        with self.assertRaises(ValueError):
            insert_admin_locations(
                NGINX_SITE.replace("location / {", "location / {\n    location / {")
            )
        for overlapping_location in (
            "location = /admin {",
            "location /admin {",
            "location ^~ /admin/ {",
        ):
            with (
                self.subTest(location=overlapping_location),
                self.assertRaises(ValueError),
            ):
                insert_admin_locations(
                    NGINX_SITE.replace(
                        "    location / {",
                        f"    {overlapping_location}\n    }}\n\n    location / {{",
                    )
                )
        catchall_prefix = insert_admin_locations(
            NGINX_SITE.replace("location / {", "location ^~ / {")
        )
        self.assertIn("location ^~ / {", catchall_prefix)
        self.assertIn(
            "location ^~ /admin/ {",
            render_nginx_locations(
                "/etc/nginx/proxy-admin.keytab",
                "HTTP/proxy.altanis.de@ALTANIS.DE",
            ),
        )
        misplaced_include = NGINX_SITE.replace(
            "    location / {",
            "    location / {\n        include /etc/nginx/snippets/"
            "proxy-altanis-admin-iwa-locations.conf;",
        )
        with self.assertRaises(ValueError):
            insert_admin_locations(misplaced_include)

    def test_router_kerberos_address_must_be_rfc1918_ipv4(self):
        """Restrict the router's Kerberos endpoint to a private IPv4 address."""
        self.assertEqual(validate_router_kerberos_address("10.0.5.1"), "10.0.5.1")
        for address in ("2001:db8::1", "8.8.8.8", "100.64.0.1", "not-an-ip"):
            with self.subTest(address=address), self.assertRaises(ValueError):
                validate_router_kerberos_address(address)

    def test_router_rules_only_allow_proxy_to_ad_dns_and_auth_ports(self):
        """Limit router rules to required proxy-to-domain-controller traffic."""
        rendered = render_router_rules(
            "10.0.5.1", forward_chain_handle=7, nat_chain_handle=12
        )
        self.assertEqual(rendered.count("insert rule "), 6)
        self.assertEqual(rendered.count("handle 7"), 4)
        self.assertEqual(rendered.count("handle 12"), 2)
        self.assertIn('iifname "ens256" oifname "wg2"', rendered)
        self.assertIn("ip saddr 192.168.20.11 ip daddr 192.168.253.5", rendered)
        self.assertIn(
            'iifname "wg2" oifname "ens256" ip saddr 192.168.253.5 '
            "ip daddr 192.168.20.11 tcp sport 88 ct state established,related",
            rendered,
        )
        self.assertIn(
            'iifname "wg2" oifname "ens256" ip saddr 192.168.253.5 '
            "ip daddr 192.168.20.11 udp sport 88 ct state established,related",
            rendered,
        )
        self.assertEqual(rendered.count("tcp dport 88"), 2)
        self.assertEqual(rendered.count("udp dport 88"), 2)
        self.assertIn("snat to 10.0.5.1", rendered)
        self.assertNotIn("192.168.20.0/24", rendered)
        with self.assertRaises(ValueError):
            render_router_rules("10.0.5.1", forward_chain_handle=0, nat_chain_handle=12)

    def test_include_is_added_to_the_matching_server_block(self):
        """Insert the admin include only into the configured proxy host block."""
        other_server = (
            "server {\n    server_name other.altanis.de;\n    location / { }\n}\n\n"
        )
        configured = insert_admin_locations(other_server + NGINX_SITE)
        proxy_server = configured.split("server_name proxy.altanis.de;", maxsplit=1)[1]
        other_server_body = configured.split(
            "server_name other.altanis.de;", maxsplit=1
        )[1].split("}", maxsplit=1)[0]
        include = "include /etc/nginx/snippets/proxy-altanis-admin-iwa-locations.conf;"
        self.assertLess(proxy_server.index(include), proxy_server.index("location / {"))
        self.assertNotIn(include, other_server_body)

    def test_existing_route_comparison_accepts_kernel_prefix_format(self):
        """Recognize an existing route in the kernel's displayed form."""
        self.assertTrue(
            _route_matches_expected(
                "192.168.253.5 via 192.168.20.31 dev ens224 proto static"
            )
        )
        self.assertFalse(
            _route_matches_expected("192.168.253.5 via 192.168.20.99 dev ens224")
        )

    def test_route_is_added_only_inside_existing_private_interface_stanza(self):
        """Add the host route only to the existing private static interface."""
        config = """auto ens224
    iface ens224 inet static
        address 192.168.20.11
        netmask 255.255.255.0

iface ens192 inet dhcp
"""
        configured = _network_config_with_route(config)
        self.assertIn(
            "        up ip route replace 192.168.253.5/32 via 192.168.20.31 dev ens224",
            configured,
        )
        self.assertLess(
            configured.index("dev ens224"), configured.index("iface ens192")
        )
        self.assertEqual(_network_config_with_route(configured), configured)

    def test_snapshot_rejects_dangling_symlink(self):
        """Reject dangling symlinks rather than treating them as absent files."""
        with tempfile.TemporaryDirectory() as directory:
            symlink = Path(directory) / "dangling.conf"
            symlink.symlink_to(Path(directory) / "missing.conf")
            with self.assertRaises(DeploymentError):
                _snapshot(symlink)

    def test_rollback_attempts_every_file_and_route_after_restore_failure(self):
        """Continue rollback across files and route when one restore fails."""
        first = Path("/etc/nginx/proxy-admin-iwa-first.conf")
        second = Path("/etc/nginx/proxy-admin-iwa-second.conf")
        snapshots = {
            first: FileSnapshot("first", 0o644, 0, 0),
            second: FileSnapshot("second", 0o644, 0, 0),
        }
        with (
            patch(
                "scripts.deploy_proxy_admin_iwa._restore",
                side_effect=[OSError("denied"), None],
            ) as restore,
            patch("scripts.deploy_proxy_admin_iwa._run") as run,
        ):
            errors = _rollback_changes([first, second], snapshots, route_added=True)

        self.assertEqual(
            restore.call_args_list,
            [call(second, snapshots[second]), call(first, snapshots[first])],
        )
        run.assert_not_called()
        self.assertEqual(len(errors), 2)
        self.assertIn(str(second), errors[0])
        self.assertIn("retained because rollback is incomplete", errors[1])

    def test_rollback_reloads_restored_nginx_before_removing_route(self):
        """Retain the KDC route when restoring or reloading Nginx fails."""
        nginx_config = NGINX_SITE_PATH
        snapshot = FileSnapshot("original", 0o644, 0, 0)
        with (
            patch("scripts.deploy_proxy_admin_iwa._restore") as restore,
            patch(
                "scripts.deploy_proxy_admin_iwa._run",
                side_effect=[None, DeploymentError("reload denied")],
            ) as run,
        ):
            errors = _rollback_changes(
                [nginx_config], {nginx_config: snapshot}, route_added=True
            )

        restore.assert_called_once_with(nginx_config, snapshot)
        self.assertEqual(
            run.call_args_list,
            [
                call(["nginx", "-t"]),
                call(["systemctl", "reload", "nginx"]),
            ],
        )
        self.assertEqual(len(errors), 2)
        self.assertIn("restored Nginx runtime", errors[0])
        self.assertIn("route", errors[1])
        self.assertIn("retained", errors[1])

    def test_conflicting_route_is_rejected(self):
        """Reject a host route that points to a different next hop."""
        config = """auto ens224
iface ens224 inet static
    address 192.168.20.11
    up ip route replace 192.168.253.5/32 via 192.168.20.99 dev ens224
"""
        with self.assertRaises(DeploymentError):
            _network_config_with_route(config)

    def test_route_is_separated_from_last_line_without_a_newline(self):
        """Preserve valid line separation when the source ends without newline."""
        config = "auto ens224\niface ens224 inet static\n    address 192.168.20.11"
        configured = _network_config_with_route(config)
        self.assertIn(
            "    address 192.168.20.11\n    up ip route replace ",
            configured,
        )

    def test_krb5_config_maps_the_realm_to_the_verified_domain_controller(self):
        """Map the Kerberos realm only to its verified domain controller."""
        rendered = render_krb5_config()
        self.assertIn("[realms]", rendered)
        self.assertIn("ALTANIS.DE = {", rendered)
        self.assertIn("kdc = 192.168.253.5", rendered)
        self.assertNotIn("default_realm", rendered)


if __name__ == "__main__":
    unittest.main()
