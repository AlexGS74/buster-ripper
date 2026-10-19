"""T006 unit coverage: the upstream TLS verification opt-out.

Pins the default posture (verification on), the --skip-tls-verify →
config.SKIP_TLS_VERIFY link at parse time, and the one shared connector
built inside on_startup. The process-level gate against a real
self-signed peer (committed fixture in tests/qa/) is QAP-MS004,
authored by forward-qa at MS-004 — not this module.
"""

import aiohttp
import pytest

from buster_ripper import config
from buster_ripper.main import main
from buster_ripper.server import on_startup


@pytest.fixture()
def restore_skip_tls_verify():
    """Restore the TLS opt-out global around each CLI-level test."""
    previous = config.SKIP_TLS_VERIFY
    yield
    config.SKIP_TLS_VERIFY = previous


class TestSkipTlsVerifyOption:
    """--skip-tls-verify assigns config.SKIP_TLS_VERIFY at parse time."""

    def test_flag_assigns_global_at_parse_time(self, restore_skip_tls_verify):
        """Passing the flag sets the opt-out before main() runs."""
        config.SKIP_TLS_VERIFY = False
        ctx = main.make_context("buster-ripper", ["--skip-tls-verify"])
        try:
            assert config.SKIP_TLS_VERIFY is True
            assert "skip_tls_verify" not in ctx.params
        finally:
            ctx.close()

    def test_default_keeps_verification_on(self, restore_skip_tls_verify):
        """No flag restores the off default over any prior mutation."""
        config.SKIP_TLS_VERIFY = True
        ctx = main.make_context("buster-ripper", [])
        try:
            assert config.SKIP_TLS_VERIFY is False
            assert "skip_tls_verify" not in ctx.params
        finally:
            ctx.close()

    def test_help_lists_flag(self):
        """--help shows --skip-tls-verify."""
        from click.testing import CliRunner

        result = CliRunner().invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "--skip-tls-verify" in result.output


class TestStartupConnectorPosture:
    """on_startup builds the package's only ClientSession on an explicit
    TCPConnector whose ssl value is the process-wide TLS posture."""

    @pytest.fixture()
    async def run_startup(self, mocker):
        """Run on_startup with the side-effectful startup steps stubbed.

        Yields a runner; every session it builds is closed afterwards so
        no connector leaks across tests.
        """
        mocker.patch("buster_ripper.server.backend.detect_backend")
        mocker.patch("buster_ripper.server.spawn_kv_cache_monitor")
        mocker.patch("buster_ripper.server.spawn_dump_flush")
        built = []

        async def _run():
            app = {}
            await on_startup(app)
            built.append(app["session"])
            return app

        yield _run
        for session in built:
            await session.close()

    async def test_default_posture_verifies(self, run_startup, restore_skip_tls_verify):
        """With the opt-out unset the connector carries ssl=True (verify)."""
        config.SKIP_TLS_VERIFY = False
        app = await run_startup()

        connector = app["session"].connector
        assert isinstance(connector, aiohttp.TCPConnector)
        assert connector is app["session"].connector
        assert connector._ssl is True

    async def test_optout_posture_disables_verification(
        self, run_startup, restore_skip_tls_verify
    ):
        """With the opt-out set the same construction carries ssl=False."""
        config.SKIP_TLS_VERIFY = True
        app = await run_startup()

        assert app["session"].connector._ssl is False

    async def test_session_binds_connectors_loop(self, run_startup, restore_skip_tls_verify):
        """The session adopts the connector's loop; construction raises nothing.

        Regression companion to the corrected connector-loop attribution: the
        running-loop requirement is TCPConnector's own; no RuntimeError is
        raised (or expected) on this default adoption path.
        """
        import asyncio

        config.SKIP_TLS_VERIFY = False
        app = await run_startup()

        assert app["session"].connector._loop is asyncio.get_running_loop()
        assert not app["session"].closed
