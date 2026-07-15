# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""API routes for plugin management.

- ``POST /api/v1/plugins/install`` installs a datus plugin from a typed source
  (``pip``/``src``/``whl``/``git``/``zip``). By default it lands in the managed
  ``~/.datus/plugins/{name}/`` store; an optional ``dest`` installs the tree
  into that exact directory instead (:func:`datus.cli.plugin_service.install`'s
  ``dest_dir`` parameter) — such a directory is not part of the managed
  store, so a successful ``dest`` install is persisted into
  ``agent.plugin_paths`` in the loaded agent.yml: the mount survives process
  restarts and other processes discover it on their next config load. A
  successful install also hot-reloads the running process: the service layer
  already refreshed ``sys.path`` + the import/plugin-registry caches, and the
  route evicts the project's cached ``DatusService`` so the next request
  rebuilds prompts/skills/transformers with the new plugin set.
- ``GET /api/v1/plugins`` enumerates installed plugins (managed +
  ``agent.plugin_paths`` mounts + externally pip-installed) with the current
  project's activation state and configured profile names.
- ``GET /api/v1/plugins/activation`` returns the project's activation view:
  the ``plugins_enabled`` master switch, whether the ``plugins:`` whitelist is
  present, and each plugin's active / pinned-profile state.
- ``PUT /api/v1/plugins/{name}/activation`` enables/disables a plugin and/or
  pins its active profiles for this project, persisted to
  ``./.datus/config.yml`` via ``AgentConfig.set_plugin_activation`` (the first
  write seeds an explicit whitelist with every installed plugin so nothing is
  silently deactivated).
- ``DELETE /api/v1/plugins/{name}`` removes a managed plugin.
- ``GET /api/v1/plugins/{name}/config-schema`` returns the normalized profile
  form fields derived from the plugin manifest's ``config_schema`` — the same
  specs the ``/plugins`` TUI renders.
"""

import asyncio
import os
from dataclasses import asdict
from pathlib import Path
from typing import Literal, Optional, get_args

from fastapi import APIRouter
from pydantic import BaseModel, Field

from datus.api import deps
from datus.api.deps import AppContextDep, ServiceDep
from datus.api.models.base_models import Result
from datus.cli import plugin_service
from datus.plugins import store
from datus.utils.loggings import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/api/v1", tags=["plugins"])

InstallType = Literal["pip", "src", "whl", "git", "zip"]
# The Literal is spelled out for OpenAPI; keep it in lockstep with the service.
assert set(get_args(InstallType)) == set(plugin_service.INSTALL_TYPES)

# stderr can be arbitrarily long (full pip output); return only the tail.
_STDERR_TAIL_CHARS = 2000


class InstallPluginRequest(BaseModel):
    """Install a plugin, by default into the managed store."""

    type: InstallType = Field(description="Install source type")
    source: str = Field(min_length=1, description="Requirement / local path / URL, interpreted per `type`")
    dest: Optional[str] = Field(
        default=None,
        description=(
            "Optional absolute destination directory (one directory = one plugin tree). "
            "Omit to install into the managed store ~/.datus/plugins/{name}/."
        ),
    )
    force: bool = Field(default=False, description="Replace the destination directory if it already exists")


class PluginActivationRequest(BaseModel):
    """Change a plugin's project activation and/or profile pins."""

    enabled: Optional[bool] = Field(default=None, description="Enable/disable the plugin; null leaves it unchanged")
    profiles: Optional[list[str]] = Field(
        default=None,
        description="Pin the active profiles to this list; null leaves the pins unchanged",
    )
    clear_profiles: bool = Field(
        default=False,
        description="Reset the profile pins to 'all profiles' (takes precedence over `profiles`)",
    )


def _ensure_plugin_discoverable(name: str, agent_config) -> bool:
    """Make plugin ``name``'s entry point visible in this process, then probe it.

    Mirrors the ``datus plugin`` CLI dispatch: append the managed directory —
    or the config's ``plugin_paths`` mounts — to ``sys.path`` (path-only, no
    plugin code import) and check the ``datus.plugins`` entry point exists.
    """
    from datus.plugins.registry import plugin_entry_point_exists

    if store.plugin_dir(name).is_dir():
        store.activate_name(name)
    else:
        store.activate_paths(getattr(agent_config, "plugin_paths", None))
    return plugin_entry_point_exists(name)


async def _evict_project_service(project_id: str) -> bool:
    """Drop the project's cached DatusService so the next request rebuilds it.

    The plugin store mutation already refreshed ``sys.path`` and the
    import/plugin-registry caches (``plugin_service._refresh``); evicting the
    service cache completes the reload at the consumer layer — prompts, skills
    and tool transformers are re-collected with the new plugin set on the next
    request. Returns whether an eviction was performed.
    """
    cache = deps._service_cache
    if cache is None:
        return False
    try:
        await cache.evict(project_id)
        return True
    except Exception:  # noqa: BLE001 - a reload failure must not fail the install itself
        logger.exception("Failed to evict service cache for project %s", project_id)
        return False


def _mount_plugin_path(dest: str) -> bool:
    """Persist ``dest`` into ``agent.plugin_paths`` so the mount is durable.

    Appends the directory to the loaded agent.yml's ``plugin_paths`` list and
    saves the file — without this, an arbitrary-``dest`` install is only known
    to this process's ``sys.path`` and vanishes on restart. Existing entries
    are compared on their expanded absolute path so a re-install never
    duplicates the mount. Returns whether ``dest`` is mounted after the call;
    never raises — a config write failure must not fail the install itself.
    """
    try:
        from datus.configuration.agent_config_loader import configuration_manager

        cm = configuration_manager()
        raw = cm.data.get("plugin_paths")
        entries = list(raw) if isinstance(raw, list) else []
        target = Path(dest).expanduser().resolve()
        for entry in entries:
            if not isinstance(entry, str):
                continue
            try:
                if Path(os.path.expandvars(entry)).expanduser().resolve() == target:
                    return True  # already mounted
            except OSError:
                continue
        entries.append(dest)
        cm.data["plugin_paths"] = entries
        cm.save()
        return True
    except Exception:  # noqa: BLE001 - a mount failure must not fail the install itself
        logger.exception("Failed to persist %s into agent.plugin_paths", dest)
        return False


@router.post(
    "/plugins/install",
    response_model=Result[dict],
    summary="Install Plugin",
    description=(
        "Install a datus plugin from a typed source, by default into the managed store "
        "(~/.datus/plugins/{name}/). Pass `dest` to install into an explicit directory instead; "
        "the directory is then persisted into `agent.plugin_paths` so the mount survives restarts. "
        "On success the project's cached service is evicted, so the plugin takes effect on the "
        "next request without restarting the API."
    ),
)
async def install_plugin_endpoint(
    body: InstallPluginRequest,
    svc: ServiceDep,  # noqa: ARG001 - authenticates the request and resolves the project
    ctx: AppContextDep,
) -> Result[dict]:
    """Run the (blocking) pip-based install off the event loop and hot-reload on success."""
    source = body.source.strip()
    if not source:
        return Result(success=False, errorCode="COMMON_FIELD_INVALID", errorMessage="'source' must not be empty.")
    dest = (body.dest or "").strip() or None
    if dest and not Path(dest).expanduser().is_absolute():
        return Result(
            success=False,
            errorCode="COMMON_FIELD_INVALID",
            errorMessage="'dest' must be an absolute directory path.",
        )

    result = await asyncio.to_thread(plugin_service.install, f"{body.type}:{source}", force=body.force, dest_dir=dest)
    payload = {
        "ok": result.ok,
        "name": result.name,
        "version": result.version,
        "plugin_dir": result.plugin_dir,
    }
    if result.ok:
        if dest:
            # Persist the mount BEFORE evicting, so the rebuilt service's
            # config load already sees the updated plugin_paths.
            payload["mounted"] = await asyncio.to_thread(_mount_plugin_path, dest)
        payload["reloaded"] = await _evict_project_service(ctx.project_id or "default")
    else:
        if result.stderr:
            payload["stderr"] = result.stderr[-_STDERR_TAIL_CHARS:]
        logger.info("Plugin install failed (type=%s dest=%s): %s", body.type, dest, result.error)
    return Result(success=result.ok, data=payload, errorMessage=result.error)


@router.get(
    "/plugins",
    response_model=Result[list],
    summary="List Plugins",
    description=(
        "Enumerate installed plugins — managed (~/.datus/plugins), `agent.plugin_paths` mounts and "
        "externally pip-installed — with the project's activation state and configured profile names."
    ),
)
async def list_plugins_endpoint(svc: ServiceDep) -> Result[list]:
    """Return every discoverable plugin as a flat record list."""
    infos = await asyncio.to_thread(plugin_service.list_plugins, svc.agent_config)
    return Result(success=True, data=[asdict(info) for info in infos])


@router.get(
    "/plugins/activation",
    response_model=Result[dict],
    summary="Get Project Plugin Activation",
    description=(
        "Return this project's plugin activation view: the `plugins_enabled` master switch, whether "
        "the `plugins:` whitelist is present in ./.datus/config.yml, and each discoverable plugin's "
        "active / pinned-profile state."
    ),
)
async def get_plugin_activation_endpoint(svc: ServiceDep) -> Result[dict]:
    """Return the activation whitelist state plus per-plugin activation records."""
    agent_config = svc.agent_config
    infos = await asyncio.to_thread(plugin_service.list_plugins, agent_config)
    return Result(
        success=True,
        data={
            "plugins_enabled": getattr(agent_config, "plugins_enabled", True),
            # Section presence (not derivable from active_plugin_names() once
            # plugins_enabled is off); absent flag on older configs = False.
            "whitelist_present": bool(getattr(agent_config, "_plugins_section_present", False)),
            "plugins": [
                {"name": i.name, "active": i.active, "active_profiles": i.active_profiles, "profiles": i.profiles}
                for i in infos
            ],
        },
    )


@router.put(
    "/plugins/{name}/activation",
    response_model=Result[dict],
    summary="Set Project Plugin Activation",
    description=(
        "Enable/disable a plugin and/or pin its active profiles for this project. Persisted to "
        "./.datus/config.yml; the first write seeds an explicit whitelist with every installed "
        "plugin so the others stay active. The project's cached service is evicted so the change "
        "applies on the next request."
    ),
)
async def set_plugin_activation_endpoint(
    name: str,
    body: PluginActivationRequest,
    svc: ServiceDep,
    ctx: AppContextDep,
) -> Result[dict]:
    """Update and persist one plugin's project activation, then hot-reload."""
    if not store.is_valid_name(name):
        return Result(success=False, errorCode="COMMON_FIELD_INVALID", errorMessage=f"invalid plugin name {name!r}")
    if body.enabled is None and body.profiles is None and not body.clear_profiles:
        return Result(
            success=False,
            errorCode="COMMON_FIELD_INVALID",
            errorMessage="At least one of 'enabled', 'profiles' or 'clear_profiles' must be provided.",
        )
    agent_config = svc.agent_config
    if not _ensure_plugin_discoverable(name, agent_config):
        return Result(
            success=False,
            errorCode="PLUGIN_STORE_ERROR",
            errorMessage=f"no installed plugin named {name!r}; run GET /api/v1/plugins to list plugins",
        )
    try:
        await asyncio.to_thread(
            agent_config.set_plugin_activation,
            name,
            enabled=body.enabled,
            active_profiles=body.profiles,
            clear_profiles=body.clear_profiles,
        )
    except Exception as exc:  # noqa: BLE001 - surface a clean error instead of a 500
        logger.exception("Failed to update activation for plugin %s", name)
        return Result(success=False, errorCode="PLUGIN_STORE_ERROR", errorMessage=str(exc))

    return Result(
        success=True,
        data={
            "name": name,
            "active": agent_config.plugin_active(name),
            "active_profiles": agent_config.active_plugin_profiles(name),
            "reloaded": await _evict_project_service(ctx.project_id or "default"),
        },
    )


@router.delete(
    "/plugins/{name}",
    response_model=Result[dict],
    summary="Uninstall Plugin",
    description=(
        "Remove a managed plugin's ~/.datus/plugins/{name}/ directory. Path-mounted "
        "(`agent.plugin_paths`) and externally pip-installed plugins cannot be removed here."
    ),
)
async def uninstall_plugin_endpoint(name: str, svc: ServiceDep) -> Result[dict]:  # noqa: ARG001
    """Uninstall a managed plugin by its entry-point name."""
    if not store.is_valid_name(name):
        return Result(success=False, errorCode="COMMON_FIELD_INVALID", errorMessage=f"invalid plugin name {name!r}")
    result = await asyncio.to_thread(plugin_service.uninstall, name)
    payload = {"ok": result.ok, "plugin": result.plugin, "package": result.package}
    if not result.ok:
        logger.info("Plugin uninstall failed (name=%s): %s", name, result.error)
    return Result(success=result.ok, data=payload, errorMessage=result.error)


@router.get(
    "/plugins/{name}/config-schema",
    response_model=Result[dict],
    summary="Get Plugin Profile Form Schema",
    description=(
        "Return the normalized profile form fields derived from the plugin manifest's `config_schema` "
        "(name/description/required/secret/default per field, insertion-ordered). An empty `fields` "
        "list means the plugin declares no schema — fall back to free-form key/value editing."
    ),
)
async def plugin_config_schema_endpoint(name: str, svc: ServiceDep) -> Result[dict]:
    """Return the profile form field specs for one plugin."""
    from datus.plugins.registry import load_plugin_manifest, plugin_config_schema

    if not store.is_valid_name(name):
        return Result(success=False, errorCode="COMMON_FIELD_INVALID", errorMessage=f"invalid plugin name {name!r}")
    _ensure_plugin_discoverable(name, svc.agent_config)
    manifest = load_plugin_manifest(name)
    if manifest is None:
        return Result(
            success=False,
            errorCode="PLUGIN_STORE_ERROR",
            errorMessage=f"no installed plugin named {name!r} (or its manifest is invalid; see server log)",
        )
    return Result(success=True, data={"name": name, "fields": plugin_config_schema(name)})
