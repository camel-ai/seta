# ========= Copyright 2023-2026 @ CAMEL-AI.org. All Rights Reserved. =========
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ========= Copyright 2023-2026 @ CAMEL-AI.org. All Rights Reserved. =========

from pathlib import Path
import logging
from abc import ABC
from typing import List

from camel.runtimes.base import BaseRuntime
from camel.toolkits import FunctionTool
from camel.logger import get_logger

from harbor.environments.factory import EnvironmentFactory
from harbor.environments.base import BaseEnvironment
from harbor.environments.docker.remote_docker_environment import RemoteDockerEnvironment
from harbor.environments.docker.remote_docker_environment import RemoteDockerEnvironment
from harbor.models.task.task import Task
from harbor.models.trial.paths import EnvironmentPaths, TrialPaths
from harbor.models.environment_type import EnvironmentType

from seta_env.toolkits import TerminalToolkit
from seta_env.toolkits.terminal_toolkit_docker import TerminalToolkit as TerminalToolkitDocker

class DockerHarborRuntime(ABC):
    def __init__(self,
                task_dir: str = None,
                trial_root: str = None,
                session_id: str = None,
                environment_type: str = None,
                environment: BaseEnvironment = None,
                **kwargs
                ):
        r"""
        During initialization, 
        Depends on runtime backend, ENV VARIABLES are required for authentication

        - Daytona:
            - DAYTONA_API_KEY
            - DAYTONA_API_URL
        - Modal
            - MODAL_API_KEY
        Args:
            task_dir (str): The path to the task directory, which should be Harbor format compliant.
            trial_root (str): The root directory where trial outputs will be stored.
            session_id (str): A unique identifier for the trial session, aka trial_name, docker container name, etc.
            environment_type (str): The type of environment to create (e.g., "docker",
                "daytona", "modal"). This determines which Harbor environment backend to use.
            environment (BaseEnvironment, optional): An optional pre-initialized Harbor environment instance.
                If provided, this environment will be used directly instead of creating a new one based on environment_type. 
                This allows for more flexible usage where the environment can be set up externally and passed in.
        """
        super().__init__()

        if environment:
            self.harbor_env = environment
            self.trial_dir = environment.trial_paths.trial_dir
            self.session_id = environment.session_id
            self._trial_paths = environment.trial_paths
            self._file_handler = None
            return

        assert environment_type in [
                            EnvironmentType.DOCKER.value,
                            EnvironmentType.DAYTONA.value,
                            EnvironmentType.MODAL.value,
                            "remote_docker",
                                ], f"Unsupported environment type: {environment_type}"

        # ------------ Set up paths and load task configuration ------------
        self.trial_root = Path(trial_root)
        self.trial_dir = self.trial_root / session_id
        self.session_id = session_id
        self._task = Task(Path(task_dir))   # this loads standard Harbor task structure from the given directory
        self._trial_paths = TrialPaths(trial_dir=self.trial_dir)

        # ------------ Set up logging for the trial ------------
        self._trial_paths.mkdir()
        self._logger = get_logger(f"{__name__}.{self.session_id}")
        self._file_handler: logging.FileHandler | None = None
        self._init_logger()

        if environment_type == "remote_docker":
            self.harbor_env = RemoteDockerEnvironment(
                node_manager_url=kwargs.pop("node_manager_url"),
                api_key=kwargs.pop("node_api_key"),
                environment_name=self._task.name,
                session_id=session_id,
                trial_paths=self._trial_paths,
                task_env_config=self._task.config.environment,
            )
        else:
            # Forward override_* (override_cpus, override_memory_mb,
            # override_storage_mb) and any other extra kwargs to the harbor env
            # constructor. env_service threads these via runtime_config (see
            # env_service.py:474-503) so each sandbox respects the YAML's
            # resource overrides. Without this, BaseEnvironment.__init__ never
            # sees override_storage_mb and sandboxes silently fall back to the
            # task.toml default (10 GiB), which can exhaust Daytona's per-org
            # disk quota under high-concurrency rollouts.
            self.harbor_env = EnvironmentFactory.create_environment(
                                                type=environment_type,
                                                environment_dir=self._task.paths.environment_dir,
                                                environment_name=self._task.name,
                                                session_id=session_id,
                                                trial_paths=self._trial_paths,
                                                task_env_config=self._task.config.environment,
                                                logger=self._logger,
                                                **kwargs,
                                            )

    # ----------------- Runtime interface methods -----------------
    async def build(self) -> None:
        """Pre-build the image/snapshot without starting the container.

        Delegates to the backend's build():
          - RemoteDockerEnvironment: runs docker build on the remote node.
          - DockerEnvironment: runs docker compose build locally.
          - DaytonaEnvironment: creates/verifies the Daytona snapshot.
        """
        await self.harbor_env.build()

    async def reset(self, force_build: bool = False, reset_timeout: float | None = None) -> None | List[FunctionTool]:
        r"""Asynchronously resets the runtime to its initial state.

        Args:
            force_build (bool): Force rebuild of the environment image.
            reset_timeout (float | None): Timeout in seconds for the reset operation.
                If None, waits indefinitely. Raises asyncio.TimeoutError if exceeded.
        """
        import asyncio
        if reset_timeout is not None:
            await asyncio.wait_for(
                self.harbor_env.start(force_build=force_build),
                timeout=reset_timeout,
            )
        else:
            await self.harbor_env.start(force_build=force_build)


    async def stop(self, delete: bool = False):
        r"""Asynchronously stops the runtime and releases resources."""
        await self.harbor_env.stop(delete=delete)
        self._close_logger()

    async def get_tools(self, toolkit: str = "auto") -> List[FunctionTool]:
        r"""Returns a list of all tools in the runtime.

        Args:
            toolkit (str): Which terminal toolkit implementation to use.
                "auto"   - default: docker env → TerminalToolkitDocker,
                           all other envs → TerminalToolkit (tmux-based).
                "tmux"   - always use TerminalToolkit (tmux-based, runtime-agnostic).
                "docker" - always use TerminalToolkitDocker (direct Docker API).
        """
        use_docker_toolkit = (
            toolkit == "docker"
            or (
                toolkit == "auto"
                and self.harbor_env.type() == EnvironmentType.DOCKER
                and not isinstance(self.harbor_env, RemoteDockerEnvironment)
            )
        )

        if use_docker_toolkit:
            self.terminal_toolkit = TerminalToolkitDocker(
                                        working_directory=None,
                                        session_logs_dir=str(self.trial_dir / "terminal_logs"),
                                        use_docker_backend=True,
                                        docker_container_name=self.session_id.lower().replace(".", "-") + "-main-1",
                                        safe_mode=False,
                                                )
        else:
            # timeout=None disables CAMEL's BaseToolkit.__init_subclass__ auto-wrap
            # that would otherwise apply asyncio.wait_for(self.timeout) to EVERY
            # async method on TerminalToolkit — including _ensure_tmux and
            # _ensure_working_directory used in toolkit setup. Default self.timeout
            # is 20s; under high concurrent Daytona load the post-create exec
            # ("which tmux" / "mkdir -p /workdir") takes >20s on the slow path,
            # which fired empty-message TimeoutError(), surfacing as `1_reset_env`
            # failures clustered at ~50-70s elapsed (the 20s budget plus the
            # sandbox-create time). The per-stage timeout in terminal_env
            # (`_reset_env`=3600s) is the correct budget for these setup calls;
            # we don't want a 20s nested deadline on top of it.
            self.terminal_toolkit = await TerminalToolkit(
                                        timeout=None,
                                        working_directory="/workdir",
                                        session_logs_dir=str(self.trial_dir / "terminal_logs"),
                                        runtime=self
                                                )
        self.tools = [
            FunctionTool(self.terminal_toolkit.shell_exec),
            FunctionTool(self.terminal_toolkit.shell_view),
            FunctionTool(self.terminal_toolkit.shell_wait),
            FunctionTool(self.terminal_toolkit.shell_write_to_process),
            FunctionTool(self.terminal_toolkit.shell_kill_process),
            FunctionTool(self.terminal_toolkit.shell_write_content_to_file),
            # FunctionTool(self.terminal_toolkit.shell_image_read),
        ]

    def __getattr__(self, name: str):
        """Proxy any unresolved attribute lookups to the underlying harbor_env."""
        return getattr(self.harbor_env, name)
    
    # Context manager support for async with statements
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.stop()

    # ----------------- Helper methods -----------------
    def _init_logger(self) -> None:
        """Attaches a session-scoped FileHandler to the logger.

        Safe to call multiple times — a second call is a no-op if the handler
        is already attached.
        """
        if self._file_handler is not None:
            return
        self._file_handler = logging.FileHandler(self._trial_paths.log_path)
        self._file_handler.setLevel(logging.DEBUG)
        self._logger.addHandler(self._file_handler)

    def _close_logger(self) -> None:
        """Flushes, closes, and detaches the session-scoped FileHandler.

        Safe to call multiple times — a second call is a no-op if the handler
        is already closed.  Only removes the handler this runtime owns, leaving
        any other handlers (e.g. console handlers added by get_logger) intact.
        """
        if self._file_handler is None:
            return
        self._file_handler.flush()
        self._file_handler.close()
        self._logger.removeHandler(self._file_handler)
        self._file_handler = None
