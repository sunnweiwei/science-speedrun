"""Harbor agent that publishes an existing package: uploads $HARBOR_COPY_PACKAGE to /workspace/submission.

Used to (re)run a task's verifier through Harbor on a saved submission (verifier fixes, success-path checks)
without running a model. --agent harbor_ext.copy_agent:CopyPackageAgent
"""
import os
from typing import override

from harbor.agents.base import BaseAgent
from harbor.agents.capabilities import AgentCapabilities
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext


class CopyPackageAgent(BaseAgent):
    capabilities = AgentCapabilities()

    @staticmethod
    @override
    def name() -> str:
        return "copy-package"

    @override
    def version(self) -> str:
        return "1.0.0"

    @override
    async def setup(self, environment: BaseEnvironment) -> None:
        pass

    @override
    async def run(self, instruction: str, environment: BaseEnvironment, context: AgentContext) -> None:
        src = os.environ["HARBOR_COPY_PACKAGE"]
        await environment.exec(command="rm -rf /workspace/submission/* /workspace/submission/.[!.]* 2>/dev/null; mkdir -p /workspace/submission")
        await environment.upload_dir(source_dir=src, target_dir="/workspace/submission")
        user = getattr(environment, "default_user", None)
        if user:
            await environment.exec(command=f"chown -R {user} /workspace/submission")
