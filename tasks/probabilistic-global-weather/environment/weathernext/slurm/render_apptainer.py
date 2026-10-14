#!/usr/bin/env python3
"""Render this task's small Dockerfile subset into a native Apptainer recipe.

Fails on unsupported instructions. The RUN commands, base digest, environment
and COPY inputs come from the actual Dockerfile, avoiding a second runtime spec.
This is an Apptainer build, not evidence that Docker/Compose was exercised.
"""
import argparse
from pathlib import Path
import shlex


def render(dockerfile, context=None):
    context = (context or dockerfile.parent).resolve()
    instructions = dockerfile.read_text().replace("\\\n", " ").splitlines()
    post, env, files = ["set -eu"], [], []
    base = None
    staged = False
    for line in instructions:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        kind, value = line.split(None, 1)
        if kind == "FROM":
            if value in ('runtime AS starter-build', 'runtime AS environment'):
                if not staged:raise ValueError('Unknown runtime stage')
                continue
            if value.endswith(' AS runtime'):
                value=value.removesuffix(' AS runtime');staged=True
            if base is not None or "@sha256:" not in value:
                raise ValueError("one digest-pinned base is required")
            repository, digest = value.split("@", 1)
            # Apptainer rejects tag@digest syntax; retain the exact digest.
            base = repository.rsplit(":", 1)[0] + "@" + digest
        elif kind == "ARG":
            key, value = value.split("=", 1)
            post.append(f"export {key}={shlex.quote(value)}")
        elif kind == "ENV":
            for assignment in shlex.split(value):
                key, value = assignment.split("=", 1)
                # PATH intentionally expands the container's existing path.
                escaped = value.replace('"', '\\"').replace('`', '\\`')
                export = f'export {key}="{escaped}"'
                post.append(export)
                env.append(export)
        elif kind == "RUN":
            post.append(value)
        elif kind == "WORKDIR":
            post.append(f"mkdir -p {shlex.quote(value)}")
            post.append(f"cd {shlex.quote(value)}")
        elif kind == "COPY":
            parts=shlex.split(value)
            if parts[0]=='--from=starter-build':
                if not staged or parts[1:] not in [['/agent/starter','/workspace/starter'],['/agent/evaluation','/workspace/evaluation']]:
                    raise ValueError('Unsupported stage copy')
                source,target=parts[1:]
                post.append(f'mkdir -p {shlex.quote(target)} && cp -a {shlex.quote(source)}/. {shlex.quote(target)}/')
                continue
            *sources,target=parts
            for source in sources:
                source = (context / source).resolve()
                if not source.is_relative_to(context):
                    raise ValueError("COPY source escapes build context")
                copy_source = str(source) + ("/." if source.is_dir() else "")
                files.append(f"{copy_source} {target}")
        elif kind == "CMD" and value == '["sleep", "infinity"]':
            pass  # Tests and development commands are invoked explicitly.
        else:
            raise ValueError(f"unsupported Docker instruction: {line}")
    if staged:post.append('rm -rf -- /build /agent')
    return (f"Bootstrap: docker\nFrom: {base}\n\n%files\n" +
            "\n".join("    " + line for line in files) + "\n\n%post\n" +
            "\n".join("    " + line for line in post) + "\n\n%environment\n" +
            "\n".join("    " + line for line in env) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dockerfile", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--context", type=Path)
    args = parser.parse_args()
    args.output.write_text(render(args.dockerfile.resolve(), args.context))
