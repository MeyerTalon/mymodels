---
name: _template
description: "TODO: what this skill does and when to use it. Replace this stub before relying on it."
disable-model-invocation: true
---

# Skill title

## Creating a new skill from this template

1. Copy this directory to `.agents/skills/<skill-name>/`. `.claude/skills` and `.cursor/skills` are symlinks to that tree; never create separate copies.
2. Set `name` to match the directory name exactly.
3. Rewrite `description` carefully: it is the only text the model sees when deciding whether to trigger the skill. State what the skill does, then list triggers: `/<skill-name>`, common phrasings, and whether implicit signals count.
4. Remove `disable-model-invocation: true` (it keeps this stub from auto-triggering) unless the skill should be invocable only by explicit `/<skill-name>`.
5. Register the skill under Skills in `AGENTS.md`.
6. Delete this section.

## Instructions

1. Replace with concrete rules or steps: imperative voice, one behavior per bullet, boldface the rule then explain it (see `ponytail` and `python-coding` for the pattern).
2. Keep SKILL.md under 500 lines; put long reference material in sibling files and link to them.
3. Compose with `python-coding` and `ponytail` for code; with `concise` when the user wants a short reply.

## Anti-goals

State what the skill must NOT do: the failure mode a too-literal reading of the rules would produce.

## Examples

- Example trigger → expected outcome
