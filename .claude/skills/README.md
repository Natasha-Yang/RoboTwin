# Project skills

Claude Code auto-discovers skills placed here (`.claude/skills/`). Each skill is a
subdirectory containing a `SKILL.md` with YAML frontmatter:

```
.claude/skills/
  my-skill/
    SKILL.md        # required: name + description in frontmatter, instructions in body
    (optional supporting files, scripts, templates)
```

Minimal `SKILL.md`:

```markdown
---
name: my-skill
description: One line telling Claude when to use this skill.
---

Instructions for the skill go here.
```

Invoke with `/my-skill`, or Claude will surface it automatically when the
description matches the task. These are checked into git so every clone of this
repo on every machine gets the same skills.
