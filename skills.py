import os
import re


SKILLS_DIR = os.path.join(os.path.dirname(__file__), "skills")


def _read_text(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def _parse_frontmatter_meta(raw_meta: str) -> dict:
    # Skill 的发现阶段只依赖 frontmatter：name、description、keywords 等轻量信息。
    meta = {}
    current_list_key = None

    for raw_line in raw_meta.splitlines():
        line = raw_line.rstrip()
        if not line.strip():
            continue
        if ":" in line and not line.lstrip().startswith("- "):
            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip()
            if value:
                meta[key] = value.strip('"').strip("'")
                current_list_key = None
            else:
                meta[key] = []
                current_list_key = key
            continue
        if current_list_key and line.strip().startswith("- "):
            meta[current_list_key].append(line.strip()[2:].strip())

    return meta


def _parse_frontmatter(content: str) -> tuple[dict, str]:
    match = re.match(r"^---\s*\n(.*?)\n---\s*\n?(.*)$", content, re.DOTALL)
    if not match:
        return {}, content

    raw_meta, body = match.groups()
    meta = _parse_frontmatter_meta(raw_meta)
    return meta, body


def _read_frontmatter(file_path: str) -> dict:
    """只读取 SKILL.md 顶部 frontmatter，避免启动发现阶段加载技能正文。"""
    with open(file_path, "r", encoding="utf-8") as f:
        first_line = f.readline()
        if first_line.strip() != "---":
            return {}

        raw_meta_lines = []
        for line in f:
            if line.strip() == "---":
                return _parse_frontmatter_meta("".join(raw_meta_lines))
            raw_meta_lines.append(line)

    return {}


def _parse_skill_manifest(skill_dir: str) -> dict | None:
    """解析技能目录中的最小元信息，用于轻量发现（discovery）"""
    skill_md = os.path.join(skill_dir, "SKILL.md")
    if not os.path.isfile(skill_md):
        return None

    meta = _read_frontmatter(skill_md)
    name = meta.get("name") or os.path.basename(skill_dir)
    description = meta.get("description", "")
    keywords = meta.get("keywords", [])
    if isinstance(keywords, str):
        keywords = [keyword.strip() for keyword in keywords.split(",") if keyword.strip()]

    return {
        "name": name,
        "description": description,
        "keywords": keywords,
        "skill_dir": skill_dir,
        "skill_md": skill_md,
    }


def _collect_skill_resources(skill_dir: str) -> list[str]:
    # 附加资源只列路径，不在加载 Skill 时一次性展开，避免上下文被模板/资料撑大。
    resources = []
    for root, _, files in os.walk(skill_dir):
        for file_name in files:
            if file_name == "SKILL.md":
                continue
            resources.append(os.path.join(root, file_name))
    resources.sort()
    return resources


class SkillRegistry:
    # Skills 不是工具函数，而是可复用的任务说明文件。
    # Registry 只在启动时轻量发现 manifest，真正需要时再加载完整 SKILL.md。
    """统一管理技能目录、轻量 manifest 和按需加载的技能正文"""

    def __init__(self, skills_dir: str = SKILLS_DIR):
        self.skills_dir = skills_dir
        self.skills: dict[str, dict] = {}
        self._discover_skills()

    def _discover_skills(self):
        # 启动阶段只读取每个 SKILL.md 的 frontmatter manifest，不把完整正文塞进 prompt。
        self.skills = {}
        if not os.path.isdir(self.skills_dir):
            return

        for entry in sorted(os.listdir(self.skills_dir)):
            skill_dir = os.path.join(self.skills_dir, entry)
            if not os.path.isdir(skill_dir):
                continue
            manifest = _parse_skill_manifest(skill_dir)
            if not manifest:
                continue
            self.skills[manifest["name"]] = {"manifest": manifest}

    def list_manifests(self) -> list[dict]:
        return [self.skills[name]["manifest"] for name in sorted(self.skills.keys())]

    def describe_available(self) -> str:
        # 系统 prompt 常驻的是技能目录，不是完整技能正文。
        manifests = self.list_manifests()
        if not manifests:
            return "- 暂无可用技能"
        return "\n".join(
            f"- {skill['name']}: {skill.get('description', '')}"
            for skill in manifests
        )

    def get_manifest(self, name: str) -> dict | None:
        entry = self.skills.get(name)
        return entry["manifest"] if entry else None

    def match(self, user_input: str) -> dict | None:
        # 关键词命中只提示“可能需要这个 skill”，真正正文仍由 load_skill 按需读取。
        text = user_input.lower()
        for skill in self.list_manifests():
            if skill["name"].lower() in text:
                return skill
            for keyword in skill.get("keywords", []):
                if keyword.lower() in text:
                    return skill
        return None

    def load_skill(self, name: str) -> str:
        # 只有模型或 slash 命令明确需要时，才加载完整技能正文和附加资源列表。
        # Skill 提供做事流程；真正读文件、搜索、执行命令仍然要走 tools。
        manifest = self.get_manifest(name)
        if not manifest:
            available = ", ".join(skill["name"] for skill in self.list_manifests())
            raise ValueError(f"技能 '{name}' 不存在，可用技能：{available}")

        _, body = _parse_frontmatter(_read_text(manifest["skill_md"]))
        body = body.strip()
        resources = _collect_skill_resources(manifest["skill_dir"])
        if resources:
            resource_block = "\n".join(f"- {path}" for path in resources)
        else:
            resource_block = "- 无附加资源"

        return (
            # 返回给模型的是结构化技能说明，附加资源仍要求按需再读取。
            f"<skill name=\"{manifest['name']}\">\n"
            f"{body}\n"
            f"</skill>\n\n"
            f"附加资源（按需再读取，不要一次性全部展开）：\n"
            f"{resource_block}"
        )


_REGISTRY: SkillRegistry | None = None


def get_skill_registry() -> SkillRegistry:
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = SkillRegistry()
    return _REGISTRY


def load_skills() -> list[dict]:
    """兼容旧接口：返回轻量 manifest 列表，而非完整技能正文"""
    return get_skill_registry().list_manifests()


def load_skill(name: str) -> str:
    """按需加载某个技能的完整正文，并仅披露附加资源目录"""
    return get_skill_registry().load_skill(name)


def match_skill(user_input: str, skills: list[dict] | None = None) -> dict | None:
    """兼容旧接口：根据技能名或关键词匹配轻量 manifest"""
    if skills is None:
        return get_skill_registry().match(user_input)

    text = user_input.lower()
    for skill in skills:
        if skill.get("name", "").lower() in text:
            return skill
        for keyword in skill.get("keywords", []):
            if keyword.lower() in text:
                return skill
    return None
