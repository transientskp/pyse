from docutils import nodes
from docutils.parsers.rst import Directive
from sourcefinder.config import IMGCONF_HELP, EXPORTSETTINGS_HELP


class RenderConfs(Directive):
    has_content = False

    @staticmethod
    def _render_entry(name: str, doc: str) -> nodes.paragraph:
        para = nodes.paragraph()
        para += nodes.strong(text=f"{name}: ")
        para += nodes.Text(doc)
        return para

    def run(self):
        sections = []

        config_sets = [
            (
                (
                    "ImgConf attributes: configuration options for image "
                    "processing "
                    "and source extraction."
                ),
                "imgconf-parameters",
                IMGCONF_HELP,
            ),
            (
                (
                    "ExportSettings attributes: configuration options for "
                    "export of source finding results."
                ),
                "exportsettings-parameters",
                EXPORTSETTINGS_HELP,
            ),
        ]

        for title_text, section_id, help_dict in config_sets:
            section = nodes.section()
            section["ids"].append(section_id)

            title = nodes.title(text=title_text)
            section += title

            for name, doc in help_dict.items():
                section += self._render_entry(name, doc)

            sections.append(section)

        return sections


def setup(app):
    app.add_directive("render_confs", RenderConfs)
