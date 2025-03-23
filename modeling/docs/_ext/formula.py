import re
import inspect
import importlib
from docutils import nodes
from sphinx.application import Sphinx
from sphinx.util.docutils import SphinxDirective
from sphinx.util.typing import ExtensionMetadata


class FormulaDirective(SphinxDirective):
    """Directive to include function docstrings and convert LaTeX math to Sphinx format."""

    required_arguments = 1

    def run(self) -> list[nodes.Node]:
        func_path = self.arguments[0]

        try:
            module_name, func_name = func_path.rsplit(".", 1)
            module = importlib.import_module(module_name)
            func = getattr(module, func_name)

            if not func.__doc__:
                return [self.state_machine.reporter.warning(f"No docstring found in {func_path}", line=self.lineno)]

            docstring = inspect.getdoc(func)
            formatted_nodes = self._convert_math_to_nodes(docstring)

            # Attach formatted math nodes to the docstring
            return formatted_nodes
        except (ImportError, AttributeError) as e:
            error_msg = f"Error importing {func_path}: {e}"
            return [self.state_machine.reporter.warning(error_msg, line=self.lineno)]

    def _convert_math_to_nodes(self, text) -> list[nodes.Node]:
        """Convert LaTeX math notation ($$...$$) into Sphinx-compatible nodes."""

        if not isinstance(text, str):
            return []

        lines = text.split("\n")
        processed_nodes = []
        inside_block_math = False
        block_math_content = []
        
        for i, line in enumerate(lines, 1):
            if line.strip() == "$$":
                if inside_block_math:
                    # Closing block math: Create a math_block node
                    math_block = nodes.math_block("", "\n".join(block_math_content), number="")
                    processed_nodes.append(math_block)
                    block_math_content = []
                inside_block_math = not inside_block_math
                continue

            if inside_block_math:
                line = re.sub("\\t", "\\\\t", line)
                line = re.sub("(    ext)", "\\\\text", line)
                line = re.sub("\\f", "\\\\f", line)
                line = re.sub("\\r", "\\\\r", line)
                block_math_content.append(line)
            else:
                parts = re.split(r"(\$\$.*?\$\$)", line)  # Split at inline math occurrences
                paragraph = nodes.paragraph()
                
                for part in parts:
                    if part.startswith("$$") and part.endswith("$$"):
                        math_code = re.match(r"\$\$(.*?)\$\$", part).group(1).strip()
                        paragraph += nodes.math("", math_code)    
                    else:
                        paragraph += nodes.Text(f"{part}")

                processed_nodes.append(paragraph)

        return processed_nodes


def setup(app: Sphinx) -> ExtensionMetadata:
    app.add_directive("formula", FormulaDirective)

    return {
        "version": "0.1",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
