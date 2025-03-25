import re
from sphinx.application import Sphinx

def replace_escape_characters(text):
    text = re.sub(r"\t", r"\\t", text)
    text = re.sub(r"  ext{", r"\\text{", text)
    text = re.sub(r"\f", r"\\f", text)
    text = re.sub(r"\r", r"\\r", text)
    return text

def replace_math_syntax(lines: list[str]):
    """Replace $$...$$ with .. math:: (block) and $...$ with :math: (inline) in autodoc."""
    processed_lines = []
    inside_block_math = False
    block_math_content = []    
    for line in lines:
        # Handle block math 
        if line.strip() == "$$":
            if inside_block_math:
                # End block math
                processed_lines.append(".. math::")
                processed_lines.append("")  # Empty line for spacing
                processed_lines.extend("    " + math_line for math_line in block_math_content)
                processed_lines.append("")  # Empty line for spacing
                block_math_content = []
            inside_block_math = not inside_block_math
            continue

        if inside_block_math:
            # Process block math content
            line = replace_escape_characters(line)
            block_math_content.append(line)
        else:
            # Process inline math 
            line = re.sub(r"\$\$(.+?)\$\$", lambda m: f":math:`{m.group(1).strip()}`", line)
            processed_lines.append(replace_escape_characters(line))
    print(processed_lines)
    # Return the processed lines
    return processed_lines


def autodoc_process_docstring(app, what, name, obj, options, lines):
    """Modify autodoc docstrings before they are processed."""
    new_lines = replace_math_syntax(replace_escape_characters(obj.__doc__).split("\n"))
    lines[:] = new_lines  # Modify lines in-place


def setup(app: Sphinx):
    app.connect("autodoc-process-docstring", autodoc_process_docstring)
