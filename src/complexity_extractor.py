# ============================
# Advanced Complexity Analyzer
# ============================

import ast
import json

class ComplexityAnalyzer(ast.NodeVisitor):

    def __init__(self, filename):
        self.filename = filename
        self.current_function = None
        self.time_blocks = {}
        self.space_blocks = {}
        self.loop_stack = []

        with open(filename, "r") as f:
            self.tree = ast.parse(f.read())

    # ----------------------------
    # Visit functions
    # ----------------------------
    def visit_FunctionDef(self, node):
        self.current_function = node.name
        self.time_blocks[node.name] = []
        self.space_blocks[node.name] = []
        self.loop_stack = []
        self.generic_visit(node)

    # ----------------------------
    # Visit for loops
    # ----------------------------
    def visit_For(self, node):
        bound = self._extract_bound(node)
        self.loop_stack.append(bound)
        self.generic_visit(node)
        self.loop_stack.pop()
        if not self.loop_stack:
            expr = self._product(self.loop_stack + [bound])
            self.time_blocks[self.current_function].append(expr)

    # ----------------------------
    # Detect list or array allocation
    # ----------------------------
    def visit_List(self, node):
        if self.current_function:
            expr = self._product(self.loop_stack) + " * ?"  # unknown element size
            self.space_blocks[self.current_function].append(expr)
        self.generic_visit(node)

    # ----------------------------
    # Detect NumPy array usage
    # ----------------------------
    def visit_Subscript(self, node):
        """
        Count indexing on arrays (e.g., batch_images[b][h][w][c])
        Each subscript dimension contributes multiplicatively.
        """
        dims = self._count_subscript_dims(node)
        if dims > 1:
            self.time_blocks[self.current_function].append(" * ".join(dims))
            self.space_blocks[self.current_function].append(" * ".join(dims))
        self.generic_visit(node)

    # ----------------------------
    # Helper: extract loop bounds
    # ----------------------------
    def _extract_bound(self, node):
        if isinstance(node.iter, ast.Call):
            if isinstance(node.iter.func, ast.Name):
                if node.iter.func.id == "range" and node.iter.args:
                    arg = node.iter.args[0]
                    if isinstance(arg, ast.Name):
                        return arg.id
                    elif isinstance(arg, ast.Constant):
                        return str(arg.value)
        return "1"

    # ----------------------------
    # Helper: multiply symbols
    # ----------------------------
    def _product(self, symbols):
        if not symbols:
            return "1"
        return " * ".join(symbols)

    # ----------------------------
    # Count subscripts for NumPy arrays
    # ----------------------------
    def _count_subscript_dims(self, node):
        """
        Recursively count dimensions in nested indexing: batch_images[b][i][j][k]
        Returns list of symbols
        """
        dims = []
        current = node
        while isinstance(current, ast.Subscript):
            index = current.slice
            if isinstance(index, ast.Index):
                if isinstance(index.value, ast.Name):
                    dims.append(index.value.id)
                elif isinstance(index.value, ast.Constant):
                    dims.append(str(index.value.value))
            elif isinstance(index, ast.Constant):
                dims.append(str(index.value))
            current = current.value
        return dims[::-1]  # reverse order for correct dimension

    # ----------------------------
    # Build final component complexity
    # ----------------------------
    def _build_component_result(self, func):
        time_expr = self._sum(self.time_blocks.get(func, []))
        space_expr = self._max_expression(self.space_blocks.get(func, []))
        return {
            "time": {
                "O": f"O({time_expr})",
                "Omega": f"Ω({time_expr})",
                "Theta": f"Θ({time_expr})"
            },
            "space": {
                "O": f"O({space_expr})",
                "Omega": f"Ω({space_expr})",
                "Theta": f"Θ({space_expr})"
            }
        }

    # ----------------------------
    # Helpers for summing / max
    # ----------------------------
    def _sum(self, expressions):
        if not expressions:
            return "1"
        return " + ".join(expressions)

    def _max_expression(self, expressions):
        if not expressions:
            return "1"
        return max(expressions, key=lambda x: len(x))

    # ----------------------------
    # Public API
    # ----------------------------
    def analyze(self):
        self.visit(self.tree)
        components = {}
        for func in self.time_blocks:
            components[func] = self._build_component_result(func)

        return {
            "metadata": {
                "generated_by": "ComplexityAnalyzer v3.2",
                "analysis_type": "static_symbolic_ast",
                "target_file": self.filename
            },
            "input_symbols": {
                "n": "Number of images",
                "b": "Batch size",
                "h": "Image height",
                "w": "Image width",
                "c": "Number of classes",
                "L": "Number of layers",
                "p": "Number of region proposals",
                "d": "Feature depth",
                "k": "Kernel size"
            },
            "assumptions": [
                "Worst-case bounds assumed",
                "All loops execute to symbolic upper bound",
                "Architecture depth L is symbolic",
                "Dense tensor representation",
                "Backprop mirrors forward pass",
                "Static analysis only (no profiling)"
            ],
            "components": components
        }

    def export_report(self, output_path):
        report = self.analyze()
        with open(output_path, "w") as f:
            json.dump(report, f, indent=4)