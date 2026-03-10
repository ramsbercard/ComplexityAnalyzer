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
            base = self._product(self.loop_stack)
            if base == "1":
                expr = "d"
            else:
                expr = f"{base} * d"
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

        # If dims is a list, use its length for comparison
        if isinstance(dims, list):
            dim_count = len(dims)
        else:
            dim_count = dims  # fallback if integer

        if dim_count > 1:
            # Convert all dims to string before join (safe)
            dims_str = [str(d) for d in dims] if isinstance(dims, list) else [str(dims)]

            self.time_blocks[self.current_function].append(" * ".join(dims_str))
            self.space_blocks[self.current_function].append(" * ".join(dims_str))

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
        Recursively count symbolic dimensions in nested indexing.
        Ignore constant indices (e.g., [0]).
        """
        dims = []
        current = node

        while isinstance(current, ast.Subscript):
            index = current.slice

            # Python 3.9+
            if isinstance(index, ast.Name):
                dims.append(index.id)

            # If constant → ignore (do NOT append)
            elif isinstance(index, ast.Constant):
                pass

            current = current.value

        return dims[::-1]

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

        # Remove duplicates
        unique = list(set(expressions))

        # If only one unique term → return it
        if len(unique) == 1:
            return unique[0]

        # Otherwise return symbolic sum
        return " + ".join(unique)

   
    def _max_expression(self, expressions):
        if not expressions:
            return "1"

        # Prefer expression with most multiplicative factors
        return max(expressions, key=lambda x: x.count("*"))

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


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python complexity_extractor.py <target_file.py>")
        sys.exit(1)

    target_file = sys.argv[1]

    analyzer = ComplexityAnalyzer(target_file)
    report = analyzer.analyze()

    print("\n======================================")
    print(" AUTOMATIC COMPLEXITY ANALYSIS REPORT ")
    print("======================================\n")

    print(report)