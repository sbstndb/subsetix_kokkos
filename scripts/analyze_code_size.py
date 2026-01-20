#!/usr/bin/env python3
"""
Code Size Analysis Script for Subsetix Kokkos

Analyzes C++ code to quantify the productivity improvement of the FVD layer.
Computes metrics like lines of code, cyclomatic complexity, and function counts.

Usage:
    python scripts/analyze_code_size.py <file1.cpp> [file2.cpp ...]

Example:
    python scripts/analyze_code_size.py \
        examples/mach2_cylinder/mach2_cylinder.cpp \
        examples/mach2_cylinder_simplified.cpp
"""

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class CodeMetrics:
    """Container for code analysis metrics."""
    file_path: str = ""
    total_lines: int = 0
    code_lines: int = 0
    comment_lines: int = 0
    blank_lines: int = 0
    preprocessor_lines: int = 0
    include_lines: int = 0

    num_functions: int = 0
    num_templates: int = 0
    num_namespaces: int = 0
    num_classes: int = 0
    num_structs: int = 0

    cyclomatic_complexity: int = 0
    nesting_depth: int = 0

    function_counts: Dict[str, int] = field(default_factory=dict)
    complexity_by_function: Dict[str, int] = field(default_factory=dict)

    # Language feature counts
    num_loops: int = 0
    num_conditionals: int = 0
    num_lambda: int = 0
    num_kokkos_parallel: int = 0

    def __add__(self, other: 'CodeMetrics') -> 'CodeMetrics':
        """Combine metrics from multiple files."""
        result = CodeMetrics()
        result.file_path = f"{self.file_path} + {other.file_path}"
        result.total_lines = self.total_lines + other.total_lines
        result.code_lines = self.code_lines + other.code_lines
        result.comment_lines = self.comment_lines + other.comment_lines
        result.blank_lines = self.blank_lines + other.blank_lines
        result.preprocessor_lines = self.preprocessor_lines + other.preprocessor_lines
        result.include_lines = self.include_lines + other.include_lines

        result.num_functions = self.num_functions + other.num_functions
        result.num_templates = self.num_templates + other.num_templates
        result.num_namespaces = self.num_namespaces + other.num_namespaces
        result.num_classes = self.num_classes + other.num_classes
        result.num_structs = self.num_structs + other.num_structs

        result.cyclomatic_complexity = (
            self.cyclomatic_complexity + other.cyclomatic_complexity
        )
        result.nesting_depth = max(self.nesting_depth, other.nesting_depth)

        for k, v in self.function_counts.items():
            result.function_counts[k] = result.function_counts.get(k, 0) + v
        for k, v in other.function_counts.items():
            result.function_counts[k] = result.function_counts.get(k, 0) + v

        for k, v in self.complexity_by_function.items():
            result.complexity_by_function[k] = v
        for k, v in other.complexity_by_function.items():
            result.complexity_by_function[k] = v

        result.num_loops = self.num_loops + other.num_loops
        result.num_conditionals = self.num_conditionals + other.num_conditionals
        result.num_lambda = self.num_lambda + other.num_lambda
        result.num_kokkos_parallel = (
            self.num_kokkos_parallel + other.num_kokkos_parallel
        )

        return result


class CppAnalyzer:
    """Analyzes C++ source code for complexity and size metrics."""

    # Patterns for line classification
    COMMENT_PATTERN = re.compile(r'^\s*//|^\s*/\*|\*/')
    BLANK_PATTERN = re.compile(r'^\s*$')
    INCLUDE_PATTERN = re.compile(r'^\s*#\s*include')
    PREPROCESSOR_PATTERN = re.compile(r'^\s*#')
    NAMESPACE_PATTERN = re.compile(r'^\s*namespace\s+(\w+)\s*{?')
    CLASS_PATTERN = re.compile(r'^\s*class\s+(\w+)')
    STRUCT_PATTERN = re.compile(r'^\s*struct\s+(\w+)')
    TEMPLATE_PATTERN = re.compile(r'^\s*template\s*<')
    FUNCTION_PATTERN = re.compile(
        r'^\s*(?:template\s*<[^>]*>\s*)?'  # Optional template
        r'(?:\w+\s*::)*?'  # Optional return type with namespace
        r'(?:\w+(?:<[^>]+>)?\s+)'  # Return type
        r'(?:\*?\s*)?'  # Pointer
        r'(\w+)\s*\('  # Function name
    )

    # Patterns for complexity analysis
    LOOP_PATTERNS = [
        re.compile(r'\bfor\s*\('),
        re.compile(r'\bwhile\s*\('),
        re.compile(r'\bdo\s*{'),
    ]

    CONDITIONAL_PATTERNS = [
        re.compile(r'\bif\s*\('),
        re.compile(r'\belse\s+if\s*\('),
        re.compile(r'\belse\s*{?'),
        re.compile(r'\bswitch\s*\('),
        re.compile(r'\bcase\s+'),
        re.compile(r'\bdefault\s*:'),
        re.compile(r'\?[^:]*:'),  # Ternary operator
    ]

    KOKKOS_PATTERNS = [
        re.compile(r'\bKOKKOS_'),
        re.compile(r'\bKokkos::'),
    ]

    LAMBDA_PATTERN = re.compile(r'\[.*?\]\s*\(')

    def __init__(self):
        self.current_function = None
        self.function_complexity = defaultdict(int)
        self.max_nesting = 0
        self.current_nesting = 0

    def analyze_file(self, file_path: str) -> CodeMetrics:
        """Analyze a single C++ file and return metrics."""
        metrics = CodeMetrics()
        metrics.file_path = file_path

        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
        except Exception as e:
            print(f"Warning: Could not read {file_path}: {e}", file=sys.stderr)
            return metrics

        metrics.total_lines = len(lines)

        in_block_comment = False
        in_function = False
        brace_count = 0

        for line_num, line in enumerate(lines, 1):
            # Track block comments
            if '/*' in line:
                in_block_comment = True
            if '*/' in line:
                in_block_comment = False
                metrics.comment_lines += 1
                continue

            if in_block_comment:
                metrics.comment_lines += 1
                continue

            # Check line type
            if self.BLANK_PATTERN.match(line):
                metrics.blank_lines += 1
                continue

            if line.strip().startswith('//'):
                metrics.comment_lines += 1
                continue

            if self.INCLUDE_PATTERN.match(line):
                metrics.include_lines += 1
                metrics.preprocessor_lines += 1
                continue

            if self.PREPROCESSOR_PATTERN.match(line):
                metrics.preprocessor_lines += 1
                continue

            # Code line
            metrics.code_lines += 1

            # Count structures
            if self.NAMESPACE_PATTERN.search(line):
                metrics.num_namespaces += 1
            if self.CLASS_PATTERN.search(line):
                metrics.num_classes += 1
            if self.STRUCT_PATTERN.search(line):
                metrics.num_structs += 1
            if self.TEMPLATE_PATTERN.search(line):
                metrics.num_templates += 1

            # Count function definitions
            func_match = self.FUNCTION_PATTERN.match(line)
            if func_match and '{' in line:
                func_name = func_match.group(1)
                # Filter out common non-function keywords
                if func_name not in ['if', 'for', 'while', 'switch', 'catch']:
                    metrics.num_functions += 1
                    metrics.function_counts[func_name] = (
                        metrics.function_counts.get(func_name, 0) + 1
                    )
                    in_function = True
                    self.current_function = func_name
                    brace_count = line.count('{') - line.count('}')

            # Track function scope
            if in_function:
                brace_count += line.count('{') - line.count('}')
                if brace_count == 0:
                    in_function = False
                    self.current_function = None

            # Count complexity features
            for pattern in self.LOOP_PATTERNS:
                if pattern.search(line):
                    metrics.num_loops += 1
                    if self.current_function:
                        self.function_complexity[self.current_function] += 1

            for pattern in self.CONDITIONAL_PATTERNS:
                if pattern.search(line):
                    metrics.num_conditionals += 1
                    if self.current_function:
                        self.function_complexity[self.current_function] += 1

            if self.LAMBDA_PATTERN.search(line):
                metrics.num_lambda += 1
                if self.current_function:
                    self.function_complexity[self.current_function] += 2  # Lambdas add complexity

            for pattern in self.KOKKOS_PATTERNS:
                if pattern.search(line):
                    metrics.num_kokkos_parallel += 1

        # Calculate cyclomatic complexity (simplified)
        # CC = 1 + number of decisions
        metrics.cyclomatic_complexity = (
            1 + metrics.num_loops + metrics.num_conditionals
        )

        # Store per-function complexity
        metrics.complexity_by_function = dict(self.function_complexity)

        # Estimate max nesting depth
        max_open_braces = 0
        current_braces = 0
        for line in lines:
            if not in_block_comment and not line.strip().startswith('//'):
                current_braces += line.count('{')
                current_braces -= line.count('}')
                max_open_braces = max(max_open_braces, current_braces)
        metrics.nesting_depth = max_open_braces

        return metrics


def format_number(num: int) -> str:
    """Format number with thousands separator."""
    return f"{num:,}"


def calculate_percentage_reduction(old: int, new: int) -> float:
    """Calculate percentage reduction from old to new."""
    if old == 0:
        return 0.0
    return ((old - new) / old) * 100


def print_metrics(metrics: CodeMetrics, label: str = ""):
    """Print metrics for a single file or aggregate."""
    if label:
        print(f"\n{'=' * 70}")
        print(f"{label}")
        print(f"{'=' * 70}")

    print(f"File: {metrics.file_path}")
    print(f"\nSize Metrics:")
    print(f"  Total Lines:           {format_number(metrics.total_lines):>10}")
    print(f"  Code Lines:            {format_number(metrics.code_lines):>10}")
    print(f"  Comment Lines:         {format_number(metrics.comment_lines):>10}")
    print(f"  Blank Lines:           {format_number(metrics.blank_lines):>10}")
    print(f"  Preprocessor Lines:    {format_number(metrics.preprocessor_lines):>10}")

    print(f"\nStructure Metrics:")
    print(f"  Functions:             {format_number(metrics.num_functions):>10}")
    print(f"  Templates:             {format_number(metrics.num_templates):>10}")
    print(f"  Namespaces:            {format_number(metrics.num_namespaces):>10}")
    print(f"  Classes:               {format_number(metrics.num_classes):>10}")
    print(f"  Structs:               {format_number(metrics.num_structs):>10}")

    print(f"\nComplexity Metrics:")
    print(f"  Cyclomatic Complexity: {format_number(metrics.cyclomatic_complexity):>10}")
    print(f"  Max Nesting Depth:     {format_number(metrics.nesting_depth):>10}")
    print(f"  Loops:                 {format_number(metrics.num_loops):>10}")
    print(f"  Conditionals:          {format_number(metrics.num_conditionals):>10}")
    print(f"  Lambdas:               {format_number(metrics.num_lambda):>10}")
    print(f"  Kokkos Constructs:     {format_number(metrics.num_kokkos_parallel):>10}")


def print_comparison(original: CodeMetrics, simplified: CodeMetrics):
    """Print comparison between original and simplified versions."""
    print(f"\n{'=' * 70}")
    print("COMPARISON: Original vs Simplified")
    print(f"{'=' * 70}")

    print(f"\n{'Metric':<30} {'Original':>12} {'Simplified':>12} {'Reduction':>12}")
    print("-" * 70)

    metrics_to_compare = [
        ("Total Lines", original.total_lines, simplified.total_lines),
        ("Code Lines", original.code_lines, simplified.code_lines),
        ("Functions", original.num_functions, simplified.num_functions),
        ("Templates", original.num_templates, simplified.num_templates),
        ("Cyclomatic Complexity", original.cyclomatic_complexity,
         simplified.cyclomatic_complexity),
        ("Loops", original.num_loops, simplified.num_loops),
        ("Conditionals", original.num_conditionals, simplified.num_conditionals),
        ("Lambdas", original.num_lambda, simplified.num_lambda),
        ("Nesting Depth", original.nesting_depth, simplified.nesting_depth),
    ]

    for name, old, new in metrics_to_compare:
        reduction = calculate_percentage_reduction(old, new)
        reduction_str = f"-{reduction:.1f}%" if reduction > 0 else f"+{abs(reduction):.1f}%"
        print(f"{name:<30} {format_number(old):>12} {format_number(new):>12} {reduction_str:>12}")

    # Calculate overall productivity improvement
    print(f"\n{'=' * 70}")
    print("PRODUCTIVITY ANALYSIS")
    print(f"{'=' * 70}")

    code_reduction = calculate_percentage_reduction(
        original.code_lines, simplified.code_lines
    )
    complexity_reduction = calculate_percentage_reduction(
        original.cyclomatic_complexity, simplified.cyclomatic_complexity
    )

    print(f"\nCode Reduction:          {code_reduction:.1f}%")
    print(f"Complexity Reduction:    {complexity_reduction:.1f}%")

    # Time-to-add-feature estimates (based on typical developer productivity)
    # Assumption: ~10 lines of working code per hour for complex scientific code
    dev_hours_original = original.code_lines / 10.0
    dev_hours_simplified = simplified.code_lines / 10.0

    print(f"\nEstimated Development Time:")
    print(f"  Original approach:     {dev_hours_original:.1f} hours")
    print(f"  Simplified approach:   {dev_hours_simplified:.1f} hours")
    print(f"  Time saved:            {dev_hours_original - dev_hours_simplified:.1f} hours")

    # Lines of code per feature (assuming 1 main feature per file)
    print(f"\nLines per Feature:")
    print(f"  Original:              {format_number(original.code_lines)} LOC")
    print(f"  Simplified:            {format_number(simplified.code_lines)} LOC")
    print(f"  Ratio:                 {original.code_lines / simplified.code_lines:.2f}x reduction")


def print_markdown_table(original: CodeMetrics, simplified: CodeMetrics):
    """Print comparison as a markdown table."""
    print(f"\n## Quantitative Comparison\n")
    print(f"| Metric | Original | Simplified | Reduction |")
    print(f"|--------|----------|------------|-----------|")

    metrics_to_compare = [
        ("Total Lines", original.total_lines, simplified.total_lines),
        ("Code Lines", original.code_lines, simplified.code_lines),
        ("Functions", original.num_functions, simplified.num_functions),
        ("Templates", original.num_templates, simplified.num_templates),
        ("Cyclomatic Complexity", original.cyclomatic_complexity,
         simplified.cyclomatic_complexity),
        ("Loops", original.num_loops, simplified.num_loops),
        ("Conditionals", original.num_conditionals, simplified.num_conditionals),
        ("Lambdas", original.num_lambda, simplified.num_lambda),
        ("Nesting Depth", original.nesting_depth, simplified.nesting_depth),
    ]

    for name, old, new in metrics_to_compare:
        reduction = calculate_percentage_reduction(old, new)
        reduction_str = f"-{reduction:.1f}%" if reduction > 0 else f"+{abs(reduction):.1f}%"
        print(f"| {name} | {format_number(old)} | {format_number(new)} | {reduction_str} |")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze C++ code size and complexity",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze single file
  python scripts/analyze_code_size.py examples/mach2_cylinder/mach2_cylinder.cpp

  # Compare two files
  python scripts/analyze_code_size.py \\
      examples/mach2_cylinder/mach2_cylinder.cpp \\
      examples/mach2_cylinder_simplified.cpp

  # Analyze multiple files
  python scripts/analyze_code_size.py src/*.cpp
        """
    )

    parser.add_argument(
        'files',
        nargs='+',
        help='C++ source files to analyze'
    )

    parser.add_argument(
        '--markdown',
        action='store_true',
        help='Output in markdown table format'
    )

    parser.add_argument(
        '--compare',
        action='store_true',
        help='Compare first two files as before/after'
    )

    args = parser.parse_args()

    analyzer = CppAnalyzer()
    all_metrics = []

    for file_path in args.files:
        path = Path(file_path)
        if not path.exists():
            print(f"Warning: File not found: {file_path}", file=sys.stderr)
            continue

        metrics = analyzer.analyze_file(str(path))
        all_metrics.append(metrics)

    if not all_metrics:
        print("Error: No valid files to analyze", file=sys.stderr)
        return 1

    if args.compare and len(all_metrics) >= 2:
        # Comparison mode
        if args.markdown:
            print_markdown_table(all_metrics[0], all_metrics[1])
        else:
            print_metrics(all_metrics[0], "ORIGINAL IMPLEMENTATION")
            print_metrics(all_metrics[1], "SIMPLIFIED IMPLEMENTATION")
            print_comparison(all_metrics[0], all_metrics[1])
    elif args.markdown:
        # Markdown output for all files
        for metrics in all_metrics:
            print(f"\n## {metrics.file_path}")
            print_markdown_table(metrics, CodeMetrics())
    else:
        # Standard output
        for i, metrics in enumerate(all_metrics):
            label = f"FILE {i + 1}/{len(all_metrics)}"
            print_metrics(metrics, label)

        if len(all_metrics) > 2:
            # Aggregate statistics
            aggregate = CodeMetrics()
            for metrics in all_metrics:
                aggregate = aggregate + metrics
            print_metrics(aggregate, f"\nAGGREGATE ({len(all_metrics)} files)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
