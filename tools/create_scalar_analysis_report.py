"""
Mercury Agent ♱
Copyright (C) 2025 Steel Security Advisory LLC

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see https://www.gnu.org/licenses/.
"""

"""
Script to generate scalar analysis reports for Mercury Agent.

Analyzes ethical scalar distribution across categories and generates
reports for documentation purposes.

Note: This file was renamed from create_novelty_report.py.
"""

import re
from collections import defaultdict

# Read ethical config
with open("omni_mercury_engine/core/ethical_config.py") as f:
    ethical_content = f.read()

# Extract all scalars with values
scalar_pattern = r"(\w+):\s*float\s*=\s*([\d.]+)"
scalars = re.findall(scalar_pattern, ethical_content)

# Categorize by theme
categories = {
    "Ancient Wisdom": [
        "thoth",
        "maat",
        "athena",
        "hermes",
        "pharaonic",
        "hieroglyphic",
        "lunar",
        "cosmic",
        "isfet",
    ],
    "Modern AI Ethics": ["ai", "model", "rogue", "neurosymbolic", "agentic", "swarm", "recursive"],
    "Quantum/Physics": [
        "quantum",
        "dark",
        "black_hole",
        "gravitational",
        "harmonic",
        "coherence",
        "entanglement",
    ],
    "Mathematics": ["riemann", "collatz", "goldbach", "prime", "navier", "yang", "hodge", "birch"],
    "Humanitarian": [
        "compassion",
        "empathetic",
        "altruistic",
        "social",
        "peace",
        "disaster",
        "inequality",
    ],
    "Guardian/Protection": [
        "guard",
        "protection",
        "defense",
        "mitigation",
        "prevention",
        "biosecurity",
    ],
    "Wisdom/Intelligence": [
        "wisdom",
        "perspicacious",
        "sagacious",
        "logic",
        "reason",
        "rationality",
    ],
    "Ethical Core": [
        "justitia",
        "fairness",
        "transparency",
        "accountability",
        "integrity",
        "morality",
    ],
}

categorized = defaultdict(list)
for name, value in scalars:
    name_lower = name.lower()
    assigned = False
    for category, keywords in categories.items():
        if any(kw in name_lower for kw in keywords):
            categorized[category].append((name, float(value)))
            assigned = True
            break
    if not assigned:
        categorized["Other"].append((name, float(value)))

# Count by category
print("## Ethical Scalar Distribution\n")
for cat in sorted(categories.keys()):
    count = len(categorized[cat])
    if count > 0:
        avg_val = sum(v for _, v in categorized[cat]) / count
        print(f"- **{cat}**: {count} scalars (avg value: {avg_val:.3f})")

print(f"\n**Total Unique Scalars**: {len(scalars)}")

# Find high-value scalars (>1.40)
print("\n## High-Value Scalars (> 1.40)\n")
high_value = [(name, val) for name, val in scalars if val > 1.40]
for name, val in sorted(high_value, key=lambda x: x[1], reverse=True):
    print(f"- `{name}`: {val}")
