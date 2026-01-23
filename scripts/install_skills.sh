#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024 Sebastien DUBOIS and the HPC@Maths Team, CMAP Laboratory, Ecole Polytechnique

set -e

# =============================================================================
# Install subsetix-optim Claude Code Skills
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CLAUDE_SKILLS_DIR="$PROJECT_ROOT/.claude/skills"
LOCAL_CLAUDE_DIR="$HOME/.claude"
LOCAL_SKILLS_DIR="$LOCAL_CLAUDE_DIR/skills"

echo "=========================================="
echo "  Subsetix Optim Skills Installer"
echo "=========================================="
echo ""
echo "Project skills: $CLAUDE_SKILLS_DIR"
echo "Target directory: $LOCAL_SKILLS_DIR"
echo ""

# Check if project skills exist
if [ ! -d "$CLAUDE_SKILLS_DIR" ]; then
  echo "ERROR: Project skills directory not found: $CLAUDE_SKILLS_DIR"
  exit 1
fi

# Count available skills
SKILL_COUNT=$(find "$CLAUDE_SKILLS_DIR" -name "SKILL.md" | wc -l)
echo "Found $SKILL_COUNT optimization skills"
echo ""

# Create local skills directory if it doesn't exist
mkdir -p "$LOCAL_SKILLS_DIR"

# Install each skill
echo "Installing skills..."
echo ""

for skill_dir in "$CLAUDE_SKILLS_DIR"/*; do
  if [ -d "$skill_dir" ] && [ -f "$skill_dir/SKILL.md" ]; then
    skill_name=$(basename "$skill_dir")
    target_dir="$LOCAL_SKILLS_DIR/$skill_name"

    echo "→ Installing $skill_name"

    # Create target directory
    mkdir -p "$target_dir"

    # Copy SKILL.md
    cp "$skill_dir/SKILL.md" "$target_dir/SKILL.md"

    # Copy any additional files if present
    if [ -f "$skill_dir/*.sh" ]; then
      cp "$skill_dir"/*.sh "$target_dir/" 2>/dev/null || true
    fi

    echo "  ✓ Installed to $target_dir"
  fi
done

echo ""
echo "=========================================="
echo "  Installation Complete!"
echo "=========================================="
echo ""
echo "Installed $SKILL_COUNT skills to: $LOCAL_SKILLS_DIR"
echo ""
echo "Available skills:"
for skill_dir in "$LOCAL_SKILLS_DIR"/*; do
  if [ -f "$skill_dir/SKILL.md" ]; then
    name=$(grep "^name:" "$skill_dir/SKILL.md" | cut -d: -f2 | xargs)
    desc=$(grep "^description:" "$skill_dir/SKILL.md" | cut -d: -f2- | xargs | head -c 80)
    echo "  - /${name}"
    echo "    $desc..."
  fi
done
echo ""
echo "Usage examples:"
echo "  /optim-orchestrator 24 4 4 1800 ./optim_logs"
echo "  /optim-benchmark 24 20260123_143000 10 2 \"3D_Large\""
echo "  /optim-antitriche 24 20260123_143000"
echo "  /optim-report 24 20260123_143000"
echo ""
echo "To uninstall skills:"
echo "  rm -rf $LOCAL_SKILLS_DIR/optim-*"
echo ""
