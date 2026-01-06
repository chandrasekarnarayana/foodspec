#!/bin/bash
# Rollback Script for mkdocs.yml Navigation Changes
# Usage: bash rollback_mkdocs_nav.sh

set -e  # Exit on error

echo "================================"
echo "MkDocs Navigation Rollback Script"
echo "================================"
echo ""

# Check if in FoodSpec directory
if [ ! -f "mkdocs.yml" ]; then
    echo "❌ Error: mkdocs.yml not found in current directory"
    echo "Please run this script from the FoodSpec root directory"
    exit 1
fi

# Show current status
echo "📋 Current Git Status:"
git status --short mkdocs.yml
echo ""

# Ask for confirmation
read -p "⚠️  This will revert mkdocs.yml to the last committed version. Continue? (y/n) " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "⏮️  Rolling back mkdocs.yml..."
    git checkout mkdocs.yml
    
    echo "✅ Rollback complete"
    echo ""
    
    # Verify rollback
    echo "🔍 Verifying mkdocs.yml is restored..."
    git diff mkdocs.yml
    
    if [ $? -eq 0 ]; then
        echo "✅ mkdocs.yml successfully reverted"
        echo ""
        echo "Next steps:"
        echo "  1. Run: mkdocs build --strict"
        echo "  2. Run: mkdocs serve"
        echo "  3. Visit: http://localhost:8000"
    fi
else
    echo "❌ Rollback cancelled"
    exit 1
fi
