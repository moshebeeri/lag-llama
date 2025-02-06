#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Starting upstream sync process...${NC}"

# Store the current branch name
CURRENT_BRANCH=$(git branch --show-current)
echo -e "Current branch: ${GREEN}$CURRENT_BRANCH${NC}"

# Check if there are uncommitted changes
if ! git diff-index --quiet HEAD --; then
    echo -e "${RED}You have uncommitted changes. Please commit or stash them first.${NC}"
    exit 1
fi

# Fetch all branches from upstream
echo -e "\n${YELLOW}Fetching updates from upstream...${NC}"
if ! git fetch upstream; then
    echo -e "${RED}Failed to fetch from upstream. Please check your connection and upstream remote.${NC}"
    exit 1
fi

# Switch to main branch
echo -e "\n${YELLOW}Switching to main branch...${NC}"
if ! git checkout main; then
    echo -e "${RED}Failed to switch to main branch.${NC}"
    exit 1
fi

# Merge upstream/main into local main
echo -e "\n${YELLOW}Merging upstream changes...${NC}"
if ! git merge upstream/main; then
    echo -e "${RED}Merge conflicts detected. Please resolve them manually.${NC}"
    echo -e "After resolving conflicts, run: ${GREEN}git merge --continue${NC}"
    exit 1
fi

# Push changes to origin (your fork)
echo -e "\n${YELLOW}Pushing changes to your fork...${NC}"
if ! git push origin main; then
    echo -e "${RED}Failed to push to origin.${NC}"
    exit 1
fi

# Return to the original branch if it wasn't main
if [ "$CURRENT_BRANCH" != "main" ]; then
    echo -e "\n${YELLOW}Returning to branch $CURRENT_BRANCH...${NC}"
    if ! git checkout "$CURRENT_BRANCH"; then
        echo -e "${RED}Failed to return to original branch.${NC}"
        exit 1
    fi
    
    # Optionally rebase the current branch on main
    echo -e "\n${YELLOW}Would you like to rebase $CURRENT_BRANCH on main? [y/N]${NC}"
    read -r answer
    if [[ $answer =~ ^[Yy]$ ]]; then
        if ! git rebase main; then
            echo -e "${RED}Rebase conflicts detected. Please resolve them manually.${NC}"
            echo -e "After resolving conflicts, run: ${GREEN}git rebase --continue${NC}"
            exit 1
        fi
    fi
fi

echo -e "\n${GREEN}Sync completed successfully!${NC}"
echo -e "\nCurrent status:"
git status 