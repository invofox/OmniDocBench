#!/bin/bash

# Check if all data is declared in dataset.yaml
# data/ subfolders should be declared in dataset.yaml

DATA_DIR="data"
YAML_FILE="datasets.yaml"
MISSING_DIRS=()
# Read declared directories from dataset.yaml (principal keys of the file)
if [ ! -f "$YAML_FILE" ]; then
    echo "Error: $YAML_FILE not found"
    exit 1
fi
DECLARED_DIRS=($(grep '^[^ ]' $YAML_FILE | sed 's/:$//'))
echo "Declared directories in $YAML_FILE: ${DECLARED_DIRS[*]}"
# Loop through each subdirectory in the data/ directory
for dir in $DATA_DIR/*/; do
    dir_name=$(basename "$dir")
    if [[ ! " ${DECLARED_DIRS[@]} " =~ " ${dir_name} " ]]; then
        MISSING_DIRS+=("$dir_name")
    fi
done
# Report results
if [ ${#MISSING_DIRS[@]} -eq 0 ]; then
    echo "All data directories are declared in $YAML_FILE."
else
    echo "The following data directories are missing in $YAML_FILE:"
    for missing in "${MISSING_DIRS[@]}"; do
        echo "- $missing"
    done
    exit 1
fi

# Check if there are any uncommitted changes in DVC
result=$(find data/ -name '*.dvc' | xargs pdm run dvc data status --not-in-remote | xargs)
if [ -z "$result" ]; then
    result="No changes."
fi
if [[ "$result" == "No changes." ]]; then
    echo "All DVC files are up to date."
    exit 0
fi
expected_result='No changes. (there are changes not tracked by dvc, use git status to see)'
if [[ "$result" == "$expected_result" ]]; then
    # Check if there are changes in data/ without staging
    git_result=$(git status --porcelain data/ | xargs)
    # Delete added files from git_result
    git_result=$(echo "$git_result" | grep -v '^A' | xargs)
    if [ -z "$git_result" ]; then
        echo "All DVC files are up to date."
        exit 0
    else
        result=$git_result
    fi
    echo "There are uncommitted changes in data folder:"
    echo "$result"
    echo ""
    echo "Please add the changes to commit using 'pdm run data-push' or discard them using 'pdm run data-pull'."
    echo "Also, double-check that this file is expected and declared in datasets.yaml."
    exit 1
fi
echo "There are uncommitted changes in DVC:"
echo "$result"
echo ""
echo "Please add the changes to commit using 'pdm run data-push' or discard them using 'pdm run data-pull'."
exit 1
