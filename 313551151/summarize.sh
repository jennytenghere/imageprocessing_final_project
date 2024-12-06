#!/bin/bash

# Script: summarize_results.sh
# Description: Extracts Mean loss per pixel and Querying time from result.log in output directories.
# Usage: ./summarize_results.sh
# Make sure to give execute permissions: chmod +x summarize_results.sh

# -------------------------------
# Initialize Header with Fixed Width
# -------------------------------

# Define column widths
DIR_WIDTH=40
LOSS_WIDTH=25
TIME_WIDTH=25

# Print the header with fixed widths
printf "%-${DIR_WIDTH}s %-${LOSS_WIDTH}s %-${TIME_WIDTH}s\n" "Directory_Name" "Mean_Loss_Per_Pixel" "Querying_Time(seconds)"
printf "%-${DIR_WIDTH}s %-${LOSS_WIDTH}s %-${TIME_WIDTH}s\n" "--------------" "--------------------" "---------------------"

# -------------------------------
# Find and Sort "output" Directories
# -------------------------------

# Find all directories in the current directory starting with "output" and sort them
# Using globbing instead of find for simplicity and better sorting
output_dirs=(output*/)

# Check if any "output" directories were found
if [ ${#output_dirs[@]} -eq 0 ]; then
    echo "No directories starting with 'output' found in the current directory."
    exit 1
fi

# Sort the directories alphabetically
IFS=$'\n' sorted_dirs=($(sort <<<"${output_dirs[*]}"))
unset IFS

# -------------------------------
# Loop Through Each Sorted "output" Directory
# -------------------------------

for dir in "${sorted_dirs[@]}"; do
    # Remove the trailing slash and extract the base directory name
    dir_name=$(basename "$dir")
    
    # Path to the result.log file
    log_file="${dir}result.log"
    
    # Initialize variables
    mean_loss="N/A"
    querying_time="N/A"
    
    # Check if result.log exists
    if [ -f "$log_file" ]; then
        # Extract Mean loss per pixel using grep and awk
        mean_loss=$(grep -E "Mean loss per pixel:" "$log_file" | awk '{print $5}')
        
        # Extract Querying time using grep and awk
        querying_time=$(grep -E "Querying time:" "$log_file" | awk '{print $3}')
        
        # Assign "N/A" if extraction fails
        if [ -z "$mean_loss" ]; then
            mean_loss="N/A"
        fi
        
        if [ -z "$querying_time" ]; then
            querying_time="N/A"
        fi
    else
        echo "Warning: $log_file not found."
    fi
    
    # Print the extracted values with fixed widths
    printf "%-${DIR_WIDTH}s %-${LOSS_WIDTH}s %-${TIME_WIDTH}s\n" "$dir_name" "$mean_loss" "$querying_time"
done

