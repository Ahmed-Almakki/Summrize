#!/bin/bash

echo "Starting installation of summrize_ahmed..."

pip install -e .

# dummy test file
echo "Artificial Intelligence is transforming the way we write software. 
By using local models, developers can ensure privacy and speed without 
relying on external APIs. This tool is a great example of that." > test_input.txt

echo "Test file created: test_input.txt"

# Run the new CLI command
echo "Running the summarizer..."
summarize -f test_input.txt -o test_output.txt

# Check the results
if [ -f test_output.txt ]; then
    echo "Success! Summary generated in test_output.txt:"
    cat test_output.txt
else
    echo "Error: Output file was not created."
fi