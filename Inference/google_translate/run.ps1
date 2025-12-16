# pip install -r requirements.txt

# Capture command line arguments
$inputFile = $args[0]
$targetLang = $args[1]

# Execute the Python script with the specified parameters
python run.py `
    --sleep 0.5 `
    --source_lang auto `
    --target_lang $targetLang `
    --input_file $inputFile `
    --output_file "$inputFile.google_translate.$targetLang" `
    --resume
