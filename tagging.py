import spacy
from pathlib import Path

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Define input and output directories
input_dir = Path("/home/haozhe/CamTraj/momask-codes/dataset/RealEstate10K_rotmat1/untagged_text")
output_dir = Path("/home/haozhe/CamTraj/momask-codes/dataset/RealEstate10K_rotmat1/texts")
output_dir.mkdir(exist_ok=True)

# Process each .txt file in the input directory
for file_path in input_dir.glob("*.txt"):
    # Read the text file
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    # Process text with spaCy
    doc = nlp(text)

    # Prepare output file
    output_file = output_dir / f"{file_path.stem}.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        # Process the entire paragraph as one unit
        # Get the original text (strip whitespace but keep as paragraph)
        original_paragraph = text.strip()
        # Generate tags for the entire document, skipping punctuation
        tags = [f"{token.text}/{token.pos_}" for token in doc if not token.is_punct]
        # Write the paragraph and its tags as a single line
        f.write(f"{original_paragraph}#{' '.join(tags)}\n")

print("Tagging complete. Output files saved in 'text' directory.")