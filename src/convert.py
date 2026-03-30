import os
import glob
import time

def convert_single_file(word_file, output_folder):
    """Convert a single Word file to PDF"""
    filename = os.path.splitext(os.path.basename(word_file))[0]
    output_file = os.path.join(output_folder, f"{filename}.pdf")
    
    # Skip if PDF already exists
    if os.path.exists(output_file):
        print(f"Skipping {filename} (already exists)")
        return True
    
    script = f'''
tell application "Microsoft Word"
    open POSIX file "{os.path.abspath(word_file)}"
    set theDoc to document 1
    save as theDoc file name "{os.path.abspath(output_file)}" file format format PDF
    close theDoc
end tell
'''
    
    with open('temp_convert.scpt', 'w') as f:
        f.write(script)
    
    result = os.system('osascript temp_convert.scpt')
    time.sleep(0.5)  # Brief pause between conversions
    
    if result == 0:
        print(f"✓ Converted {filename}")
        return True
    else:
        print(f"✗ Failed to convert {filename}")
        return False

# Convert ALL files one by one
word_files = glob.glob("wikipedia_docs/*.docx")
os.makedirs("word_pdfs", exist_ok=True)

print(f"Converting {len(word_files)} Word documents to PDF...")
success_count = 0
failed_files = []

for i, word_file in enumerate(word_files):
    print(f"[{i+1}/{len(word_files)}] Processing {os.path.basename(word_file)}")
    if convert_single_file(word_file, "word_pdfs"):
        success_count += 1
    else:
        failed_files.append(word_file)

print(f"\nConversion complete!")
print(f"Successfully converted: {success_count}/{len(word_files)}")
if failed_files:
    print(f"Failed files: {[os.path.basename(f) for f in failed_files]}")

# Cleanup temp file
if os.path.exists('temp_convert.scpt'):
    os.remove('temp_convert.scpt')