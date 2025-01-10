import json
import re

def categorize_batter(style):
    """
    Take a batting-style string (e.g., 'Right-handed', 'Left-handed bat', etc.)
    and normalize it to 'Right Handed', 'Left Handed', or 'Unknown'.
    """
    # Convert to lowercase for easier matching
    style_lower = style.lower().strip()
    
    # Handle unknown or invalid entries
    if not style_lower or style_lower in ['unknown', '-', '–']:
        return 'Unknown'
    
    # Some typical patterns:
    # "Right-handed bat", "Right-hand bat", "Right handed batsman", etc.
    # "Left-handed bat", "Left-hand bat", "Left handed batsman", etc.
    # We'll look for 'right' or 'left' in the string.
    
    if 'right' in style_lower:
        return 'Right Handed'
    elif 'left' in style_lower:
        return 'Left Handed'
    else:
        return 'Unknown'

def main():
    input_file = 'batters_with_styles.json'
    output_file = 'batters_categorized.json'
    
    # Load the raw batting data
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Create a dictionary to hold cleaned results
    result = {}
    
    # Categorize each player's batting style
    for player, style in data.items():
        category = categorize_batter(style)
        result[player] = category
    
    # Save the categorized data
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=4, ensure_ascii=False)
    
    print(f"Batting categorization complete. Output saved to '{output_file}'.")

if __name__ == "__main__":
    main()
