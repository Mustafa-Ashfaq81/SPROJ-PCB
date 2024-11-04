import json
import re

# Function to categorize bowling styles
def categorize_bowler(style):
    style = style.lower()
    
    # Check for unknown or invalid entries
    if style in ['unknown', '-', '–'] or re.match(r'^\d+(\.\d+)?$', style.strip()):
        return 'Unknown'
    
    # Determine the arm
    if 'right' in style:
        arm = 'Right arm'
    elif 'left' in style:
        arm = 'Left arm'
    elif 'slow left-arm' in style:
        arm = 'Left arm'
    elif 'legbreak' in style or 'leg break' in style:
        # Default to right arm if "legbreak" is used without specifying arm
        arm = 'Right arm'
    else:
        arm = 'Unknown'
    
    # Determine the type
    if any(spin in style for spin in ['spin', 'offbreak', 'off-break', 'off break', 
                                      'legbreak', 'leg-break', 'leg break', 'leg-spin', 
                                      'leg spin', 'googly', 'orthodox', 'wrist']):
        type_ = 'Spin'
    elif any(pace in style for pace in ['fast', 'medium', 'pace', 'slow']):
        type_ = 'Pace'
    else:
        type_ = 'Unknown'
    
    # Combine arm and type
    if arm != 'Unknown' and type_ != 'Unknown':
        category = f"{arm} {type_}"
    else:
        category = 'Unknown'
    
    # Ensure category is one of the specified categories
    if category in ['Right arm Spin', 'Left arm Spin', 'Right arm Pace', 'Left arm Pace']:
        return category
    else:
        return 'Unknown'

# Load the JSON data from the file
with open('bowlers_with_styles.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Initialize the result dictionary
result = {}

# Process each bowler
for player, style in data.items():
    category = categorize_bowler(style)
    result[player] = category

# Output the result to a new JSON file
with open('bowlers_categorized.json', 'w', encoding='utf-8') as f:
    json.dump(result, f, indent=4)

print("Bowler categorization complete. Output saved to 'bowlers_categorized.json'.")
