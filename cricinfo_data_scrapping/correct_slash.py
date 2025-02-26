import os
import json

# Directory containing JSON files
directory = "match_scorecards_json"

# List of matching files from your previous script
matching_files = ['match_1388394.json', 'match_1388397.json', 'match_1388398.json', 'match_1388400.json', 'match_1388401.json', 'match_1388403.json', 'match_1388406.json', 'match_1388407.json', 'match_1388410.json', 'match_1388412.json', 'match_1388414.json', 'match_433558.json', 'match_433559.json', 'match_433560.json', 'match_433561.json', 'match_433562.json', 'match_433563.json', 'match_433564.json', 'match_433565.json', 'match_433566.json', 'match_433567.json', 'match_433568.json', 'match_433569.json', 'match_433570.json', 'match_433571.json', 'match_433572.json', 'match_433573.json', 'match_433574.json', 'match_433575.json', 'match_433576.json', 'match_433577.json', 'match_433578.json', 'match_433579.json', 'match_433580.json', 'match_433581.json', 'match_433582.json', 'match_433583.json', 'match_433584.json', 'match_433585.json', 'match_433586.json', 'match_433587.json', 'match_433588.json', 'match_433589.json', 'match_433590.json', 'match_433591.json', 'match_433592.json', 'match_433593.json', 'match_433594.json', 'match_433595.json', 'match_433596.json', 'match_433597.json', 'match_433598.json', 'match_433599.json', 'match_433600.json', 'match_433601.json', 'match_433602.json', 'match_433603.json', 'match_433604.json', 'match_433605.json', 'match_433606.json', 'match_65235.json', 'match_65236.json', 'match_65237.json', 'match_65238.json', 'match_65239.json', 'match_65240.json', 'match_65241.json', 'match_65242.json', 'match_65243.json', 'match_65244.json', 'match_65245.json', 'match_65246.json', 'match_65247.json', 'match_65248.json', 'match_65249.json', 'match_65250.json', 'match_65251.json', 'match_65252.json', 'match_65253.json', 'match_65254.json', 'match_65255.json', 'match_65256.json', 'match_65257.json', 'match_65258.json', 'match_65259.json', 'match_65260.json', 'match_65261.json', 'match_65262.json', 'match_65263.json', 'match_65264.json', 'match_65265.json', 'match_65266.json', 'match_65267.json', 'match_65268.json', 'match_65269.json', 'match_65270.json', 'match_65271.json', 'match_65272.json', 'match_65273.json', 'match_65274.json', 'match_65275.json', 'match_65276.json', 'match_65277.json', 'match_65278.json', 'match_65279.json', 'match_65280.json', 'match_65281.json', 'match_65282.json', 'match_65283.json', 'match_65284.json', 'match_65285.json', 'match_65286.json', 'match_656399.json', 'match_656401.json', 'match_656403.json', 'match_656405.json', 'match_656407.json', 'match_656409.json', 'match_656411.json', 'match_656413.json', 'match_656415.json', 'match_656417.json', 'match_656419.json', 'match_656421.json', 'match_656423.json', 'match_656425.json', 'match_656427.json', 'match_656429.json', 'match_656431.json', 'match_656433.json', 'match_656435.json', 'match_656437.json', 'match_656439.json', 'match_656441.json', 'match_656443.json', 'match_656445.json', 'match_656447.json', 'match_656449.json', 'match_656451.json', 'match_656453.json', 'match_656455.json', 'match_656457.json', 'match_656459.json', 'match_656461.json', 'match_656463.json', 'match_656465.json', 'match_656467.json', 'match_656469.json', 'match_656471.json', 'match_656473.json', 'match_656475.json', 'match_656477.json', 'match_656479.json', 'match_656481.json', 'match_656483.json', 'match_656485.json', 'match_656487.json', 'match_656489.json', 'match_656491.json', 'match_656493.json', 'match_656495.json', 'match_66427.json', 'match_66428.json']

for filename in matching_files:
    filepath = os.path.join(directory, filename)
    
    with open(filepath, 'r', encoding='utf-8') as file:
        try:
            data = json.load(file)  # Load JSON
        except json.JSONDecodeError:
            print(f"Skipping invalid JSON: {filename}")
            continue

    # Extract necessary fields
    series_name = data.get("series_name", "N/A")
    date = data.get("match_details", {}).get("date", "N/A")

    print(f"\nEditing file: {filename}")
    print(f"📅 Match Date: {date}")
    print(f"🏆 Current Series Name: {series_name}")

    # Ask user for modification
    new_series_name = input("Enter new series_name (or press Enter to keep unchanged): ").strip()

    # Update and save if there's a change
    if new_series_name and new_series_name != series_name:
        data["series_name"] = new_series_name
        with open(filepath, 'w', encoding='utf-8') as file:
            json.dump(data, file, indent=4)
        print(f"✅ Updated and saved: {filename}")
    else:
        print("No changes made.")

print("\n🎉 All files processed!")
